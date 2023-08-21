from math import ceil
import numpy as np
from report import SimulationReport
from hardware_models.utils import *


class MultiCoresPredictor:
    def __init__(
        self,
        n_pes,
        pe_fp32s,
        frequency,
        mem_bandwidth,
        fp32_cycles=4,
        mem_concurrent_fp32=32 / 4,
        memory_efficiency=0.9,
        l2_speed_frac=4,
        latency_mode="max",
        verbose=False,
        batch_size=1,
        launch_time=8e-6,
    ) -> None:
        """
        frequency: Hz
        mem_bandwidth: B/s
        l2_speed_frac: specifies the bandwidth of the L2 cache as a multiple of the mem_bandwidth
        launch_time: launch kernel time in seconds, default to 8us
        """
        self.n_pes = n_pes
        self.pe_fp32s = pe_fp32s
        self.fp32_cycles = fp32_cycles
        self.frequency = frequency  # in Hz
        self.mem_concurrent_fp32 = mem_concurrent_fp32  # sector

        self.mem_bandwidth = mem_bandwidth  # in B/s
        self.mem_fp32_bandwidth = mem_bandwidth / 4  # in fp32/s
        self.l2_fp32_bandwidth = self.mem_fp32_bandwidth * l2_speed_frac  # in fp32/s
        self.memory_efficiency = memory_efficiency
        self.verbose = verbose
        self.latency_mode = latency_mode  # add or max
        self.batch_size = batch_size
        self.launch_time = launch_time

    def pe_reduce_compute_latency(self, c_parallel, n_elements):
        n = ceil(n_elements / 2)
        latency = 0
        while n > 1:
            efficiency = ceil_efficiency(
                n * c_parallel, self.pe_fp32s * self.fp32_cycles
            )
            latency += (
                ceil(n * c_parallel / self.pe_fp32s) / efficiency / self.frequency
            )
            n = ceil(n / 2)
        return latency * self.batch_size

    def calc_memory_latency(
        self, all_pe_memory, tot_fused_memory, req_size, req_interval
    ):
        global_latency = tot_fused_memory / self.mem_fp32_bandwidth
        l2_efficiency = mem_concurrent_efficiency(
            req_size, req_interval, self.mem_concurrent_fp32
        )
        l2_latency = (all_pe_memory) / self.l2_fp32_bandwidth / l2_efficiency
        return global_latency + l2_latency

    def search_conv_tile_cfg(
        self,
        cout,
        cin,
        outh,
        outw,
        groups,
        stride,
        ks,
        search_upper=8,
        ic_density=1,
        oc_density=1,
        c_n_groups=1,
    ):
        best_latency = None
        best_tile = None
        best_info = None
        peak_parallelism = self.pe_fp32s * self.fp32_cycles
        for c_tile in pow2_div_tile_search_space(cout, pow2_upper=search_upper):
            n_c_tile = ceil(cout / c_tile)
            for h_tile in pow2_div_tile_search_space(outh, pow2_upper=search_upper):
                n_h_tile = ceil(outh / h_tile)
                for w_tile in pow2_div_tile_search_space(outw, pow2_upper=search_upper):
                    n_w_tile = ceil(outw / w_tile)
                    n_tiles = n_c_tile * n_h_tile * n_w_tile

                    pe_weight = c_tile * (cin // groups) * ks * ks
                    n_groups = ceil(c_tile / (cout // groups))
                    pe_input = (
                        n_groups
                        * (cin // groups)
                        * (h_tile + ks - 1)
                        * stride
                        * (w_tile + ks - 1)
                        * stride
                    )
                    pe_output = c_tile * h_tile * w_tile
                    if self.batch_size == 1:
                        pe_weight *= ic_density * oc_density
                    all_pe_memory = (
                        pe_weight
                        + pe_input * ic_density * self.batch_size
                        + pe_output * oc_density * self.batch_size
                    ) * n_tiles
                    # weight all load into L2 and each PE fetch their required weight
                    all_weight = cout * (cin // groups) * ks * ks
                    if self.batch_size == 1:
                        all_weight *= oc_density
                    all_input = (
                        cin
                        * outh
                        * stride
                        * outw
                        * stride
                        * self.batch_size
                        * ic_density
                    )
                    all_output = cout * outh * outw * self.batch_size * oc_density
                    tot_fused_memory = all_weight + all_input + all_output
                    mem_latency = self.calc_memory_latency(
                        all_pe_memory,
                        tot_fused_memory,
                        req_interval=outw - w_tile,
                        req_size=w_tile,
                    )
                    compute = (
                        c_tile
                        * h_tile
                        * w_tile
                        * (cin // groups)
                        * ks
                        * ks
                        * self.batch_size
                    )
                    max_oc_density = calc_max_c_density(
                        n_c_tile, c_tile, ic_density * oc_density, c_n_groups, n=100
                    )
                    pe_compute_latency = (
                        compute
                        / self.frequency
                        / self.pe_fp32s
                        * ic_density
                        * max_oc_density
                    )

                    tile_size = c_tile * h_tile * w_tile
                    pe_efficiency = tile_size / (
                        ceil(tile_size / peak_parallelism) * peak_parallelism
                    )

                    compute_latency = (
                        pe_compute_latency / pe_efficiency * ceil(n_tiles / self.n_pes)
                    )
                    latency = (
                        compute_latency + mem_latency
                        if self.latency_mode == "add"
                        else max(compute_latency, mem_latency)
                    )
                    if best_latency is None or latency < best_latency:
                        best_latency = latency
                        best_tile = {
                            "c_tile": c_tile,
                            "h_tile": h_tile,
                            "w_tile": w_tile,
                            "n_tiles": n_tiles,
                        }
                        c_efficiency = cout / (n_c_tile * c_tile)
                        h_efficiency = outh / (n_h_tile * h_tile)
                        w_efficiency = outw / (n_w_tile * w_tile)
                        chip_efficiency = n_tiles / (
                            ceil(n_tiles / self.n_pes) * self.n_pes
                        )
                        best_info = {
                            "c_efficiency": c_efficiency,
                            "w_efficiency": w_efficiency,
                            "h_efficiency": h_efficiency,
                            "pe_efficiency": pe_efficiency,
                            "chip_efficiency": chip_efficiency,
                            "compute_latency": compute_latency,
                            "memory_latency": mem_latency,
                        }
        return best_tile, best_info, best_latency

    def simulate_conv(self, cin, cout, inh, inw, ks, groups=1, stride=1):
        outh = inh // stride
        outw = inw // stride
        # compute
        compute_conv = cin // groups * cout * outh * outw * ks * ks
        # data
        data_input = cin * inh * inw
        data_weight = cin // groups * cout * ks * ks
        data_output = cout * outh * outw
        cfg, best_info, latency = self.search_conv_tile_cfg(
            cout, cin, outh, outw, groups, stride, ks
        )
        if self.verbose:
            # print(f"tile={cfg} efficiency={efficiency}  best_info={best_info} theory compute {compute_conv}")
            print(
                f"tile={cfg} latency={latency}  best_info={best_info} theory compute {compute_conv} theory compute t {compute_conv/self.frequency/self.n_pes/self.pe_fp32s} theory mem {data_input+data_output+data_weight}"
            )
        conv_cfg = {
            "cin": cin,
            "cout": cout,
            "inh": inh,
            "inw": inw,
            "groups": groups,
            "stride": stride,
        }
        cfg.update(conv_cfg)
        report = SimulationReport(
            compute_latency=best_info["compute_latency"],
            memory_latency=best_info["memory_latency"],
            latency=latency + self.launch_time,
            cfg=cfg,
        )
        return report

    def search_elewise_tile_cfg(self, c, outh, outw, search_upper=8):
        best_latency = None
        best_tile = None
        best_info = None
        peak_parallelism = self.pe_fp32s * self.fp32_cycles
        for c_tile in pow2_div_tile_search_space(c, pow2_upper=search_upper):
            n_c_tile = ceil(c / c_tile)
            for h_tile in pow2_div_tile_search_space(outh, pow2_upper=search_upper):
                n_h_tile = ceil(outh / h_tile)
                for w_tile in pow2_div_tile_search_space(outw, pow2_upper=search_upper):
                    n_w_tile = ceil(outw / w_tile)
                    onchip_input = c_tile * h_tile * w_tile * 2
                    onchip_output = c_tile * h_tile * w_tile
                    # mem_efficiency=ceil_efficiency(w_tile,self.mem_concurrent_fp32)
                    # pe_mem_latency=(onchip_input/mem_efficiency+onchip_output/mem_efficiency)/self.mem_fp32_bandwidth
                    pe_mem_latency = (
                        onchip_input + onchip_output
                    ) / self.mem_fp32_bandwidth
                    # pe_mem_latency=(onchip_weight+onchip_input+onchip_output)/self.offchip_mem_bandwidth
                    compute = c_tile * h_tile * w_tile
                    pe_compute_latency = compute / self.frequency / self.pe_fp32s

                    tile_size = c_tile * h_tile * w_tile
                    pe_efficiency = ceil_efficiency(tile_size, peak_parallelism)

                    n_tiles = n_c_tile * n_h_tile * n_w_tile
                    compute_latency = (
                        pe_compute_latency / pe_efficiency * ceil(n_tiles / self.n_pes)
                    )
                    mem_latency = pe_mem_latency * n_tiles
                    if self.batch_size > 1:
                        mem_latency = mem_latency * self.batch_size
                        compute_latency = compute_latency * self.batch_size

                    latency = (
                        compute_latency + mem_latency
                        if self.latency_mode == "add"
                        else max(compute_latency, mem_latency)
                    )
                    if best_latency is None or latency < best_latency:
                        best_latency = latency
                        best_tile = {
                            "c_tile": c_tile,
                            "h_tile": h_tile,
                            "w_tile": w_tile,
                            "n_tiles": n_tiles,
                        }
                        c_efficiency = c / (n_c_tile * c_tile)
                        h_efficiency = outh / (n_h_tile * h_tile)
                        w_efficiency = outw / (n_w_tile * w_tile)
                        chip_efficiency = n_tiles / (
                            ceil(n_tiles / self.n_pes) * self.n_pes
                        )
                        best_info = {
                            "c_efficiency": c_efficiency,
                            "h_efficiency": h_efficiency,
                            "w_efficiency": w_efficiency,
                            "pe_efficiency": pe_efficiency,
                            "chip_efficiency": chip_efficiency,
                            # 'mem_efficiency':mem_efficiency,
                            "compute_latency": compute_latency,
                            "mem_latency": mem_latency,
                        }
                        # best_info=[c_efficiency,h_efficiency,w_efficiency,pe_efficiency,chip_efficiency,n_tiles,mem_efficiency,compute_latency,mem_latency]
        return best_tile, best_info, best_latency

    def simulate_avg_pool(self, c, h, w, oh, ow):
        # Assumption: computation can be overlapped by memory
        all_input = c * h * w
        all_output = c * oh * ow
        all_pe_memory = all_input + oh * ow * c
        global_latency = (all_input + all_output) / self.mem_fp32_bandwidth
        l2_latency = all_pe_memory * self.batch_size / self.l2_fp32_bandwidth
        mem_latency = global_latency * self.batch_size + l2_latency
        report = SimulationReport(
            compute_latency=0,
            memory_latency=mem_latency,
            latency=mem_latency+self.launch_time,
        )
        return report

    def simulate_add(self, c, h, w):

        best_tile, best_info, latency = self.search_elewise_tile_cfg(c, h, w)
        if self.verbose:
            print(f"tile={best_tile} best_info={best_info}")
        report = SimulationReport(
            compute_latency=best_info["compute_latency"],
            memory_latency=best_info["mem_latency"],
            latency=latency + self.launch_time,
            cfg=best_tile,
        )
        return report

    def search_global_avg_pool_tile_cfg(self, c, h, w, search_upper=8):
        best_latency = None
        best_tile = None
        best_info = None
        peak_parallelism = self.pe_fp32s * self.fp32_cycles
        for c_i in range(0, search_upper):
            c_tile = 2**c_i
            if c_tile > c * 2:
                break
            n_c_tile = ceil(c / c_tile)
            for h_i in range(0, 1):
                h_tile = 2**h_i
                if h_tile > h * 2:
                    break
                n_h_tile = ceil(h / h_tile)
                for w_i in range(0, 1):
                    w_tile = 2**w_i
                    if w_tile > w * 3:
                        break
                    n_w_tile = ceil(w / w_tile)
                    compute_latency = mem_latency = 0
                    tile_size = c_tile * h_tile * w_tile
                    pe_efficiency = ceil_efficiency(tile_size, peak_parallelism)
                    onchip_input = c_tile * w_tile * h_tile
                    onchip_output = c_tile

                    # input_mem_efficiency=ceil_efficiency(w_tile,self.mem_concurrent_fp32)
                    # pe_mem_latency=(onchip_input/input_mem_efficiency+onchip_output)/self.mem_fp32_bandwidth
                    pe_mem_latency = (
                        onchip_input / self.mem_fp32_bandwidth
                        + onchip_output / self.l2_fp32_bandwidth
                    )
                    compute = c_tile * h_tile * w_tile
                    pe_compute_latency = compute / self.frequency / self.pe_fp32s

                    n_tiles = n_c_tile * n_h_tile * n_w_tile
                    compute_latency += (
                        pe_compute_latency / pe_efficiency * ceil(n_tiles / self.n_pes)
                    )
                    mem_latency += pe_mem_latency * n_tiles

                    # reduce in one core
                    onchip_input = c_tile * n_h_tile * n_w_tile
                    onchip_output = c_tile
                    pe_mem_latency = (
                        onchip_input / self.l2_fp32_bandwidth
                        + onchip_output / self.l2_fp32_bandwidth
                    )
                    # compute=c_tile*n_h_tile*n_w_tile
                    # pe_compute_latency=compute/self.frequency/self.pe_fp32s
                    # pe_efficiency=ceil_efficiency(n_h_tile*n_w_tile,peak_parallelism)
                    reduce_compute_latency = self.pe_reduce_compute_latency(
                        c_tile, n_h_tile * n_w_tile
                    )
                    compute_latency += reduce_compute_latency
                    mem_latency += pe_mem_latency * n_c_tile
                    mem_latency*=self.batch_size

                    latency = (
                        compute_latency + mem_latency
                        if self.latency_mode == "add"
                        else max(compute_latency, mem_latency)
                    )
                    if best_latency is None or latency < best_latency:
                        best_latency = latency
                        best_tile = {
                            "c_tile": c_tile,
                            "h_tile": h_tile,
                            "w_tile": w_tile,
                            "n_tiles": n_tiles,
                        }
                        best_info = {
                            "pe_efficiency": pe_efficiency,
                            # 'input_mem_efficiency':input_mem_efficiency,
                            "compute_latency": compute_latency,
                            "mem_latency": mem_latency,
                        }
        return best_tile, best_info, best_latency

    def search_spatial_broadcast_mult_cfg(self, c, h, w, search_upper=8):
        best_latency = None
        best_tile = None
        best_info = None
        peak_parallelism = self.pe_fp32s * self.fp32_cycles
        for c_tile in pow2_div_tile_search_space(c, pow2_upper=search_upper):
            n_c_tile = ceil(c / c_tile)
            for h_tile in pow2_div_tile_search_space(h, pow2_upper=search_upper):
                n_h_tile = ceil(h / h_tile)
                for w_tile in pow2_div_tile_search_space(w, pow2_upper=search_upper):
                    n_w_tile = ceil(w / w_tile)
                    n_tiles = n_c_tile * n_h_tile * n_w_tile

                    pe_memory = c_tile * h_tile * w_tile * 2 + c_tile
                    all_pe_memory = pe_memory * n_tiles
                    tot_fused_memory = c * h * w * 2 + c
                    mem_latency = self.calc_memory_latency(
                        all_pe_memory,
                        tot_fused_memory,
                        req_size=w_tile,
                        req_interval=w - w_tile,
                    )

                    compute = c_tile * h_tile * w_tile
                    pe_compute_latency = compute / self.frequency / self.pe_fp32s

                    tile_size = c_tile * h_tile * w_tile
                    pe_efficiency = ceil_efficiency(tile_size, peak_parallelism)

                    n_tiles = n_c_tile * n_h_tile * n_w_tile
                    compute_latency = (
                        pe_compute_latency / pe_efficiency * ceil(n_tiles / self.n_pes)
                    )

                    if self.batch_size > 1:
                        mem_latency = mem_latency * self.batch_size
                        compute_latency = compute_latency * self.batch_size

                    latency = (
                        compute_latency + mem_latency
                        if self.latency_mode == "add"
                        else max(compute_latency, mem_latency)
                    )
                    if best_latency is None or latency < best_latency:
                        best_latency = latency
                        best_tile = {
                            "c_tile": c_tile,
                            "h_tile": h_tile,
                            "w_tile": w_tile,
                            "n_tiles": n_tiles,
                        }
                        c_efficiency = c / (n_c_tile * c_tile)
                        h_efficiency = h / (n_h_tile * h_tile)
                        w_efficiency = w / (n_w_tile * w_tile)
                        chip_efficiency = n_tiles / (
                            ceil(n_tiles / self.n_pes) * self.n_pes
                        )
                        best_info = {
                            "c_efficiency": c_efficiency,
                            "h_efficiency": h_efficiency,
                            "w_efficiency": w_efficiency,
                            "pe_efficiency": pe_efficiency,
                            "chip_efficiency": chip_efficiency,
                            # 'mem_efficiency':mem_efficiency,
                            "compute_latency": compute_latency,
                            "mem_latency": mem_latency,
                        }
                        # best_info=[c_efficiency,h_efficiency,w_efficiency,pe_efficiency,chip_efficiency,n_tiles,mem_efficiency,compute_latency,mem_latency]
        return best_tile, best_info, best_latency

    def simulate_fc(self, cin, cout, search_upper=8):
        """ """
        # search fc tile cfg
        best_latency = None
        best_tile = None
        best_info = None
        peak_parallelism = self.pe_fp32s * self.fp32_cycles
        for co_tile in pow2_div_tile_search_space(cout, pow2_upper=search_upper):
            for ci_tile in pow2_div_tile_search_space(cin, pow2_upper=search_upper):
                n_tiles = co_tile * ci_tile
                pe_weight = co_tile * ci_tile
                pe_input = ci_tile
                pe_output = co_tile
                all_pe_memory = (
                    (pe_weight + pe_input + pe_output) * n_tiles * self.batch_size
                )
                all_weight = cin * cout
                all_input = cin
                all_output = cout
                tot_fused_memory = (
                    all_weight + all_input + all_output
                ) * self.batch_size
                mem_latency = self.calc_memory_latency(
                    all_pe_memory, tot_fused_memory, ci_tile, cin
                )

                compute = ci_tile * co_tile * self.batch_size
                pe_compute_latency = compute / self.frequency / self.pe_fp32s

                tile_size = ci_tile * co_tile * self.batch_size
                pe_efficiency = ceil_efficiency(tile_size, peak_parallelism)

                compute_latency = (
                    pe_compute_latency / pe_efficiency * ceil(n_tiles / self.n_pes)
                )

                latency = (
                    compute_latency + mem_latency
                    if self.latency_mode == "add"
                    else max(compute_latency, mem_latency)
                )
                if best_latency is None or latency < best_latency:
                    best_latency = latency
                    best_tile = {
                        "ci_tile": ci_tile,
                        "co_tile": co_tile,
                    }
                    best_info = {
                        "pe_efficiency": pe_efficiency,
                        "compute_latency": compute_latency,
                        "memory_latency": mem_latency,
                    }
        if self.verbose:
            print(best_info, best_tile)
        report = SimulationReport(
            compute_latency=best_info["compute_latency"],
            memory_latency=best_info["memory_latency"],
            latency=best_latency + self.launch_time,
        )
        return report

    def simulate_se(self, c, h, w, squeeze_channels):

        # compute pool with multi cores
        best_tile, best_info_pool, pool_latency = self.search_global_avg_pool_tile_cfg(
            c, h, w
        )
        if self.verbose:
            print(
                f"pool tile={best_tile} latency={pool_latency}  best_info={best_info_pool}"
            )
        fc1 = self.simulate_fc(c, squeeze_channels)
        fc2 = self.simulate_fc(squeeze_channels, c)
        # mult latency
        (
            best_tile,
            best_info_mult,
            mult_latency,
        ) = self.search_spatial_broadcast_mult_cfg(c, h, w)
        if self.verbose:
            print(
                f"mult tile={best_tile} latency={mult_latency}  best_info={best_info_mult}"
            )

        latency = pool_latency + self.launch_time + mult_latency + self.launch_time
        report = SimulationReport(latency=latency, cfg=best_tile)
        report += fc1 + fc2
        return report
