from math import ceil
import numpy as np
from report import SimulationReport
from hardware_models.utils import *
from hardware_models.static_predictor import MultiCoresPredictor
from hardware_models.dynamic_conv import *


class GPGPUDynamicPredictor(MultiCoresPredictor):
    def simulate_gather(self, c, h, w, granul_size, density, granul_size_pad=0):
        n_patches = ceil(density * ceil(h / granul_size) * ceil(w / granul_size))
        data_input = c * h * w
        padded_granul_size = granul_size_pad * 2 + granul_size
        l2_data = c * n_patches * padded_granul_size * padded_granul_size
        l2_efficiency = mem_concurrent_efficiency(
            padded_granul_size, max(0, w - padded_granul_size), self.mem_concurrent_fp32
        )
        data_output = c * n_patches * padded_granul_size * padded_granul_size
        info = {"l2_efficiency": l2_efficiency}

        mem_latency = (
            (data_input + data_output) / self.mem_fp32_bandwidth
            + l2_data / self.l2_fp32_bandwidth / l2_efficiency
            + l2_data / self.l2_fp32_bandwidth
        )
        if self.batch_size > 1:
            mem_latency = mem_latency * self.batch_size
        if self.verbose:
            print(
                f"gather {info} mem_latency={mem_latency}",
            )
        report = SimulationReport(
            latency=mem_latency + self.launch_time,
            compute_latency=0,
            memory_latency=mem_latency,
        )
        return report

    def simulate_scatter(self, c, h, w, granul_size, density):
        n_patches = ceil(density * ceil(h / granul_size) * ceil(w / granul_size))
        data_output = c * h * w
        l2_data = c * n_patches * granul_size * granul_size
        l2_efficiency = mem_concurrent_efficiency(
            granul_size, max(0, w - granul_size), self.mem_concurrent_fp32
        )
        data_input = c * n_patches * granul_size * granul_size
        info = {"l2_efficiency": l2_efficiency}

        mem_latency = (
            (data_input + data_output) / self.mem_fp32_bandwidth
            + l2_data / self.l2_fp32_bandwidth / l2_efficiency
            + l2_data / self.l2_fp32_bandwidth
        )
        if self.batch_size > 1:
            mem_latency = mem_latency * self.batch_size
        if self.verbose:
            print(
                f"scatter {info} mem_latency={mem_latency}",
            )
        report = SimulationReport(
            latency=mem_latency + self.launch_time,
            compute_latency=0,
            memory_latency=mem_latency,
        )
        return report

    def simulate_masker_conv1(
        self,
        cin,
        cout,
        h,
        w,
        ks,
        granul_size,
        density,
        nxt_layer_ks,
        test_nofuse=True,
        c_density=1,
        n_c_dy_group=1,
        channel_masker=True,
        channel_masker_hid_size=32,
        spatial_masker=False,
        no_fuse=False,
    ):
        """
        n_c_dy_group: number of channels in a dynamic groups
        channel_masker: is generate channel mask (conv->pool->fc)
        spatial_masker: is generate spatial mask
        """
        c_n_groups = cin // n_c_dy_group
        # fuse
        fused_cout = cout
        if channel_masker:
            fused_cout += channel_masker_hid_size
        if spatial_masker:
            fused_cout += 1
        fuse_report = self.simulate_conv(cin, fused_cout, h, w, 1)
        if channel_masker:
            pool_report = self.simulate_avg_pool(channel_masker_hid_size, h, w, 1, 1)
            fc2_report = self.simulate_fc(channel_masker_hid_size, c_n_groups)
            DEBUG = 0
            if DEBUG:
                all_lat = fuse_report.latency + pool_report.latency + fc2_report.latency
                print(pool_report, fc2_report, fuse_report)
                original = self.simulate_conv(cin, cout, h, w, 1)
                print(original)
                print(
                    pool_report.latency / all_lat,
                    fc2_report.latency / all_lat,
                    fuse_report.latency / all_lat,
                )
                print(
                    pool_report.latency / original.latency,
                    fc2_report.latency / original.latency,
                    fuse_report.latency / original.latency,
                )
            fuse_report += pool_report + fc2_report
        else:
            # with extra channel masker
            fuse_report = self.simulate_dynamic_conv(
                cin,
                fused_cout,
                h,
                w,
                1,
                1,
                1,
                spatial_masker=False,
                channel_masker=True,
                c_n_groups=c_n_groups,
                ic_density=1,
                oc_density=c_density,
            )

        if self.verbose:
            print(f"fused maker_conv1 {fuse_report}")

        if test_nofuse or no_fuse:
            spatial_masker_report = self.simulate_conv(cin, 2, h, w, 1, 1, 1)
            conv1_report = self.simulate_dynamic_conv(
                cin,
                cout,
                h,
                w,
                1,
                1,
                1,
                granul_size,
                density,
                granul_size_pad=(nxt_layer_ks - 1) // 2,
                with_indexing=True,
                ic_density=1,
                oc_density=c_density,
                c_n_groups=c_n_groups,
                spatial_masker=spatial_masker,
            )

            nofuse_report = spatial_masker_report + conv1_report

            if channel_masker:
                channel_masker_conv1 = self.simulate_conv(
                    cin, channel_masker_hid_size, h, w, 1, 1, 1
                )
                pool_report = self.simulate_avg_pool(
                    channel_masker_hid_size, h, w, 1, 1
                )
                fc2_report = self.simulate_fc(channel_masker_hid_size, c_n_groups)
                nofuse_report += channel_masker_conv1 + pool_report + fc2_report

            if self.verbose:
                print(f"no fused maker_conv1 {nofuse_report}")
            report = (
                nofuse_report
                if nofuse_report.latency < fuse_report.latency or no_fuse
                else fuse_report
            )
        else:
            report = fuse_report
        return report

    def simulate_dynamic_conv(
        self,
        cin,
        cout,
        inh,
        inw,
        ks,
        groups,
        stride,
        granul_size=1,
        density=1,
        with_indexing=False,
        granul_size_pad=0,
        ic_density=1,
        oc_density=1,
        c_n_groups=1,
        channel_masker=True,
        spatial_masker=True,
    ):
        outh = inh // stride
        outw = inw // stride
        n_patches = ceil(density * ceil(outh / granul_size) * ceil(outw / granul_size))

        if spatial_masker:
            # compute
            compute_conv = (
                cin
                // groups
                * cout
                * n_patches
                * (granul_size + granul_size_pad * 2) ** 2
                * ks
                * ks
            )

            # data
            data_input = (
                cin
                * n_patches
                * ((granul_size + granul_size_pad * 2) * stride + ks - 1)
                * ((granul_size + granul_size_pad * 2) * stride + ks - 1)
            )
            data_output = (
                cout
                * n_patches
                * (granul_size + granul_size_pad * 2)
                * (granul_size + granul_size_pad * 2)
            )
            data_weight = cin // groups * cout * ks * ks

            best_tile, best_info, _ = search_dynamic_conv_tile_cfg(
                self,
                cout,
                cin,
                outh,
                outw,
                groups,
                stride,
                ks,
                granul_size,
                input_gathered=not with_indexing,
                ic_density=ic_density,
                oc_density=oc_density,
                c_n_groups=c_n_groups,
                batch_size=self.batch_size,
            )

            # pe_mem_latency=self.calc_dynamic_tile_memory_latency(best_tile['c_tile'],best_tile['h_tile'],best_tile['w_tile'],cout,cin,groups,ks,stride,n_patches)
            # mem_latency=pe_mem_latency*best_tile['n_tiles']
            mem_latency = calc_dynamic_conv_memory_latency(
                self,
                best_tile["n_tiles"],
                best_tile["c_tile"],
                best_tile["h_tile"],
                best_tile["w_tile"],
                cout,
                cin,
                outh,
                outw,
                groups,
                stride,
                ks,
                granul_size,
                not with_indexing,
                n_patches,
                ic_density,
                oc_density,
                batch_size=self.batch_size,
            )
            efficiency = mem_concurrent_efficiency(
                best_tile["w_tile"],
                granul_size - best_tile["w_tile"]
                if not with_indexing
                else outw - best_tile["w_tile"],
                self.mem_concurrent_fp32,
            )
            if self.verbose:
                print(
                    f"==== memory l2 efficiency {efficiency} mem_latency {mem_latency}"
                )

            pe_compute_latency = calc_dynamic_conv_pe_compute_latency(
                self,
                best_tile["c_tile"],
                best_tile["h_tile"],
                best_tile["w_tile"],
                best_tile["n_patches_parallel"],
                cin,
                cout,
                groups,
                ks,
                n_patches,
                ic_density=ic_density,
                oc_density=oc_density,
                c_n_groups=c_n_groups,
                batch_size=self.batch_size,
            )
            compute_latency = pe_compute_latency * ceil(
                best_tile["n_tiles"] / self.n_pes
            )

            latency = (
                mem_latency + compute_latency
                if self.latency_mode == "add"
                else max(compute_latency, mem_latency)
            )
            if self.verbose:
                print(
                    f"n_patches {n_patches} best_latency {latency} best_tile {best_tile} best_info {best_info} theory compute {compute_conv} theory mem {data_input+data_output+data_weight}"
                )
            report = SimulationReport(
                compute_latency=compute_latency,
                memory_latency=mem_latency,
                latency=latency + self.launch_time,
                cfg=best_tile,
            )

        else:
            outh = inh // stride
            outw = inw // stride
            cfg, best_info, latency = self.search_conv_tile_cfg(
                cout,
                cin,
                outh,
                outw,
                groups,
                stride,
                ks,
                ic_density=ic_density,
                oc_density=oc_density,
            )
            report = SimulationReport(
                compute_latency=best_info["compute_latency"],
                memory_latency=best_info["memory_latency"],
                latency=latency + self.launch_time,
            )
        return report

    def calc_dynamic_elewise_memory_latency(
        self, n_tiles, c_tile, h_tile, w_tile, c, h, w, granul_size, n_patches
    ):
        pe_input = c_tile * h_tile * w_tile * n_patches * 2
        pe_output = c_tile * h_tile * w_tile * n_patches

        all_pe_memory = (pe_input + pe_output) * n_tiles * self.batch_size

        all_input = (
            n_patches * c * (granul_size) * (granul_size) + c * h * w
        )  # one gathered, another not gathered
        all_output = c * h * w
        tot_fused_memory = all_input * self.batch_size + all_output * self.batch_size
        global_latency = tot_fused_memory / self.mem_fp32_bandwidth
        # efficiency=ceil_efficiency(w_tile,self.mem_concurrent_fp32)
        efficiency = mem_concurrent_efficiency(
            w_tile, granul_size - w_tile, self.mem_concurrent_fp32
        )
        l2_latency = (all_pe_memory) / self.l2_fp32_bandwidth / efficiency
        return global_latency + l2_latency

    def calc_dynamic_elewise_pe_compute_latency(
        self, c_tile, h_tile, w_tile, n_patches_parallel, n_patches
    ):
        compute_patch_batch = c_tile * h_tile * w_tile * n_patches_parallel
        tile_size = c_tile * h_tile * w_tile * self.batch_size
        pe_efficiency = ceil_efficiency(
            tile_size * n_patches_parallel, self.pe_fp32s * self.fp32_cycles
        )
        pe_compute_patch_batch_latency = (
            compute_patch_batch / self.frequency / self.pe_fp32s / pe_efficiency
        )
        patch_batches = ceil(n_patches / n_patches_parallel)
        pe_compute_latency = (
            pe_compute_patch_batch_latency * patch_batches * self.batch_size
        )
        # print(patch_batches,)
        return pe_compute_latency

    def search_dynamic_elewise_tile_cfg(
        self, c, outh, outw, granul_size, search_upper=8
    ):
        """
        using average of n_patches to search for the tile_cfg and n_patches_parallel in one tile
        input_gathered: if True, the input is gathered with n_paches; if False, the input is not gathered
        """
        best_latency = None
        best_efficiency = None
        best_tile = None
        best_info = None
        n_h_patches = ceil(outh / granul_size)
        n_w_patches = ceil(outw / granul_size)
        candidates = np.arange(1, n_h_patches * n_w_patches + 1)
        mean_n_patches = candidates.mean()
        for n_patches_parallel in pow2_div_tile_search_space(
            candidates[-1], pow2_upper=search_upper
        ):
            for c_tile in pow2_div_tile_search_space(c, pow2_upper=search_upper):
                n_c_tile = ceil(c / c_tile)
                for h_tile in pow2_div_tile_search_space(
                    granul_size, pow2_upper=search_upper
                ):
                    n_h_tile = ceil(granul_size / h_tile)
                    for w_tile in pow2_div_tile_search_space(
                        granul_size, pow2_upper=search_upper
                    ):
                        n_w_tile = ceil(granul_size / w_tile)
                        n_tiles = n_c_tile * n_h_tile * n_w_tile

                        mem_latency = self.calc_dynamic_elewise_memory_latency(
                            n_tiles,
                            c_tile,
                            h_tile,
                            w_tile,
                            c,
                            outh,
                            outw,
                            granul_size,
                            mean_n_patches,
                        )

                        pe_compute_latency = (
                            self.calc_dynamic_elewise_pe_compute_latency(
                                c_tile,
                                h_tile,
                                w_tile,
                                n_patches_parallel,
                                mean_n_patches,
                            )
                        )
                        compute_latency = pe_compute_latency * ceil(
                            n_tiles / self.n_pes
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
                                "n_patches_parallel": n_patches_parallel,
                                "n_tiles": n_tiles,
                            }
                            c_efficiency = c / (n_c_tile * c_tile)
                            h_efficiency = outh / (n_h_tile * h_tile)
                            w_efficiency = outw / (n_w_tile * w_tile)
                            chip_efficiency = n_tiles / (
                                ceil(n_tiles / self.n_pes) * self.n_pes
                            )
                            pe_efficiency = ceil_efficiency(
                                c_tile
                                * h_tile
                                * w_tile
                                * n_patches_parallel
                                * self.batch_size,
                                self.pe_fp32s * self.fp32_cycles,
                            )
                            best_info = {
                                "c_efficiency": c_efficiency,
                                "h_efficiency": h_efficiency,
                                "w_efficiency": w_efficiency,
                                "pe_efficiency": pe_efficiency,
                                "chip_efficiency": chip_efficiency,
                            }
        return best_tile, best_info, best_latency

    def simulate_scatter_add(self, c, h, w, granul_size, density):
        n_patches = ceil(density * ceil(h / granul_size) * ceil(w / granul_size))

        # compute
        compute_add = c * n_patches * granul_size * granul_size

        # data
        data_input = data_output = c * n_patches * granul_size * granul_size
        data_ind = n_patches

        best_tile, best_info, _ = self.search_dynamic_elewise_tile_cfg(
            c, h, w, granul_size
        )

        n_tiles, c_tile, h_tile, w_tile = (
            best_tile["n_tiles"],
            best_tile["c_tile"],
            best_tile["h_tile"],
            best_tile["w_tile"],
        )
        mem_latency = self.calc_dynamic_elewise_memory_latency(
            n_tiles, c_tile, h_tile, w_tile, c, h, w, granul_size, n_patches
        )

        pe_compute_latency = self.calc_dynamic_elewise_pe_compute_latency(
            c_tile, h_tile, w_tile, best_tile["n_patches_parallel"], n_patches
        )
        compute_latency = pe_compute_latency * ceil(n_tiles / self.n_pes)

        latency = (
            compute_latency + mem_latency
            if self.latency_mode == "add"
            else max(compute_latency, mem_latency)
        )

        report = SimulationReport(
            compute_latency=compute_latency,
            memory_latency=mem_latency,
            latency=latency + self.launch_time,
            cfg=best_tile,
        )
        return report

    def calc_dynamic_global_avg_pool_memory_latency(
        self, n_tiles, c_tile, h_tile, w_tile, c, h, w, granul_size, n_patches
    ):
        pe_input = c_tile * h_tile * w_tile * n_patches
        pe_output = c_tile

        pe_reduce_input = c_tile * (ceil(h / h_tile) * ceil(w / w_tile))
        pe_reduce_output = c_tile

        all_pe_memory = (
            (pe_input + pe_output) * n_tiles + pe_reduce_input + pe_reduce_output
        )

        all_input = n_patches * c * (granul_size) * (granul_size)
        all_output = c * h * w
        tot_fused_memory = all_input + all_output
        global_latency = tot_fused_memory / self.mem_fp32_bandwidth
        # efficiency=ceil_efficiency(w_tile,self.mem_concurrent_fp32)
        efficiency = mem_concurrent_efficiency(
            w_tile, granul_size - w_tile, self.mem_concurrent_fp32
        )
        l2_latency = (all_pe_memory) / self.l2_fp32_bandwidth / efficiency
        return (global_latency + l2_latency)*self.batch_size

    def calc_dynamic_global_avg_pool_pe_compute_latency(
        self, c_tile, h_tile, w_tile, n_patches_parallel, n_patches
    ):
        compute_patch_batch = c_tile * h_tile * w_tile * n_patches_parallel
        tile_size = c_tile
        pe_efficiency = ceil_efficiency(
            tile_size * n_patches_parallel, self.pe_fp32s * self.fp32_cycles
        )
        pe_compute_patch_batch_latency = (
            compute_patch_batch / self.frequency / self.pe_fp32s / pe_efficiency
        )
        patch_batches = ceil(n_patches / n_patches_parallel)
        pe_compute_latency = pe_compute_patch_batch_latency * patch_batches
        # reduce in pe
        pe_compute_reduce = c_tile * n_patches_parallel 
        pe_reduce_efficiency = ceil_efficiency(
            tile_size, self.pe_fp32s * self.fp32_cycles
        )
        pe_reduce_latency = (
            pe_compute_reduce / self.frequency / self.pe_fp32s / pe_reduce_efficiency
        )
        pe_compute_latency += pe_reduce_latency
        # print(patch_batches,)
        return pe_compute_latency* self.batch_size

    def search_dynamic_global_avg_pool_tile_cfg(
        self, c, h, w, granul_size, search_upper=8
    ):
        best_latency = None
        best_tile = None
        best_info = None
        peak_parallelism = self.pe_fp32s * self.fp32_cycles
        n_h_patches = ceil(h / granul_size)
        n_w_patches = ceil(w / granul_size)
        candidates = np.arange(1, n_h_patches * n_w_patches + 1)
        mean_n_patches = candidates.mean()
        for n_patches_parallel in pow2_div_tile_search_space(
            candidates[-1], pow2_upper=search_upper
        ):
            for c_tile in pow2_div_tile_search_space(c, pow2_upper=search_upper):
                n_c_tile = ceil(c / c_tile)
                for h_tile in pow2_div_tile_search_space(
                    granul_size, pow2_upper=search_upper
                ):
                    n_h_tile = ceil(granul_size / h_tile)
                    for w_tile in pow2_div_tile_search_space(
                        granul_size, pow2_upper=search_upper
                    ):
                        n_w_tile = ceil(granul_size / w_tile)
                        n_tiles = n_c_tile * n_h_tile * n_w_tile

                        mem_latency = self.calc_dynamic_global_avg_pool_memory_latency(
                            n_tiles,
                            c_tile,
                            h_tile,
                            w_tile,
                            c,
                            h,
                            w,
                            granul_size,
                            mean_n_patches,
                        )
                        pe_compute_latency = (
                            self.calc_dynamic_global_avg_pool_pe_compute_latency(
                                c_tile,
                                h_tile,
                                w_tile,
                                n_patches_parallel,
                                mean_n_patches,
                            )
                        )
                        reduce_compute_latency = self.pe_reduce_compute_latency(
                            c_tile, n_h_tile * n_w_tile
                        )
                        compute_latency = (
                            pe_compute_latency * ceil(n_tiles / self.n_pes)
                            + reduce_compute_latency
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
                                "n_patches_parallel": n_patches_parallel,
                            }
                            best_info = {
                                # 'input_mem_efficiency':input_mem_efficiency,
                                "compute_latency": compute_latency,
                                "mem_latency": mem_latency,
                            }
        return best_tile, best_info, best_latency

    def simulate_dynamic_se(self, c, h, w, squeeze_channels, granul_size, density):
        n_patches = ceil(density * ceil(h / granul_size) * ceil(w / granul_size))
        # compute
        # compute pool with multi cores
        (
            best_tile,
            best_info_pool,
            pool_latency,
        ) = self.search_dynamic_global_avg_pool_tile_cfg(c, h, w, granul_size)

        n_tiles, c_tile, n_patches_parallel = (
            best_tile["n_tiles"],
            best_tile["c_tile"],
            best_tile["n_patches_parallel"],
        )
        h_tile, w_tile = best_tile["h_tile"], best_tile["w_tile"]
        mem_latency = self.calc_dynamic_global_avg_pool_memory_latency(
            n_tiles, c_tile, h_tile, w_tile, c, h, w, granul_size, n_patches
        )
        pe_compute_latency = self.calc_dynamic_global_avg_pool_pe_compute_latency(
            c_tile, h_tile, w_tile, n_patches_parallel, n_patches
        )
        reduce_compute_latency = self.pe_reduce_compute_latency(
            c_tile, ceil(h / h_tile) * ceil(w / w_tile)
        )
        compute_latency = (
            pe_compute_latency * ceil(n_tiles / self.n_pes) + reduce_compute_latency
        )
        # print(f"DEBUG: pe_compute_latency={pe_compute_latency}, reduce_compute_latency={reduce_compute_latency}")

        pool_latency = (
            compute_latency + mem_latency
            if self.latency_mode == "add"
            else max(compute_latency, mem_latency)
        )

        if self.verbose:
            print(
                f"pool tile={best_tile} latency={pool_latency}  best_info={best_info_pool}"
            )
        fc1 = self.simulate_fc(c, squeeze_channels)
        fc2 = self.simulate_fc(squeeze_channels, c)
        (
            best_tile,
            best_info_mult,
            mult_latency,
        ) = self.search_spatial_broadcast_mult_cfg(c, h, w)
        if self.verbose:
            print(
                f"mult tile={best_tile} latency={mult_latency}  best_info={best_info_mult}"
            )
        latency = (
            fc1.latency
            + fc2.latency
            + pool_latency
            + self.launch_time
            + mult_latency
            + self.launch_time
        )
        # print(f"DEBUG latency={latency}, pool_latency={pool_latency}, mult_latency={mult_latency}, fc1={fc1}, fc2={fc2}")
        report = SimulationReport(latency=latency, cfg=best_tile)
        return report

    def simulate_channel_masker_predictor(
        self, cin, h, w, n_c_dy_group, n_fc_layers, reduction_size=None
    ):
        """
        n_c_dy_group: number of channels in a dynamic groups
        """
        c_n_groups = cin // n_c_dy_group
        if c_n_groups == 1:
            # no channel dynamic
            return SimulationReport(latency=0)
        pool_report = self.simulate_avg_pool(cin, h, w, 1, 1)
        if n_fc_layers == 2:
            # fc1_report = self.simulate_conv(cin, c_n_groups * 2, 1, 1, 1)
            if reduction_size is None:
                reduction_size = 16
            if reduction_size > c_n_groups:
                print(
                    f"WARNING: reduction size {reduction_size} is larger than c_n_groups {c_n_groups}, set to c_n_groups"
                )
                reduction_size = c_n_groups
            hidden_size = c_n_groups // reduction_size
            fc1_report = self.simulate_fc(cin, hidden_size)
            # Note that the second fc2 predict one value for one channel group in inference time to reduce latency
            # because XW1>XW2 is equal to X(W1-W2)>0
            fc2_report = self.simulate_fc(hidden_size, c_n_groups)
            return pool_report + fc1_report + fc2_report
        elif n_fc_layers == 1:
            # fc1_report = self.simulate_conv(cin, c_n_groups, 1, 1, 1)
            fc1_report = self.simulate_fc(cin, c_n_groups)
            if self.verbose:
                all_lat = fc1_report.latency + pool_report.latency
                original = self.simulate_conv(cin, cin, h, w, 1)
                print("pool", pool_report, "fc1", fc1_report, "ori conv1", original)
                print(
                    pool_report.latency / all_lat,
                    fc1_report.latency / all_lat,
                    pool_report.latency / original.latency,
                    fc1_report.latency / original.latency,
                )
            # Note that the second fc2 predict one value for one channel group in inference time to reduce latency
            # because XW1>XW2 is equal to X(W1-W2)>0
            return pool_report + fc1_report
        else:
            raise NotImplementedError()
