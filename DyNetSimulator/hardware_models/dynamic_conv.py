from math import ceil
import numpy as np
from report import SimulationReport
from hardware_models.utils import *


def calc_dynamic_conv_pe_compute_latency(
    self,
    c_tile,
    h_tile,
    w_tile,
    n_patches_parallel,
    cin,
    cout,
    groups,
    ks,
    n_patches,
    ic_density=1,
    oc_density=1,
    c_n_groups=1,
    batch_size=1,
):
    if n_patches == 0:
        return 0
    compute_patch_batch = (
        c_tile * h_tile * w_tile * (cin // groups) * ks * ks * n_patches_parallel
    )
    tile_size = c_tile * h_tile * w_tile * batch_size
    pe_efficiency = ceil_efficiency(
        tile_size * n_patches_parallel, self.pe_fp32s * self.fp32_cycles
    )
    pe_compute_patch_batch_latency = (
        compute_patch_batch / self.frequency / self.pe_fp32s / pe_efficiency
    )
    patch_batches = ceil(n_patches / n_patches_parallel)
    max_oc_density = calc_max_c_density(
        ceil(cout / c_tile), c_tile, ic_density * oc_density, c_n_groups, n=100
    )
    pe_compute_latency = (
        pe_compute_patch_batch_latency * patch_batches * ic_density * max_oc_density
    )
    if batch_size > 1:
        pe_compute_latency = pe_compute_latency * batch_size
    # print(patch_batches,)
    return pe_compute_latency


def calc_dynamic_conv_memory_latency(
    self,
    n_tiles,
    c_tile,
    h_tile,
    w_tile,
    cout,
    cin,
    outh,
    outw,
    groups,
    stride,
    ks,
    granul_size,
    input_gathered,
    n_patches,
    ic_density=1,
    oc_density=1,
    c_n_groups=1,
    batch_size=1,
):
    pe_weight = c_tile * (cin // groups) * ks * ks
    n_groups = ceil(c_tile / (cout // groups))
    pe_input = (
        n_groups
        * (cin // groups)
        * (h_tile + ks - 1)
        * stride
        * (w_tile + ks - 1)
        * stride
        * n_patches
    )
    # n_groups*(cin//groups)*(h_tile+ks-1)*stride*(w_tile+ks-1)*stride
    pe_output = c_tile * h_tile * w_tile * n_patches
    # c_tile*h_tile*w_tile
    efficiency_input = mem_concurrent_efficiency(
        w_tile,
        granul_size - w_tile if input_gathered else outw - w_tile,
        self.mem_concurrent_fp32,
    )
    efficiency_input = 1
    # efficiency_output=mem_concurrent_efficiency(w_tile,granul_size-w_tile,self.mem_concurrent_fp32)
    efficiency_output = 1

    if batch_size == 1:
        pe_weight *= ic_density * oc_density

    all_pe_memory = (
        pe_weight
        + pe_input / efficiency_input * ic_density * batch_size
        + pe_output / efficiency_output * oc_density * batch_size
    ) * n_tiles
    # weight all load into L2 and each PE fetch their required weight
    all_weight = cout * (cin // groups) * ks * ks
    if input_gathered:
        all_input = (
            n_patches
            * cin
            * (granul_size + ks - 1)
            * stride
            * (granul_size + ks - 1)
            * stride
        )
    else:
        all_input = cin * outh * stride * outw * stride
    all_output = n_patches * cout * granul_size * granul_size

    tot_fused_memory = (
        all_weight
        + all_input * ic_density * batch_size
        + all_output * oc_density * batch_size
    )
    global_latency = tot_fused_memory / self.mem_fp32_bandwidth
    # efficiency=ceil_efficiency(w_tile,self.mem_concurrent_fp32)

    # print(f"==== memory l2 efficiency {efficiency}")
    l2_latency = (all_pe_memory) / self.l2_fp32_bandwidth
    return global_latency + l2_latency


def search_dynamic_conv_tile_cfg(
    self,
    cout,
    cin,
    outh,
    outw,
    groups,
    stride,
    ks,
    granul_size,
    input_gathered,
    search_upper=8,
    ic_density=1,
    oc_density=1,
    c_n_groups=1,
    batch_size=1,
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
    for n_patches_parallel in pow2_div_tile_search_space(2**search_upper):
        # n_patches_parallel = 2**p_i
        # patches_parallel_efficiency=(candidates/(np.ceil(candidates/n_patches_parallel)*n_patches_parallel)).mean()
        # mean_patch_batches=np.ceil(candidates/n_patches_parallel).mean()
        # for c_i in range(0, search_upper):
        #     c_tile = 2**c_i
        #     if c_tile > cout * 2:
        #         break
        for c_tile in pow2_div_tile_search_space(cout):
            n_c_tile = ceil(cout / c_tile)
            # c_efficiency=cout/(n_c_tile*c_tile)
            # for h_i in range(0, search_upper):
            #     h_tile = 2**h_i
            #     if h_tile > granul_size * 2:
            #         break
            #     n_h_tile = ceil(granul_size / h_tile)
            for h_tile in pow2_div_tile_search_space(granul_size):
                n_h_tile = ceil(granul_size / h_tile)
                # h_efficiency=inh/(n_h_tile*h_tile)
                # for w_i in range(0, search_upper):
                #     w_tile = 2**w_i
                #     if w_tile > granul_size * 2:
                #         break
                for w_tile in pow2_div_tile_search_space(granul_size):
                    n_w_tile = ceil(granul_size / w_tile)
                    n_tiles = n_c_tile * n_h_tile * n_w_tile
                    # w_efficiency=inw/(n_w_tile*w_tile)
                    # if c_tile*h_tile*w_tile<tile_parallel:
                    #     continue

                    # pe_mem_latency=self.calc_dynamic_tile_memory_latency(c_tile,h_tile,w_tile,cout,cin,groups,ks,stride,mean_n_patches)
                    # mem_latency=pe_mem_latency*n_tiles
                    mem_latency = calc_dynamic_conv_memory_latency(
                        self,
                        n_tiles,
                        c_tile,
                        h_tile,
                        w_tile,
                        cout,
                        cin,
                        outh,
                        outw,
                        groups,
                        stride,
                        ks,
                        granul_size,
                        input_gathered,
                        mean_n_patches,
                        ic_density,
                        oc_density,
                        c_n_groups,
                        batch_size,
                    )

                    pe_compute_latency = calc_dynamic_conv_pe_compute_latency(
                        self,
                        c_tile,
                        h_tile,
                        w_tile,
                        n_patches_parallel,
                        cin,
                        cout,
                        groups,
                        ks,
                        mean_n_patches,
                        ic_density,
                        oc_density,
                        c_n_groups,
                        batch_size,
                    )
                    compute_latency = pe_compute_latency * ceil(n_tiles / self.n_pes)

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
                        c_efficiency = cout / (n_c_tile * c_tile)
                        h_efficiency = granul_size / (n_h_tile * h_tile)
                        w_efficiency = granul_size / (n_w_tile * w_tile)
                        chip_efficiency = n_tiles / (
                            ceil(n_tiles / self.n_pes) * self.n_pes
                        )
                        pe_efficiency = ceil_efficiency(
                            c_tile * h_tile * w_tile * n_patches_parallel * batch_size,
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
