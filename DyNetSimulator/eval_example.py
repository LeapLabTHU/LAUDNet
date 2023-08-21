import os
import sys
sys.path.append('.')
sys.path.append('..')
from hardware_models.multi_cores import GPGPUDynamicPredictor
from matplotlib import pyplot as plt
from scipy import optimize
import argparse
import numpy as np


def get_static_block_latency(predictor, c_in, c_out, b, h, w, n_groups, stride, down, is_se):
    conv1 = predictor.simulate_conv(c_in, c_out//b, inh=h, inw=w, ks=1, groups=1)
    conv2 = predictor.simulate_conv(c_out//b, c_out//b, inh=h, inw=w, ks=3, groups=n_groups, stride=stride)
    conv3 = predictor.simulate_conv(c_out//b, c_out, inh=h//down, inw=w//down, ks=1, groups=1)
    add = predictor.simulate_add(c=c_out, h=h//down, w=w//down)
    raw_report = conv1 + conv2 + conv3 + add

    if down == 2:
        down1x1 = predictor.simulate_conv(c_in, c_out, inh=h, inw=w, ks=1, groups=1, stride=2)
        raw_report += down1x1
    if is_se == True:
        se = predictor.simulate_se(c=c_out//b, h=h//down, w=w//down, squeeze_channels=int(round(0.25 * c_in)))
        se.compute_latency = 0
        se.memory_latency = 0
        raw_report += se

    return raw_report.latency


def get_dynamic_block_latency_spatial(predictor, c_in, c_out, b, h, w, n_groups, granul_size, c_granul_size,
                                      density_conv1, density_conv2, density_conv3, c_density, stride, down, is_se):
    masker_conv1 = predictor.simulate_masker_conv1(
        c_in, c_out//b, h, w, ks=1, granul_size=granul_size, 
        density=density_conv1, nxt_layer_ks=3, c_density=c_density, 
        n_c_dy_group=c_granul_size, test_nofuse=True,
        channel_masker=False, spatial_masker=True)
    c_n_groups = c_out // b//c_granul_size
    gather_conv2 = predictor.simulate_dynamic_conv(
        c_out//b, c_out//b, h, w, ks=3, groups=n_groups, stride=stride, 
        granul_size=granul_size, density=density_conv2, ic_density=c_density, 
        oc_density=c_density, c_n_groups=c_n_groups, with_indexing=True,
        channel_masker=False, spatial_masker=True)
    conv3 = predictor.simulate_dynamic_conv(
        c_out//b, c_out, h//down, w//down, ks=1, groups=1, stride=1, 
        granul_size=granul_size, density=density_conv3, ic_density=c_density, 
        c_n_groups=c_n_groups, with_indexing=False,
        channel_masker=False, spatial_masker=True)
    scatter_add = predictor.simulate_scatter_add(
        c_out, h//down, w//down, granul_size, density_conv3)
    report = masker_conv1 + gather_conv2 + conv3 + scatter_add

    if down == 2:
        down1x1 = predictor.simulate_conv(c_in, c_out, inh=h, inw=w, ks=1, groups=1, stride=2)
        report += down1x1
    if is_se == True:
        se = predictor.simulate_dynamic_se(c=c_out//b, h=h//down, w=w//down, squeeze_channels=int(round(0.25 * c_in)), granul_size=granul_size, density=density_conv1)
        report += se

    return report.latency


def get_dynamic_block_latency_channel(predictor, c_in, c_out, b, h, w, n_groups, granul_size, c_granul_size,
                                      density_conv1, density_conv2, density_conv3, c_density, stride, down, is_se, layer=2):
    c_n_groups = c_out // b//c_granul_size
    masker_conv1 = predictor.simulate_dynamic_conv(
        c_in, c_out//b, h, w, ks=1, groups=1, stride=1, granul_size=granul_size,  # groups=1 for conv1 and conv3
        density=density_conv1, ic_density=1.0, oc_density=c_density,
        c_n_groups=c_n_groups, with_indexing=False,
        channel_masker=False, spatial_masker=False)
    masker_channel = predictor.simulate_channel_masker_predictor(
        c_in, h, w, c_granul_size, layer, reduction_size=16)
    gather_conv2 = predictor.simulate_dynamic_conv(
        c_out//b, c_out//b, h, w, ks=3, groups=n_groups, stride=stride, 
        granul_size=granul_size, density=density_conv2, ic_density=c_density, 
        oc_density=c_density, c_n_groups=c_n_groups, with_indexing=False,
        channel_masker=True, spatial_masker=False)
    conv3 = predictor.simulate_dynamic_conv(
        c_out//b, c_out, h//down, w//down, ks=1, groups=1, stride=1, 
        granul_size=granul_size, density=density_conv3, ic_density=c_density, 
        oc_density=1.0, c_n_groups=c_n_groups, with_indexing=False,
        channel_masker=True, spatial_masker=False)
    scatter_add = predictor.simulate_scatter_add(
        c_out, h//down, w//down, granul_size, density_conv3)
    report = masker_conv1 + masker_channel + gather_conv2 + conv3 + scatter_add

    if down == 2:
        down1x1 = predictor.simulate_conv(c_in, c_out, inh=h, inw=w, ks=1, groups=1, stride=2)
        report += down1x1
    if is_se == True:
        se = predictor.simulate_dynamic_se(c=c_out//b, h=h//down, w=w//down, squeeze_channels=int(round(0.25 * c_in)), granul_size=granul_size, density=density_conv1)
        report += se

    return report.latency


def get_skipping_block_latency(predictor, c_in, c_out, b, h, w, n_groups, granul_size, c_granul_size,
                              density_conv1, density_conv2, density_conv3, c_density, stride, down, is_se, layer=2):
    conv1 = predictor.simulate_conv(c_in, c_out//b, inh=h, inw=w, ks=1, groups=1)
    conv2 = predictor.simulate_conv(c_out//b, c_out//b, inh=h, inw=w, ks=3, groups=n_groups, stride=stride)
    conv3 = predictor.simulate_conv(c_out//b, c_out, inh=h//down, inw=w//down, ks=1, groups=1)
    add = predictor.simulate_add(c=c_out, h=h//down, w=w//down)
    spatial_masker = predictor.simulate_masker_conv1(
        c_in, c_out//b, h, w, ks=1, granul_size=granul_size, 
        density=0, nxt_layer_ks=3, c_density=0, 
        n_c_dy_group=c_granul_size, test_nofuse=False,
        channel_masker=False, spatial_masker=True)
    fixed_latency = spatial_masker.latency
    layer_latency = conv1.latency + conv2.latency + conv3.latency + add.latency

    if down == 2:
        down1x1 = predictor.simulate_conv(c_in, c_out, inh=h, inw=w, ks=1, groups=1, stride=2)
        fixed_latency += down1x1.latency
    if is_se == True:
        se = predictor.simulate_se(c=c_out//b, h=h//down, w=w//down, squeeze_channels=int(round(0.25 * c_in)))
        se.compute_latency = 0
        se.memory_latency = 0
        layer_latency += se.latency
    
    raw_report = fixed_latency + layer_latency * density_conv1

    return raw_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['resnet50', 'resnet101', 'regnety004', 'regnety008'])
    parser.add_argument('--hardware', type=str, default='v100')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--is_static', action='store_true')
    parser.add_argument('--consider_mem_concurrent_fp32', action='store_true')
    args = parser.parse_args()

    verbose = args.verbose
    if args.consider_mem_concurrent_fp32:
        if args.hardware=='v100':
            predictor = GPGPUDynamicPredictor(80, 64, 1500e6, 700e9, verbose=verbose, latency_mode="add", batch_size=128, mem_concurrent_fp32=1)
        if args.hardware=='3090':
            predictor = GPGPUDynamicPredictor(82, 10496//82, 1250e6, 936e9, verbose=0, latency_mode="add",batch_size=128,l2_speed_frac=1, mem_concurrent_fp32=1)
        if args.hardware=='3060':
            predictor=GPGPUDynamicPredictor(28, 3584/28,1777e6, 360e9, verbose=verbose,latency_mode="add",batch_size=128, mem_concurrent_fp32=1)
        if args.hardware=='tx2':
            predictor = GPGPUDynamicPredictor(2, 128, 1300e6, 59.7e9, verbose=verbose, latency_mode="add", batch_size=1, mem_concurrent_fp32=1)
        if args.hardware=='nano':
            predictor=GPGPUDynamicPredictor(1, 128, 921e6, 25.6e9, verbose=verbose, latency_mode="add", batch_size=1, mem_concurrent_fp32=1)
    else:
        if args.hardware=='v100':
            predictor = GPGPUDynamicPredictor(80, 64, 1500e6, 700e9, verbose=verbose, latency_mode="add", batch_size=128)
        if args.hardware=='3090':
            predictor = GPGPUDynamicPredictor(82, 10496//82, 1250e6, 936e9, verbose=0, latency_mode="add",batch_size=128,l2_speed_frac=1)
        if args.hardware=='3060':
            predictor=GPGPUDynamicPredictor(28, 3584/28,1777e6, 360e9, verbose=verbose,latency_mode="add",batch_size=128)
        if args.hardware=='tx2':
            predictor = GPGPUDynamicPredictor(2, 128, 1300e6, 59.7e9, verbose=verbose, latency_mode="add", batch_size=1)
        if args.hardware=='nano':
            predictor=GPGPUDynamicPredictor(1, 128, 921e6, 25.6e9, verbose=verbose, latency_mode="add", batch_size=1)

    if args.model=='regnety004':
        widths = [56, 28, 14, 7]
        last_channels = [48, 104, 208, 440]
        first_channels = [32, 48, 104, 208]
        first_block_strides = [2, 2, 2, 2]
        bottleneck = 1  # bottleneck ratio
        group_width = 8
        is_se = True
        n_groupss = [channel// group_width for channel in last_channels]
    if args.model=='regnety008':
        widths = [56, 28, 14, 7]
        last_channels = [64, 144, 320, 784]
        first_channels = [32, 64, 144, 320]
        first_block_strides = [2, 2, 2, 2]
        bottleneck = 1  # bottleneck ratio
        group_width = 16
        is_se = True
        n_groupss = [channel// group_width for channel in last_channels]
    if args.model=='resnet50':
        widths = [56, 28, 14, 7]
        last_channels = [256, 512, 1024, 2048]
        first_channels = [64, 256, 512, 1024]
        first_block_strides = [1, 2, 2, 2]
        bottleneck = 4  # bottleneck ratio
        is_se = False
        n_groupss = [1 for _ in last_channels]
    if args.model=='resnet101' or args.model=='dynconv' or args.model=='convaig':
        widths = [56, 28, 14, 7]
        last_channels = [256, 512, 1024, 2048]
        first_channels = [64, 256, 512, 1024]
        first_block_strides = [1, 2, 2, 2]
        bottleneck = 4  # bottleneck ratio
        is_se = False
        n_groupss = [1 for _ in last_channels]

    if args.model == 'resnet50':
        n_block = [3, 4, 6, 3]
    elif args.model == 'resnet101':
        n_block = [3, 4, 23, 3]
    elif args.model == 'regnety004':
        n_block = [1, 3, 6, 6]
    elif args.model == 'regnety008':
        n_block = [1, 3, 8, 2]

    # static latency
    static_latency = 0
    for i_stage in range(4):
        # static latency
        static_latency_first_block = get_static_block_latency(
            predictor, c_in=first_channels[i_stage], c_out=last_channels[i_stage],
            b=bottleneck, n_groups=n_groupss[i_stage],
            h=widths[i_stage] * first_block_strides[i_stage], w=widths[i_stage] * first_block_strides[i_stage],
            stride=first_block_strides[i_stage], down=first_block_strides[i_stage], is_se=is_se)
        static_latency_other_block = get_static_block_latency(
            predictor, c_in=last_channels[i_stage], c_out=last_channels[i_stage],
            b=bottleneck, n_groups=n_groupss[i_stage],
            h=widths[i_stage], w=widths[i_stage],
            stride=1, down=1, is_se=is_se)
        static_latency += static_latency_first_block + (n_block[i_stage] - 1) * static_latency_other_block


    # spatial mode latency
    s_granul = [1, 1, 1, 1] # TODO: target spatial granularity, ex: [1, 1, 1, 1]
    s_act_rate = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] # TODO: target spaital activation rate per block, ex. [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] for ResNet50
    s_latency = 0
    for i_stage, stage in enumerate(s_act_rate):
        for j, density in enumerate(stage):
            if j == 0:
                latency = get_dynamic_block_latency_spatial(
                    predictor, 
                    c_in=first_channels[i_stage], 
                    c_out=last_channels[i_stage],
                    b=bottleneck,
                    n_groups=n_groupss[i_stage],
                    h=widths[i_stage] * first_block_strides[i_stage],
                    w=widths[i_stage] * first_block_strides[i_stage],
                    granul_size=s_granul[i_stage], 
                    c_granul_size=1,
                    density_conv1=density,
                    density_conv2=density,
                    density_conv3=density,
                    c_density=1.0,
                    stride=first_block_strides[i_stage],
                    down=first_block_strides[i_stage],
                    is_se=is_se
                )
            else:
                latency = get_dynamic_block_latency_spatial(
                    predictor,
                    c_in=last_channels[i_stage],
                    c_out=last_channels[i_stage],
                    b=bottleneck,
                    h=widths[i_stage],
                    w=widths[i_stage],
                    n_groups=n_groupss[i_stage],
                    granul_size=s_granul[i_stage], 
                    c_granul_size=1,
                    density_conv1=density,
                    density_conv2=density,
                    density_conv3=density,
                    c_density=1.0,
                    stride=1,
                    down=1,
                    is_se=is_se
                )
            s_latency += latency


    # layer skipping mode latency
    l_act_rate = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] # TODO: target layer activation rate per block, ex. [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] for ResNet50
    l_latency = 0
    for i_stage in range(4):
        for i, block in enumerate(l_act_rate[i_stage]):
            density_conv = l_act_rate[i_stage][i]
            if i == 0:
                latency = get_skipping_block_latency(
                            predictor, 
                            c_in=first_channels[i_stage], 
                            c_out=last_channels[i_stage],
                            b=bottleneck,
                            n_groups=n_groupss[i_stage],
                            h=widths[i_stage] * first_block_strides[i_stage],
                            w=widths[i_stage] * first_block_strides[i_stage],
                            granul_size=widths[i_stage], 
                            c_granul_size=1,
                            density_conv1=density_conv,
                            density_conv2=density_conv,
                            density_conv3=density_conv,
                            c_density=1,
                            stride=first_block_strides[i_stage],
                            down=first_block_strides[i_stage],
                            is_se=is_se,
                        )
            else:
                latency = get_skipping_block_latency(
                            predictor,
                            c_in=last_channels[i_stage],
                            c_out=last_channels[i_stage],
                            b=bottleneck,
                            h=widths[i_stage],
                            w=widths[i_stage],
                            n_groups=n_groupss[i_stage],
                            granul_size=widths[i_stage], 
                            c_granul_size=1,
                            density_conv1=density_conv,
                            density_conv2=density_conv,
                            density_conv3=density_conv,
                            c_density=1,
                            stride=1,
                            down=1,
                            is_se=is_se,
                        )
            l_latency += latency


    # channel mode latency
    c_granul = [1, 1, 1, 1]
    c_act_rate = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] # TODO: target channel activation rate per block, ex. [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] for ResNet50
    c_latency = 0
    for i_stage in range(4):
        for i, block in enumerate(c_act_rate[i_stage]):
            c_density = c_act_rate[i_stage][i]
            if i == 0:
                latency = get_dynamic_block_latency_channel(
                            predictor, 
                            c_in=first_channels[i_stage], 
                            c_out=last_channels[i_stage],
                            b=bottleneck,
                            n_groups=n_groupss[i_stage],
                            h=widths[i_stage] * first_block_strides[i_stage],
                            w=widths[i_stage] * first_block_strides[i_stage],
                            granul_size=s_granul[i_stage], 
                            c_granul_size=c_granul[i_stage],
                            density_conv1=1.0,
                            density_conv2=1.0,
                            density_conv3=1.0,
                            c_density=c_density,
                            stride=first_block_strides[i_stage],
                            down=first_block_strides[i_stage],
                            is_se=is_se,
                            layer=2
                        )
            else:
                latency = get_dynamic_block_latency_channel(
                            predictor,
                            c_in=last_channels[i_stage],
                            c_out=last_channels[i_stage],
                            b=bottleneck,
                            h=widths[i_stage],
                            w=widths[i_stage],
                            n_groups=n_groupss[i_stage],
                            granul_size=s_granul[i_stage], 
                            c_granul_size=c_granul[i_stage],
                            density_conv1=1.0,
                            density_conv2=1.0,
                            density_conv3=1.0,
                            c_density=c_density,
                            stride=1,
                            down=1,
                            is_se=is_se,
                            layer=2,
                        )
            c_latency += latency
