# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F


from ..builder import BACKBONES
from ..utils import LAD_MMDet_Reslayer

from .utils import Masker_channel_MLP, Masker_channel_conv_linear, Masker_spatial, ExpandMask, apply_channel_mask, apply_spatial_mask


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None,
                 spatial_mask_channel_group=1,
                 channel_dyn_granularity=1,
                 output_size=56,
                 mask_spatial_granularity=1,
                 group_width=1,

                 dyn_mode='both',
                 channel_masker='conv_linear',
                 channel_masker_layers=2,
                 reduction=16):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None
        self.dyn_mode = dyn_mode
        self.output_size = output_size

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)


        self.conv1_flops_per_pixel = inplanes * planes
        self.conv2_flops_per_pixel = planes * planes * 9 // self.conv2.groups
        self.conv3_flops_per_pixel = planes * planes*self.expansion

        if self.downsample is not None:
            self.downsample_flops = inplanes * planes * self.expansion

        if dyn_mode == 'channel':
            base_width = 64
            width = int(planes * (base_width / 64.)) * group_width
            
            assert channel_dyn_granularity <= width
            channel_dyn_group = width // channel_dyn_granularity
        
            if channel_masker == 'conv_linear':
                self.masker_channel = Masker_channel_conv_linear(inplanes, channel_dyn_group, reduction=reduction)
            else:
                self.masker_channel = Masker_channel_MLP(inplanes, channel_dyn_group, layers=channel_masker_layers, reduction=reduction)

        if dyn_mode == 'layer':
            self.masker_spatial = Masker_spatial(inplanes, spatial_mask_channel_group, 1)
            self.mask_expander2 = ExpandMask(stride=1, padding=0, mask_channel_group=spatial_mask_channel_group)
            self.mask_expander1 = ExpandMask(stride=stride, padding=1, mask_channel_group=spatial_mask_channel_group)


    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x, temperature):
        """Forward function."""

        def _inner_forward(x):
            x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops, dense_flops_tot = x
            identity = x

            if self.dyn_mode == 'channel':
                channel_mask, channel_sparsity, channel_mask_flops = self.masker_channel(x, temperature)
                spatial_sparsity_conv1, spatial_sparsity_conv2, spatial_sparsity_conv3 = torch.tensor(1.0, device=channel_sparsity.device), torch.tensor(1.0, device=channel_sparsity.device), torch.tensor(1.0, device=channel_sparsity.device)
                spatial_mask_flops = 0
            elif self.dyn_mode in ['spatial', 'layer']:
                spatial_mask_conv3, spatial_sparsity_conv3, spatial_mask_flops = self.masker_spatial(x, temperature)
                channel_sparsity = torch.tensor(1.0, device=spatial_mask_conv3.device)
                channel_mask_flops = 0

            # if self.dyn_mode != 'channel':
            #     spatial_mask_conv3 = F.interpolate(spatial_mask_conv3, size=(x.shape[-2], x.shape[-1]), mode='nearest')
            #     spatial_mask_conv2 = self.mask_expander2(spatial_mask_conv3)
            #     spatial_sparsity_conv2 = spatial_mask_conv2.float().mean()
            #     spatial_mask_conv1 = self.mask_expander1(spatial_mask_conv2)
            #     spatial_sparsity_conv1 = spatial_mask_conv1.float().mean()

            sparse_flops = channel_mask_flops + spatial_mask_flops
            dense_flops = channel_mask_flops + spatial_mask_flops
            out = self.conv1(x)
            out = apply_channel_mask(out, channel_mask) if self.dyn_mode == 'channel' else out
            out = self.norm1(out)
            out = self.relu(out)
            conv1_h, conv1_w = out.shape[2], out.shape[3]
            # dense_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3]
            # sparse_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity * spatial_sparsity_conv1

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = apply_channel_mask(out, channel_mask) if self.dyn_mode == 'channel' else out
            out = self.norm2(out)
            out = self.relu(out)
            conv2_h, conv2_w = out.shape[2], out.shape[3]
            # dense_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3]
            # sparse_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity**2 * spatial_sparsity_conv2

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.dyn_mode != 'channel':
                spatial_mask_conv3 = F.interpolate(spatial_mask_conv3, size=(out.shape[2], out.shape[3]), mode='nearest')
                spatial_mask_conv2 = self.mask_expander2(spatial_mask_conv3)
                spatial_sparsity_conv2 = spatial_mask_conv2.float().mean()
                spatial_mask_conv1 = self.mask_expander1(spatial_mask_conv2)
                spatial_sparsity_conv1 = spatial_mask_conv1.float().mean()

            dense_flops += self.conv1_flops_per_pixel * conv1_h * conv1_w
            sparse_flops += self.conv1_flops_per_pixel * conv1_h * conv1_w * channel_sparsity * spatial_sparsity_conv1

            dense_flops += self.conv2_flops_per_pixel * conv2_h * conv2_w
            sparse_flops += self.conv2_flops_per_pixel * conv2_h * conv2_w * channel_sparsity**2 * spatial_sparsity_conv2

            out = apply_spatial_mask(out, spatial_mask_conv3) if self.dyn_mode in ['layer'] else out
            dense_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3]
            sparse_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity * spatial_sparsity_conv3

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)
                dense_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
                sparse_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]

            out += identity

            flops += sparse_flops
            dense_flops_tot += dense_flops
            flops_perc = sparse_flops / dense_flops

            spatial_sparsity_conv3_list = spatial_sparsity_conv3.unsqueeze(0) if spatial_sparsity_conv3_list is None else \
                torch.cat((spatial_sparsity_conv3_list,spatial_sparsity_conv3.unsqueeze(0)), dim=0)
            
            spatial_sparsity_conv2_list = spatial_sparsity_conv2.unsqueeze(0) if spatial_sparsity_conv2_list is None else \
                torch.cat((spatial_sparsity_conv2_list,spatial_sparsity_conv2.unsqueeze(0)), dim=0)
            
            spatial_sparsity_conv1_list = spatial_sparsity_conv1.unsqueeze(0) if spatial_sparsity_conv1_list is None else \
                torch.cat((spatial_sparsity_conv1_list,spatial_sparsity_conv1.unsqueeze(0)), dim=0)
            
            channel_sparsity_list = channel_sparsity.unsqueeze(0) if channel_sparsity_list is None else \
                torch.cat((channel_sparsity_list,channel_sparsity.unsqueeze(0)), dim=0)
                
            flops_perc_list = flops_perc.unsqueeze(0) if flops_perc_list is None else \
                torch.cat((flops_perc_list,flops_perc.unsqueeze(0)), dim=0)

            return out, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops, dense_flops_tot

        if self.with_cp and x.requires_grad:
            out, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops, dense_flops_tot = cp.checkpoint(_inner_forward, x)
        else:
            out, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops, dense_flops_tot = _inner_forward(x)

        out = self.relu(out)

        return out, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops, dense_flops_tot


@BACKBONES.register_module()
class LAD_MMDet_ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        # 18: (BasicBlock, (2, 2, 2, 2)),
        # 34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 sparsity_target=None,
                 temperature_0=None,
                 temperature_t=None,
                 spatial_mask_channel_group=[1,1,1,1],
                 mask_spatial_granularity=[1,1,1,1],
                 channel_dyn_granularity=[1,1,1,1],
                 dyn_mode=['both','both','both','both'],
                 channel_masker=['MLP','MLP','MLP','MLP'],
                 channel_masker_layers=[1,1,1,1],
                 reduction_ratio=[16,16,16,16]):
        super(LAD_MMDet_ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    # if block is BasicBlock:
                    #     block_init_cfg = dict(
                    #         type='Constant',
                    #         val=0,
                    #         override=dict(name='norm2'))
                    if block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.sparsity_target = sparsity_target
        self.dyn_mode = dyn_mode

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
                spatial_mask_channel_group=spatial_mask_channel_group[i],
                channel_dyn_granularity=channel_dyn_granularity[i],
                mask_spatial_granularity=mask_spatial_granularity[i],

                dyn_mode=dyn_mode[i],
                channel_masker=channel_masker[i],
                channel_masker_layers=channel_masker_layers[i],
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

        self.temperature_0 = temperature_0
        self.temperature_t = temperature_t

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return LAD_MMDet_Reslayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, iter_now=0, len_loader=100):
        c_in = x.shape[1]
        temperature = self.temperature_0
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]
        dense_flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]

        x = self.maxpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]*9
        dense_flops += x.shape[1]*x.shape[2]*x.shape[3]*9

        spatial_sparsity_conv1_list, spatial_sparsity_conv2_list, spatial_sparsity_conv3_list, \
            channel_sparsity_list, flops_perc_list = None, None, None, None, None

        outs = []
        # for i, layer_name in enumerate(self.res_layers):
        #     res_layer = getattr(self, layer_name)
        #     x = res_layer(x)
        #     if i in self.out_indices:
        #         outs.append(x[0])
        x = (x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops, dense_flops)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x, temperature)
        x, spatial_sparsity_conv3_stage1, spatial_sparsity_conv2_stage1, spatial_sparsity_conv1_stage1, channel_sparsity_stage1, flops_perc_list, flops, dense_flops = x
        outs.append(x)

        x = (x, None, None, None, None, flops_perc_list, flops, dense_flops)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x, temperature)
        x, spatial_sparsity_conv3_stage2, spatial_sparsity_conv2_stage2, spatial_sparsity_conv1_stage2, channel_sparsity_stage2, flops_perc_list, flops, dense_flops = x
        outs.append(x)
        
        x = (x, None, None, None, None, flops_perc_list, flops, dense_flops)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x, temperature)
        x, spatial_sparsity_conv3_stage3, spatial_sparsity_conv2_stage3, spatial_sparsity_conv1_stage3, channel_sparsity_stage3, flops_perc_list, flops, dense_flops = x
        outs.append(x)
        
        x = (x, None, None, None, None, flops_perc_list, flops, dense_flops)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x, temperature)
        x, spatial_sparsity_conv3_stage4, spatial_sparsity_conv2_stage4, spatial_sparsity_conv1_stage4, channel_sparsity_stage4, flops_perc_list, flops, dense_flops = x
        # x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops = x
        outs.append(x)
        
        spatial_sparsity_conv3 = [spatial_sparsity_conv3_stage1, spatial_sparsity_conv3_stage2, spatial_sparsity_conv3_stage3, spatial_sparsity_conv3_stage4]
        spatial_sparsity_conv2 = [spatial_sparsity_conv2_stage1, spatial_sparsity_conv2_stage2, spatial_sparsity_conv2_stage3, spatial_sparsity_conv2_stage4]
        spatial_sparsity_conv1 = [spatial_sparsity_conv1_stage1, spatial_sparsity_conv1_stage2, spatial_sparsity_conv1_stage3, spatial_sparsity_conv1_stage4]
        channel_sparsity = [channel_sparsity_stage1, channel_sparsity_stage2, channel_sparsity_stage3, channel_sparsity_stage4]

        additional = {
            'spatial_sparsity_conv3': spatial_sparsity_conv3,
            'spatial_sparsity_conv2': spatial_sparsity_conv2,
            'spatial_sparsity_conv1': spatial_sparsity_conv1,
            'channel_sparsity': channel_sparsity,
            'flops_perc_list': flops_perc_list,
            'flops': flops,
            'dense_flops': dense_flops,
        }

        model_configs = {
            "dyn_mode": self.dyn_mode,
            "sparsity_target": self.sparsity_target,
        }

        return tuple(outs), additional, model_configs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(LAD_MMDet_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
