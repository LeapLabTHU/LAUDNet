# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import math
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from ..layers import ResLayer_sd







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
                 mask_channel_group=[1],
                 mask_spatial_granularity=[1],
                 mask_target_sparsity=[1]
                 ):
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
        self.dyn_mode = 'spatial'

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

        self.mask_spatial_granularity = mask_spatial_granularity
        self.masker = Masker(inplanes, mask_channel_group, mask_spatial_granularity, dilate_stride=self.conv2_stride)

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
            out = getattr(self, name)(out)
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
            x, sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops = x

            identity = x
        
            mask, mask_dil, sparsity, sparsity_dil, mask_flops = self.masker(x, temperature=temperature)
            sparse_flops = mask_flops
            dense_flops = mask_flops

            if sparsity_list == None:
                sparsity_list = sparsity.unsqueeze(0)
            else:
                sparsity_list = torch.cat((sparsity_list, sparsity.unsqueeze(0)), dim=0)
            if sparsity_list_dil == None:
                sparsity_list_dil = sparsity_dil.unsqueeze(0)
            else:
                sparsity_list_dil = torch.cat((sparsity_list_dil, sparsity_dil.unsqueeze(0)), dim=0)

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            dense_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3]
            sparse_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3] * sparsity_dil

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            dense_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3]
            sparse_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3] * sparsity

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)
            dense_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3]
            sparse_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3] * sparsity

            mask = F.interpolate(mask, size=(out.shape[2], out.shape[3]), mode='nearest')
            out = apply_mask(out, mask)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)
                dense_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
                sparse_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]

            out += identity

            flops += sparse_flops
            overall_dense_flops += dense_flops
            perc = sparse_flops / dense_flops

            if flops_perc_list == None:
                flops_perc_list = perc.unsqueeze(0)
            else:
                flops_perc_list = torch.cat((flops_perc_list,perc.unsqueeze(0)),dim=0)

            return out, sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops

        if self.with_cp and x.requires_grad:
            out, sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops = cp.checkpoint(_inner_forward, x)
        else:
            out, sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops = _inner_forward(x)

        out = self.relu(out)

        return out, sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops


@MODELS.register_module()
class ResNet_sd(BaseModule):
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
                 temperature_0=None,
                 temperature_t=None,
                 mask_channel_group=[1],
                 mask_spatial_granularity=[1],
                 mask_target_sparsity=[1]):
        super(ResNet_sd, self).__init__(init_cfg)
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
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
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
        self.mask_target_sparsity = mask_target_sparsity

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        num_blocks_accumulative = 0
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
                mask_channel_group=mask_channel_group[num_blocks_accumulative: num_blocks_accumulative+num_blocks],
                mask_spatial_granularity=mask_spatial_granularity[num_blocks_accumulative: num_blocks_accumulative+num_blocks],
                mask_target_sparsity=mask_target_sparsity[num_blocks_accumulative: num_blocks_accumulative+num_blocks])
            num_blocks_accumulative += num_blocks
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
        """Pack all blocks in a stage into a ``ResLayer_sd``."""
        return ResLayer_sd(**kwargs)

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

    def forward(self, x):
        c_in = x.shape[1]
        
        # t0 = self.temperature_0
        # t_last = self.temperature_t
        # T_total = 12 * len_loader
        # T_cur = iter_now
        # alpha = math.pow(t_last / t0, 1 / T_total)
        # temperature = math.pow(alpha, T_cur) * t0

        temperature = self.temperature_0

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]
        overall_dense_flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]

        x = self.maxpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]*9
        overall_dense_flops += x.shape[1]*x.shape[2]*x.shape[3]*9
        
        sparsity_list = None
        flops_perc_list = None
        sparsity_list_dil = None

        outs = []
        x = (x, sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            for j in range(len(res_layer)):
                x = res_layer[j](x, temperature)
            if i in self.out_indices:
                outs.append(x[0])
        
        x, sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops = x
        target_sparsity_rate = self.mask_target_sparsity[0]
        # others = (sparsity_list, sparsity_list_dil, flops_perc_list, flops, overall_dense_flops, target_sparsity_rate)

        additional = {
            'spatial_sparsity_conv3': sparsity_list,
            'spatial_sparsity_conv2': sparsity_list,
            'spatial_sparsity_conv1': sparsity_list_dil,
            'channel_sparsity': [1.0 for _ in range(len(sparsity_list_dil))],
            'flops_perc_list': flops_perc_list,
            'flops': flops,
            'dense_flops': overall_dense_flops,
        }

        model_configs = {
            "dyn_mode": "spatial",
            "sparsity_target": target_sparsity_rate,
        }

        return tuple(outs), additional, model_configs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet_sd, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def apply_mask(x, mask):
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape
    if (g > 1) and (g != c):
        # print(g)
        mask = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,hw_mask,hw_mask)
    
    # if (hw_mask < h) and (hw_mask > 1):
    #     mask = F.interpolate(mask, size = (h,w))
    return x * mask


class Masker(nn.Module):
    def __init__(self, in_channels, mask_channel_group, mask_spatial_granularity, dilate_stride=1):
        super(Masker, self).__init__()
        self.mask_channel_group = mask_channel_group
        # self.conv2 = conv1x1(in_channels, mask_channel_group*2,bias=True)
        self.conv2 = nn.Conv2d(in_channels, mask_channel_group*2, kernel_size=1, stride=1, bias=True)
        
        self.conv2_flops_pp = self.conv2.weight.shape[0] * self.conv2.weight.shape[1] + self.conv2.weight.shape[1]
        
        
        # trunc_normal_(self.conv1.weight, std=.01)
        # trunc_normal_(self.conv2.weight, std=.01)
        # init_gate = 0.95
        # num_splits = 2
        # bias_value = math.log(math.sqrt(init_gate * (1 - num_splits) / (init_gate - 1)))
        self.conv2.bias.data[:mask_channel_group] = 5.0
        self.conv2.bias.data[mask_channel_group+1:] = 1.0
        # self.feature_size = feature_size
        self.expandmask = ExpandMask(stride=dilate_stride, mask_channel_group=mask_channel_group)
        self.mask_spatial_granularity = mask_spatial_granularity
        self.stride = dilate_stride

    def forward(self, x, temperature):

        # print(x.shape)
        mask_size = (x.shape[2] // self.stride // self.mask_spatial_granularity, x.shape[3] // self.stride // self.mask_spatial_granularity)
        feature_size = (x.shape[2]//self.stride, x.shape[3]//self.stride)
        
        mask =  F.adaptive_avg_pool2d(x, mask_size) if mask_size[0] < x.shape[2] else x
        flops = mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        mask = self.conv2(mask)
        flops += self.conv2_flops_pp * mask.shape[2] * mask.shape[3]
        
        b,c,h,w = mask.shape
        mask = mask.view(b,2,c//2,h,w)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        sparsity = mask.sum() / mask.numel()
        
        if h < feature_size[0]:
            mask = F.interpolate(mask, size=feature_size)

        mask_dil = self.expandmask(mask)
        sparsity_dil = mask_dil.sum() / mask_dil.numel()
        
        return mask, mask_dil, sparsity, sparsity_dil, flops

class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1, mask_channel_group=1): 
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding
        self.mask_channel_group = mask_channel_group
        
    def forward(self, x):
        if self.stride > 1:
            self.pad_kernel = torch.zeros((self.mask_channel_group,1,self.stride, self.stride), device=x.device)
            self.pad_kernel[:,:,0,0] = 1
        self.dilate_kernel = torch.ones((self.mask_channel_group,self.mask_channel_group,1+2*self.padding,1+2*self.padding), device=x.device)

        x = x.float()
        
        if self.stride > 1:
            # print(x.shape, self.pad_kernel.shape)
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
            # print(x.shape)
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5
