# Modified from
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/anynet.py
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py


import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from torchvision.models._utils import _make_divisible
import torch.nn.functional as F
from .utils import Masker_channel_MLP, Masker_channel_conv_linear, Masker_spatial, ExpandMask, apply_channel_mask, apply_spatial_mask


__all__ = [
    "LAD_RegNet",
    "lad_regnet_y_400mf",
    "lad_regnet_y_800mf",
    "lad_regnet_y_1_6gf",
    "lad_regnet_y_3_2gf",
    "lad_regnet_y_8gf",
    "lad_regnet_y_16gf",
    "lad_regnet_y_32gf",
    "lad_regnet_y_128gf",
    "lad_regnet_x_400mf",
    "lad_regnet_x_800mf",
    "lad_regnet_x_1_6gf",
    "lad_regnet_x_3_2gf",
    "lad_regnet_x_8gf",
    "lad_regnet_x_16gf",
    "lad_regnet_x_32gf",
]


model_urls = {
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
    "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
    "regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
    "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
    "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
    "regnet_y_16gf": "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
    "regnet_y_32gf": "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
    "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
    "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
    "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
    "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
    "regnet_x_8gf": "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
    "regnet_x_16gf": "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
    "regnet_x_32gf": "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
}


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],

        spatial_mask_channel_group=1,
        channel_dyn_granularity=1,
        output_size=56,
        mask_spatial_granularity=1,
        
        dyn_mode='both',
        channel_masker='conv_linear',
        channel_masker_layers=2,
        reduction=16,
    ) -> None:
        super(BottleneckTransform, self).__init__()

        assert dyn_mode in ['channel', 'spatial', 'both']
        assert channel_masker in ['conv_linear', 'MLP']

        self.dyn_mode = dyn_mode

        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        self.a = ConvNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        self.b = ConvNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            self.se = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        self.c = ConvNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )
        
        assert channel_dyn_granularity <= w_b
        channel_dyn_group = w_b // channel_dyn_granularity
        
        self.conv1_flops_per_pixel = width_in * w_b
        self.conv2_flops_per_pixel = w_b * w_b *9 // g
        self.conv3_flops_per_pixel = w_b * width_out
        self.se_flops_per_pixel = w_b * width_se_out * 2 if se_ratio else 0

        self.output_size = output_size
        self.mask_spatial_granularity = mask_spatial_granularity
        self.mask_size = self.output_size // self.mask_spatial_granularity

        self.masker_spatial = None
        self.masker_channel = None

        if dyn_mode in ['spatial', 'both']:
            self.masker_spatial = Masker_spatial(width_in, spatial_mask_channel_group, self.mask_size)
            self.mask_expander2 = ExpandMask(stride=1, padding=0, mask_channel_group=spatial_mask_channel_group)
            self.mask_expander1 = ExpandMask(stride=stride, padding=1, mask_channel_group=spatial_mask_channel_group)

        if dyn_mode in ['channel', 'both']:
            if channel_masker == 'conv_linear':
                self.masker_channel = Masker_channel_conv_linear(width_in, channel_dyn_group, reduction=reduction)
            else:
                self.masker_channel = Masker_channel_MLP(width_in, channel_dyn_group, layers=channel_masker_layers, reduction=reduction)
            # print(f'w_b: {w_b}, channel_dyn_group: {channel_dyn_group}')
            # print(self.masker_channel)

    def forward(self, x, temperature=1.0):
        x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops = x
        
        if self.dyn_mode == 'channel':
            channel_mask, channel_sparsity, channel_mask_flops = self.masker_channel(x, temperature)
            spatial_sparsity_conv1, spatial_sparsity_conv2, spatial_sparsity_conv3 = torch.tensor(1.0, device=channel_sparsity.device), torch.tensor(1.0, device=channel_sparsity.device), torch.tensor(1.0, device=channel_sparsity.device)
            spatial_mask_flops = 0
        elif self.dyn_mode == 'spatial':
            spatial_mask_conv3, spatial_sparsity_conv3, spatial_mask_flops = self.masker_spatial(x, temperature)
            channel_sparsity = torch.tensor(1.0, device=spatial_mask_conv3.device)
            channel_mask_flops = 0
        else:
            channel_mask, channel_sparsity, channel_mask_flops = self.masker_channel(x, temperature)
            spatial_mask_conv3, spatial_sparsity_conv3, spatial_mask_flops = self.masker_spatial(x, temperature)
        
        if self.dyn_mode != 'channel':
            spatial_mask_conv3 = F.interpolate(spatial_mask_conv3, size=self.output_size, mode='nearest')
            spatial_mask_conv2 = self.mask_expander2(spatial_mask_conv3)
            spatial_sparsity_conv2 = spatial_mask_conv2.float().mean()
            spatial_mask_conv1 = self.mask_expander1(spatial_mask_conv2)
            spatial_sparsity_conv1 = spatial_mask_conv1.float().mean()
        
        sparse_flops = channel_mask_flops + spatial_mask_flops
        dense_flops = channel_mask_flops + spatial_mask_flops

        out = self.a(x)
        out = apply_channel_mask(out, channel_mask) if self.dyn_mode != 'spatial' else out

        dense_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity * spatial_sparsity_conv1

        out = self.b(out)
        out = apply_channel_mask(out, channel_mask) if self.dyn_mode != 'spatial' else out

        dense_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity**2 * spatial_sparsity_conv2

        out = self.se(out)
        flops += self.se_flops_per_pixel

        out = self.c(out)
        out = apply_spatial_mask(out, spatial_mask_conv3) if self.dyn_mode != 'channel' else out
        
        dense_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity * spatial_sparsity_conv3

        flops += sparse_flops

        spatial_sparsity_conv3_list = spatial_sparsity_conv3.unsqueeze(0) if spatial_sparsity_conv3_list is None else \
            torch.cat((spatial_sparsity_conv3_list,spatial_sparsity_conv3.unsqueeze(0)), dim=0)
        
        spatial_sparsity_conv2_list = spatial_sparsity_conv2.unsqueeze(0) if spatial_sparsity_conv2_list is None else \
            torch.cat((spatial_sparsity_conv2_list,spatial_sparsity_conv2.unsqueeze(0)), dim=0)
        
        spatial_sparsity_conv1_list = spatial_sparsity_conv1.unsqueeze(0) if spatial_sparsity_conv1_list is None else \
            torch.cat((spatial_sparsity_conv1_list,spatial_sparsity_conv1.unsqueeze(0)), dim=0)
        
        channel_sparsity_list = channel_sparsity.unsqueeze(0) if channel_sparsity_list is None else \
            torch.cat((channel_sparsity_list,channel_sparsity.unsqueeze(0)), dim=0)
                
        return out, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, sparse_flops, dense_flops, flops



class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,

        spatial_mask_channel_group=1,
        channel_dyn_granularity=1,
        output_size=56,
        mask_spatial_granularity=1,
        
        dyn_mode='both',
        channel_masker='conv_linear',
        channel_masker_layers=2,
        reduction=16,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,

            spatial_mask_channel_group,
            channel_dyn_granularity,
            output_size,
            mask_spatial_granularity,
            
            dyn_mode,
            channel_masker,
            channel_masker_layers,
            reduction,
        )
        self.activation = activation_layer(inplace=True)

        self.dyn_mode = dyn_mode

        if self.proj is not None:
            self.downsample_flops = width_in * width_out

    def forward(self, x, temperature):
        x_tensor, _, _, _, _, _, _ = x
        x_new, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, sparse_flops, dense_flops, flops = self.f(x, temperature=temperature)
        if self.proj is not None:
            x_tensor = self.proj(x_tensor) + x_new
            sparse_flops += self.downsample_flops * x_tensor.shape[2] * x_tensor.shape[3]
            dense_flops += self.downsample_flops * x_tensor.shape[2] * x_tensor.shape[3]
            flops += self.downsample_flops * x_tensor.shape[2] * x_tensor.shape[3]
        else:
            x_tensor = x_tensor + x_new

        flops_perc = sparse_flops / dense_flops
        flops_perc_list = flops_perc.unsqueeze(0) if flops_perc_list is None else \
            torch.cat((flops_perc_list,flops_perc.unsqueeze(0)), dim=0)
        return self.activation(x_tensor), spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,

        spatial_mask_channel_group=1,
        channel_dyn_granularity=1,
        output_size=56,
        mask_spatial_granularity=1,
        
        dyn_mode='both',
        channel_masker='conv_linear',
        channel_masker_layers=2,
        reduction=16,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,

                spatial_mask_channel_group,
                channel_dyn_granularity,
                output_size,
                mask_spatial_granularity,
                
                dyn_mode,
                channel_masker,
                channel_masker_layers,
                reduction,
            )

            self.add_module(f"block{stage_index}-{i}", block)
    
    def forward(self, x, temperature):
        for layer in self.children():
            x = layer(x, temperature)
        return x


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class LAD_RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,

        input_size=224,
        spatial_mask_channel_group=[1,1,1,1],
        mask_spatial_granularity=[1,1,1,1],
        channel_dyn_granularity=[1,1,1,1],
        dyn_mode=['both','both','both','both'],
        channel_masker=['MLP','MLP','MLP','MLP'],
        channel_masker_layers=[1,1,1,1],
        reduction_ratio=[16,16,16,16],
        lr_mult=1.0,
        **kwargs,
    ) -> None:
        super(LAD_RegNet, self).__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        self.dyn_mode = dyn_mode
        assert lr_mult is not None
        self.lr_mult = lr_mult

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            print(stride, depth)
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,

                        spatial_mask_channel_group=spatial_mask_channel_group[i],
                        channel_dyn_granularity=channel_dyn_granularity[i],
                        output_size=input_size//(2**(i + 2)),
                        mask_spatial_granularity=mask_spatial_granularity[i],
                        
                        dyn_mode=dyn_mode[i],
                        channel_masker=channel_masker[i],
                        channel_masker_layers=channel_masker_layers[i],
                        reduction=reduction_ratio[i],
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Performs ResNet-style weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'masker' not in name:
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and 'masker' not in name:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, temperature):
        c_in = x.shape[1]
        x = self.stem(x)
        flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * (3 * 3)

        spatial_sparsity_conv1_list, spatial_sparsity_conv2_list, spatial_sparsity_conv3_list, \
            channel_sparsity_list, flops_perc_list = None, None, None, None, None
        
        x = (x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops)
        x = self.trunk_output.block1(x, temperature)
        x, spatial_sparsity_conv3_stage1, spatial_sparsity_conv2_stage1, spatial_sparsity_conv1_stage1, channel_sparsity_stage1, flops_perc_list, flops = x
        
        x = (x, None, None, None, None, flops_perc_list, flops)
        x = self.trunk_output.block2(x, temperature)
        x, spatial_sparsity_conv3_stage2, spatial_sparsity_conv2_stage2, spatial_sparsity_conv1_stage2, channel_sparsity_stage2, flops_perc_list, flops = x
        
        x = (x, None, None, None, None, flops_perc_list, flops)
        x = self.trunk_output.block3(x, temperature)
        x, spatial_sparsity_conv3_stage3, spatial_sparsity_conv2_stage3, spatial_sparsity_conv1_stage3, channel_sparsity_stage3, flops_perc_list, flops = x
        
        x = (x, None, None, None, None, flops_perc_list, flops)
        x = self.trunk_output.block4(x, temperature)
        x, spatial_sparsity_conv3_stage4, spatial_sparsity_conv2_stage4, spatial_sparsity_conv1_stage4, channel_sparsity_stage4, flops_perc_list, flops = x

        x = self.avgpool(x)
        flops += x.shape[1] * x.shape[2] * x.shape[3]

        x = x.flatten(start_dim=1)

        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in * x.shape[1]

        spatial_sparsity_conv3 = [spatial_sparsity_conv3_stage1, spatial_sparsity_conv3_stage2, spatial_sparsity_conv3_stage3, spatial_sparsity_conv3_stage4]
        spatial_sparsity_conv2 = [spatial_sparsity_conv2_stage1, spatial_sparsity_conv2_stage2, spatial_sparsity_conv2_stage3, spatial_sparsity_conv2_stage4]
        spatial_sparsity_conv1 = [spatial_sparsity_conv1_stage1, spatial_sparsity_conv1_stage2, spatial_sparsity_conv1_stage3, spatial_sparsity_conv1_stage4]
        channel_sparsity = [channel_sparsity_stage1, channel_sparsity_stage2, channel_sparsity_stage3, channel_sparsity_stage4]


        return x, spatial_sparsity_conv3, spatial_sparsity_conv2, spatial_sparsity_conv1, channel_sparsity, flops_perc_list, flops

    def get_optim_policies(self):
        backbone_params = []
        masker_params = []

        for name, m in self.named_modules():
            if 'masker' in name:
                # print(name)
                if isinstance(m, torch.nn.Conv2d):
                    # print(f'{name}: conv2d')
                    ps = list(m.parameters())
                    masker_params.append(ps[0]) # ps[0] is a tensor, use append
                    if len(ps) == 2:
                        masker_params.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    # print(f'{name}: BN-2D')
                    masker_params.extend(list(m.parameters()))  # this is a list, use extend
                elif isinstance(m, torch.nn.BatchNorm1d):
                    # print(f'{name}: BN-1D')
                    masker_params.extend(list(m.parameters()))  # this is a list, use extend
                elif isinstance(m, torch.nn.Linear):
                    # print(f'{name}: linear')
                    ps = list(m.parameters())
                    masker_params.append(ps[0])
                    if len(ps) == 2:
                        masker_params.append(ps[1])
            else:
                if isinstance(m, torch.nn.Conv2d):
                    ps = list(m.parameters())
                    backbone_params.append(ps[0]) # ps[0] is a tensor, use append
                    if len(ps) == 2:
                        backbone_params.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    backbone_params.extend(list(m.parameters()))  # this is a list, use extend
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    backbone_params.append(ps[0])
                    if len(ps) == 2:
                        backbone_params.append(ps[1])
        return [
            {'params': backbone_params, 'lr_mult': self.lr_mult, 'decay_mult': 1.0, 'name': "backbone_params"},
            {'params': masker_params, 'lr_mult': 1.0, 'decay_mult': 1.0, 'name': "masker_params"},
        ]



def _lad_regnet(arch: str, block_params: BlockParams, pretrained: bool, progress: bool, **kwargs: Any) -> LAD_RegNet:
    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = LAD_RegNet(block_params, norm_layer=norm_layer, **kwargs)
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def lad_regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
    return _lad_regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)


def lad_regnet_y_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
    return _lad_regnet("regnet_y_800mf", params, pretrained, progress, **kwargs)


def lad_regnet_y_1_6gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_1.6GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
    )
    return _lad_regnet("regnet_y_1_6gf", params, pretrained, progress, **kwargs)


def lad_regnet_y_3_2gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_3.2GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
    )
    return _lad_regnet("regnet_y_3_2gf", params, pretrained, progress, **kwargs)


def lad_regnet_y_8gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_8GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return _lad_regnet("regnet_y_8gf", params, pretrained, progress, **kwargs)


def lad_regnet_y_16gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_16GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs
    )
    return _lad_regnet("regnet_y_16gf", params, pretrained, progress, **kwargs)


def lad_regnet_y_32gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_32GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_32gf", params, pretrained, progress, **kwargs)


def lad_regnet_y_128gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetY_128GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.
    NOTE: Pretrained weights are not available for this model.
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25, **kwargs
    )
    return _lad_regnet("regnet_y_128gf", params, pretrained, progress, **kwargs)


def lad_regnet_x_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetX_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return _lad_regnet("regnet_x_400mf", params, pretrained, progress, **kwargs)


def lad_regnet_x_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetX_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
    return _lad_regnet("regnet_x_800mf", params, pretrained, progress, **kwargs)


def lad_regnet_x_1_6gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetX_1.6GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs)
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def lad_regnet_x_3_2gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetX_3.2GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)
    return _lad_regnet("regnet_x_3_2gf", params, pretrained, progress, **kwargs)


def lad_regnet_x_8gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetX_8GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs)
    return _lad_regnet("regnet_x_8gf", params, pretrained, progress, **kwargs)


def lad_regnet_x_16gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetX_16GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs)
    return _lad_regnet("regnet_x_16gf", params, pretrained, progress, **kwargs)


def lad_regnet_x_32gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LAD_RegNet:
    """
    Constructs a RegNetX_32GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs)
    return _lad_regnet("regnet_x_32gf", params, pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = lad_regnet_y_400mf(pretrained=True,
                         spatial_mask_channel_group=[1,1,1,1],
                         mask_spatial_granularity=[1,1,1,1],
                         channel_dyn_granularity=[1,1,1,1],
                         dyn_mode=['channel', 'channel', 'channel', 'channel'],
                         channel_masker=['MLP','MLP','MLP','MLP'],
                        channel_masker_layers=[2,2,2,2],
                        reduction_ratio=[16,16,16,16])
    # print(model)
    x = torch.rand(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        y, spatial_sparsity_conv3, spatial_sparsity_conv2, \
            spatial_sparsity_conv1, channel_sparsity, flops_perc_list, flops = model(x, temperature=0.1)
    
    print('channel:', channel_sparsity)
    print('spatial_conv1:', spatial_sparsity_conv1)
    print('spatial_conv2:', spatial_sparsity_conv2)
    print('spatial_conv3:', spatial_sparsity_conv3)
    print('flops:', flops_perc_list)
    print(flops/1e9)