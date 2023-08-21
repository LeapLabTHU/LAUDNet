import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F


# from torchvision.models import resnet50, ResNet50_Weights


from .utils import conv1x1, conv3x3, Masker_channel_MLP, Masker_channel_conv_linear, Masker_spatial, ExpandMask, apply_channel_mask, apply_spatial_mask

__all__ = ['uni_resnet50', 'uni_resnet101']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None,
                 spatial_mask_channel_group=1,
                 channel_dyn_granularity=1,
                 output_size=56,
                 mask_spatial_granularity=1,
                 
                 dyn_mode='both',
                 channel_masker='conv_linear',
                 channel_masker_layers=2,
                 reduction=16):
        super(Bottleneck, self).__init__()
        
        assert dyn_mode in ['channel', 'spatial', 'both', 'layer']
        assert channel_masker in ['conv_linear', 'MLP']
        
        self.dyn_mode = dyn_mode
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        base_width = 64
        width = int(planes * (base_width / 64.)) * group_width
        
        assert channel_dyn_granularity <= width
        channel_dyn_group = width // channel_dyn_granularity
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, group_width, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.conv1_flops_per_pixel = inplanes*width
        self.conv2_flops_per_pixel = width*width*9 // self.conv2.groups
        self.conv3_flops_per_pixel = width*planes*self.expansion

        if self.downsample is not None:
            self.downsample_flops = inplanes * planes * self.expansion

        self.output_size = output_size
        self.mask_spatial_granularity = mask_spatial_granularity
        self.mask_size = self.output_size // self.mask_spatial_granularity if dyn_mode != 'layer' else 1
        print(self.mask_size)
        self.masker_spatial = None
        self.masker_channel = None
        
        if dyn_mode in ['spatial', 'layer', 'both']:
            self.masker_spatial = Masker_spatial(inplanes, spatial_mask_channel_group, self.mask_size)
            self.mask_expander2 = ExpandMask(stride=1, padding=0, mask_channel_group=spatial_mask_channel_group)
            self.mask_expander1 = ExpandMask(stride=stride, padding=1, mask_channel_group=spatial_mask_channel_group)
             
        if dyn_mode in ['channel', 'both']:
            if channel_masker == 'conv_linear':
                self.masker_channel = Masker_channel_conv_linear(inplanes, channel_dyn_group, reduction=reduction)
            else:
                self.masker_channel = Masker_channel_MLP(inplanes, channel_dyn_group, layers=channel_masker_layers, reduction=reduction)

    def forward(self, x, temperature=1.0):
        
        x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops = x
        identity = x
        
        if self.dyn_mode == 'channel':
            channel_mask, channel_sparsity, channel_mask_flops = self.masker_channel(x, temperature)
            spatial_sparsity_conv1, spatial_sparsity_conv2, spatial_sparsity_conv3 = torch.tensor(1.0, device=channel_sparsity.device), torch.tensor(1.0, device=channel_sparsity.device), torch.tensor(1.0, device=channel_sparsity.device)
            spatial_mask_flops = 0
        elif self.dyn_mode in ['spatial', 'layer']:
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
        
        out = self.conv1(x)
        out = apply_channel_mask(out, channel_mask) if self.dyn_mode in ['channel', 'both'] else out
        out = self.bn1(out)
        out = self.relu(out)
        
        dense_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity * spatial_sparsity_conv1
        
        out = self.conv2(out)
        out = apply_channel_mask(out, channel_mask) if self.dyn_mode in ['channel', 'both'] else out
        out = self.bn2(out)
        out = self.relu(out)
        
        dense_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity**2 * spatial_sparsity_conv2
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = apply_spatial_mask(out, spatial_mask_conv3) if self.dyn_mode in ['layer', 'spatial', 'both'] else out
        
        dense_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3] * channel_sparsity * spatial_sparsity_conv3
        
        if self.downsample is not None:
            identity = self.downsample(x)
            dense_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
            sparse_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
        
        out += identity
        out = self.relu(out)

        flops += sparse_flops
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
        
        
        return out, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, width_mult=1.,
                 input_size=224,
                 spatial_mask_channel_group=[1,1,1,1],
                 mask_spatial_granularity=[1,1,1,1],
                 channel_dyn_granularity=[1,1,1,1],
                 dyn_mode=['both','both','both','both'],
                 channel_masker=['MLP','MLP','MLP','MLP'],
                 channel_masker_layers=[1,1,1,1],
                 reduction_ratio=[16,16,16,16],
                 lr_mult=1.0,
                 **kwargs):
        super(ResNet, self).__init__()
        self.dyn_mode = dyn_mode
        assert lr_mult is not None
        self.lr_mult = lr_mult

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64*width_mult)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64*width_mult), layers[0], stride=1,
                                       dilate=False,
                                       output_size=input_size//4,
                                       spatial_mask_channel_group=spatial_mask_channel_group[0],
                                       mask_spatial_granularity=mask_spatial_granularity[0],
                                       channel_dyn_granularity=channel_dyn_granularity[0],
                                       dyn_mode=dyn_mode[0],
                                       channel_masker=channel_masker[0],
                                       channel_masker_layers=channel_masker_layers[0],
                                       reduction_ratio=reduction_ratio[0])
        
        self.layer2 = self._make_layer(block, int(128*width_mult), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       output_size=input_size//8,
                                       spatial_mask_channel_group=spatial_mask_channel_group[1],
                                       mask_spatial_granularity=mask_spatial_granularity[1],
                                       channel_dyn_granularity=channel_dyn_granularity[1],
                                       dyn_mode=dyn_mode[1],
                                       channel_masker=channel_masker[1],
                                       channel_masker_layers=channel_masker_layers[1],
                                       reduction_ratio=reduction_ratio[1])
        
        self.layer3 = self._make_layer(block, int(256*width_mult), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       output_size=input_size//16,
                                       spatial_mask_channel_group=spatial_mask_channel_group[2],
                                       mask_spatial_granularity=mask_spatial_granularity[2],
                                       channel_dyn_granularity=channel_dyn_granularity[2],
                                       dyn_mode=dyn_mode[2],
                                       channel_masker=channel_masker[2],
                                       channel_masker_layers=channel_masker_layers[2],
                                       reduction_ratio=reduction_ratio[2])
        
        self.layer4 = self._make_layer(block, int(512*width_mult), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       output_size=input_size//32,
                                       spatial_mask_channel_group=spatial_mask_channel_group[3],
                                       mask_spatial_granularity=mask_spatial_granularity[3],
                                       channel_dyn_granularity=channel_dyn_granularity[3],
                                       dyn_mode=dyn_mode[3],
                                       channel_masker=channel_masker[3],
                                       channel_masker_layers=channel_masker_layers[3],
                                       reduction_ratio=reduction_ratio[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*width_mult * block.expansion), num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'masker' not in name:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    output_size=56,
                    spatial_mask_channel_group=1,
                    mask_spatial_granularity=1,
                    channel_dyn_granularity=1,
                    dyn_mode='both',
                    channel_masker='MLP',
                    channel_masker_layers=1,
                    reduction_ratio=16):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, group_width=self.groups,
                            dilation=previous_dilation, norm_layer=norm_layer, 
                            output_size=output_size,
                            spatial_mask_channel_group=spatial_mask_channel_group,
                            mask_spatial_granularity=mask_spatial_granularity,
                            channel_dyn_granularity=channel_dyn_granularity,
                            dyn_mode=dyn_mode,
                            channel_masker=channel_masker,
                            channel_masker_layers=channel_masker_layers,
                            reduction=reduction_ratio))
        self.inplanes = planes * block.expansion
        for j in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.groups,
                                dilation=self.dilation,
                                norm_layer=norm_layer, 
                                output_size=output_size,
                                spatial_mask_channel_group=spatial_mask_channel_group,
                                mask_spatial_granularity=mask_spatial_granularity,
                                channel_dyn_granularity=channel_dyn_granularity,
                                dyn_mode=dyn_mode,
                                channel_masker=channel_masker,
                                channel_masker_layers=channel_masker_layers,
                                reduction=reduction_ratio))

        return nn.ModuleList(layers)

    def forward(self, x, temperature):
        c_in = x.shape[1]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]

        x = self.maxpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]*9
        
        spatial_sparsity_conv1_list, spatial_sparsity_conv2_list, spatial_sparsity_conv3_list, \
            channel_sparsity_list, flops_perc_list = None, None, None, None, None
        
        x = (x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x, temperature)
        x, spatial_sparsity_conv3_stage1, spatial_sparsity_conv2_stage1, spatial_sparsity_conv1_stage1, channel_sparsity_stage1, flops_perc_list, flops = x
        
        x = (x, None, None, None, None, flops_perc_list, flops)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x, temperature)
        x, spatial_sparsity_conv3_stage2, spatial_sparsity_conv2_stage2, spatial_sparsity_conv1_stage2, channel_sparsity_stage2, flops_perc_list, flops = x
        
        x = (x, None, None, None, None, flops_perc_list, flops)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x, temperature)
        x, spatial_sparsity_conv3_stage3, spatial_sparsity_conv2_stage3, spatial_sparsity_conv1_stage3, channel_sparsity_stage3, flops_perc_list, flops = x
        
        x = (x, None, None, None, None, flops_perc_list, flops)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x, temperature)
        x, spatial_sparsity_conv3_stage4, spatial_sparsity_conv2_stage4, spatial_sparsity_conv1_stage4, channel_sparsity_stage4, flops_perc_list, flops = x
        
        x = self.avgpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]
        
        x = torch.flatten(x, 1)
        
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in*x.shape[1]
        
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
                if isinstance(m, torch.nn.Conv2d):
                    ps = list(m.parameters())
                    masker_params.append(ps[0]) # ps[0] is a tensor, use append
                    if len(ps) == 2:
                        masker_params.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    masker_params.extend(list(m.parameters()))  # this is a list, use extend
                elif isinstance(m, torch.nn.BatchNorm1d):
                    masker_params.extend(list(m.parameters()))  # this is a list, use extend
                elif isinstance(m, torch.nn.Linear):
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


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def uni_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 50')
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def uni_resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 101')
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)
