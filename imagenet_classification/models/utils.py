import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def apply_channel_mask(x, mask):
    b, c, h, w = x.shape
    _, g = mask.shape
    if (g > 1) and (g != c):
        mask = mask.repeat(1,c//g).view(b, c//g, g).transpose(-1,-2).reshape(b,c,1,1)
    else:
        mask = mask.view(b,g,1,1)
    return x * mask

def apply_spatial_mask(x, mask):
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape
    if (g > 1) and (g != c):
        mask = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,hw_mask,hw_mask)
        # print(mask)
    return x * mask

class Masker_spatial(nn.Module):
    def __init__(self, in_channels, mask_channel_group, mask_size):
        super(Masker_spatial, self).__init__()
        self.mask_channel_group = mask_channel_group
        self.mask_size = mask_size
        self.conv = conv1x1(in_channels, mask_channel_group*2,bias=True)
        self.conv_flops_pp = self.conv.weight.shape[0] * self.conv.weight.shape[1] + self.conv.weight.shape[1]
        self.conv.bias.data[:mask_channel_group] = 5.0
        self.conv.bias.data[mask_channel_group+1:] = 0.0
        # self.feature_size = feature_size
        # self.expandmask = ExpandMask(stride=dilate_stride, padding=1, mask_channel_group=mask_channel_group)

    def forward(self, x, temperature):
        mask =  F.adaptive_avg_pool2d(x, self.mask_size) if self.mask_size < x.shape[2] else x
        flops = mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        mask = self.conv(mask)
        flops += self.conv_flops_pp * mask.shape[2] * mask.shape[3]
        
        b,c,h,w = mask.shape
        mask = mask.view(b,2,c//2,h,w)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        sparsity = mask.mean()
        # print('spatial mask:')
        # print(mask)
        # print('spatial mask sparsity:', sparsity)
        return mask, sparsity, flops

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
            
            # print(f'self.pad_kernel: {self.pad_kernel}')

        self.dilate_kernel = torch.ones((self.mask_channel_group,self.mask_channel_group,1+2*self.padding,1+2*self.padding), device=x.device)
        # print(f'self.dilate_kernel: {self.dilate_kernel}')
        
        x = x.float()
        
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5


class Masker_channel_MLP(nn.Module):
    def __init__(self, in_channels, channel_dyn_group, layers=2, reduction=16):
        super(Masker_channel_MLP, self).__init__()
        assert(layers in [1,2])
        
        self.channel_dyn_group = channel_dyn_group
        width = max(channel_dyn_group//reduction, 16)
        self.conv = nn.Sequential(
            nn.Linear(in_channels, width),
            nn.ReLU(),
            nn.Linear(width, channel_dyn_group*2,bias=True)
        ) if layers == 2 else nn.Linear(in_channels, channel_dyn_group*2,bias=True)
        
        self.conv_flops = in_channels * width + width * channel_dyn_group*2 if layers == 2 else in_channels * channel_dyn_group*2
        if layers == 2:
            self.conv[-1].bias.data[:channel_dyn_group] = 2.0
            self.conv[-1].bias.data[channel_dyn_group+1:] = -2.0
        else:
            self.conv.bias.data[:channel_dyn_group] = 2.0
            self.conv.bias.data[channel_dyn_group+1:] = -2.0

    def forward(self, x, temperature):
        b, c, h, w = x.shape
        flops = c * h * w
        mask =  F.adaptive_avg_pool2d(x, (1,1)).view(b,c)
        
        mask = self.conv(mask)
        flops += self.conv_flops
        
        b,c = mask.shape
        mask = mask.view(b,2,c//2)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        
        sparsity = torch.mean(mask)
        
        return mask, sparsity, flops

class Masker_channel_conv_linear(nn.Module):
    def __init__(self, in_channels, channel_dyn_group, reduction=16):
        super(Masker_channel_conv_linear, self).__init__()
        self.channel_dyn_group = channel_dyn_group
        
        self.conv = nn.Sequential(
            conv1x1(in_channels, in_channels//reduction),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(),
        )
        self.linear = nn.Linear(in_channels//reduction, channel_dyn_group*2,bias=True)
        
        self.linear.bias.data[:channel_dyn_group] = 2.0
        self.linear.bias.data[channel_dyn_group+1:] = -2.0
        
        self.masker_flops = in_channels * in_channels // reduction + in_channels // reduction * channel_dyn_group*2

    def forward(self, x, temperature):
        mask = self.conv(x)
        b, c, h, w = mask.shape
        flops = c * h * w
        mask =  F.adaptive_avg_pool2d(mask, (1,1)).view(b,c)
        
        mask = self.linear(mask)
        flops += self.masker_flops
        
        b,c = mask.shape
        mask = mask.view(b,2,c//2)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        
        sparsity = torch.mean(mask)
        
        return mask, sparsity, flops

if __name__ == '__main__':
    with torch.no_grad():
        # mask = torch.zeros(1,2,2,2)
        # mask[0,0,0,0] = 1.0
        # mask[0,0,1,1] = 1.0
        
        # mask[0,1,0,1] = 1.0
        # # mask[0,1,1,0] = 1.0
        
        # expandmask = ExpandMask(stride=2, padding=1, mask_channel_group=2)
        # mask_dil = expandmask(mask)
        
        # print(mask_dil.float())
        
        output_size = 4
        spatial_mask_channel_group = 1
        channel_dyn_group = 2
        
        x = torch.rand(1, 16, output_size, output_size)
        
        mask_spatial_granularity = 2
        mask_size = output_size // mask_spatial_granularity
        
        masker_spatial = Masker_spatial(16, spatial_mask_channel_group, mask_size)
        mask_expander2 = ExpandMask(stride=1, padding=0, mask_channel_group=spatial_mask_channel_group)
        
        # masker_channel = Masker_channel(16, channel_dyn_group)
        # mask_expander1 = ExpandMask(stride=1, padding=1, mask_channel_group=spatial_mask_channel_group)
        
        
        
        
        # channel_mask, channel_sparsity, channel_mask_flops = masker_channel(x, temperature=0.1)
        spatial_mask_conv3, spatial_sparsity, spatial_mask_flops = masker_spatial(x, temperature=0.1)
        
        # spatial_mask_conv3 = F.upsample_nearest(spatial_mask_conv3, size=output_size)
        # spatial_mask_conv2 = mask_expander2(spatial_mask_conv3)
        # spatial_mask_conv1 = mask_expander1(spatial_mask_conv2)
        
        # print(f'channel mask : {channel_mask}')
        # print(f'spatial_mask_conv3 :')
        # print(spatial_mask_conv3.float())
        # print(f'spatial_mask_conv2 :')
        # print(spatial_mask_conv2.float())
        # print(f'spatial_mask_conv1 :')
        # print(spatial_mask_conv1.float())
        
        
        # mask_channel = torch.zeros(1,2)
        # mask_channel[0,1] = 1.
        # x = torch.rand(1,6,3,3)
        # print(apply_channel_mask(x, mask_channel))
        
        
        # mask_spatial = torch.zeros(1,2,3,3)
        # mask_spatial[0,0,0,0] = 1.
        # mask_spatial[0,0,1,1] = 1.
        # mask_spatial[0,0,2,2] = 1.
        
        # mask_spatial[0,1,0,2] = 1.
        # mask_spatial[0,1,1,1] = 1.
        # mask_spatial[0,1,2,0] = 1.
        
        # x = torch.rand(1,6,3,3)
        # print(apply_spatial_mask(x, mask_spatial))
        