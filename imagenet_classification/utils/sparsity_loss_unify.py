import math
import torch.nn as nn
import torch


class SparsityCriterion_bounds(nn.Module):
    def __init__(self, sparsity_target, num_epochs, full_flops):
        super(SparsityCriterion_bounds, self).__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops

    def forward(self, epoch, sparsity_list, flops):
        
        loss_block_bounds = 0.0
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound = (1 - progress*(1-self.sparsity_target))
        lower_bound = progress*(self.sparsity_target)
            
        for i in range(len(sparsity_list)):
            loss_block_bounds += max(0, sparsity_list[i] - upper_bound)**2
            
            loss_block_bounds += max(0, lower_bound - sparsity_list[i])**2
        
        loss_block_bounds /= len(sparsity_list)
        loss_sparsity = (flops/self.full_flops - self.sparsity_target)**2

        return  loss_block_bounds + loss_sparsity

class SparsityCriterion(nn.Module):
    def __init__(self, flops_perc_target, num_epochs, full_flops):
        super(SparsityCriterion, self).__init__()
        self.flops_perc_target = flops_perc_target
        self.channel_target = math.sqrt(flops_perc_target)
        self.num_epochs = num_epochs
        self.full_flops = full_flops

    def forward(self, epoch, channel_sparsity_list, flops_perc_list, flops):
        
        loss_flops_perc_per_block_bounds = 0.0
        # loss_channel_perc_per_block_bounds = 0.0
        
        
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound_flops = (1 - progress*(1-self.flops_perc_target))
        lower_bound_flops = progress*(self.flops_perc_target)
        
        # upper_bound_channel = (1 - progress*(1-self.channel_target))
        # lower_bound_channel = progress*(self.channel_target)
        
        num_blocks = len(flops_perc_list)
        
        for i in range(num_blocks):
            loss_flops_perc_per_block_bounds += max(0, flops_perc_list[i] - upper_bound_flops)**2
            loss_flops_perc_per_block_bounds += max(0, lower_bound_flops - flops_perc_list[i])**2
            
            # loss_channel_perc_per_block_bounds += max(0, channel_sparsity_list[i] - upper_bound_channel)**2
            # loss_channel_perc_per_block_bounds += max(0, lower_bound_channel - channel_sparsity_list[i])**2
        
        loss_channel_perc_per_block = torch.mean((channel_sparsity_list - self.channel_target)**2)
        
        loss_flops_perc_per_block_bounds /= num_blocks
        # loss_channel_perc_per_block_bounds /= num_blocks
        
        loss_flops_perc_overall = (flops/self.full_flops - self.flops_perc_target)**2
        
        return  loss_channel_perc_per_block + loss_flops_perc_per_block_bounds + loss_flops_perc_overall

class SparsityCriterion_channel_factor(nn.Module):
    def __init__(self, flops_perc_target=1.0, num_epochs=100, full_flops=4.1, channel_loss_factor=1.0, channel_target=None, dyn_mode=['both', 'both', 'both', 'both']):
        super(SparsityCriterion_channel_factor, self).__init__()
        self.flops_perc_target = flops_perc_target
        
        self.channel_target = math.sqrt(flops_perc_target) if channel_target is None else channel_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops
        self.channel_loss_factor = channel_loss_factor
        self.dyn_mode = dyn_mode

    def forward(self, epoch, channel_sparsity_list, flops_perc_list, flops):
        
        loss_flops_perc_per_block_bounds = 0.0        
        
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound_flops = (1 - progress*(1-self.flops_perc_target))
        lower_bound_flops = progress*(self.flops_perc_target)
        
        
        num_blocks = len(flops_perc_list)
        
        for i in range(num_blocks):
            loss_flops_perc_per_block_bounds += max(0, flops_perc_list[i] - upper_bound_flops)**2
            loss_flops_perc_per_block_bounds += max(0, lower_bound_flops - flops_perc_list[i])**2
        
        loss_channel_perc_per_block = 0.0
        for i in range(4):
            if self.dyn_mode[i] == 'both':
                loss_channel_perc_per_block += torch.mean((channel_sparsity_list[i] - self.channel_target)**2)
        
        loss_flops_perc_per_block_bounds /= num_blocks
        
        loss_flops_perc_overall = (flops/self.full_flops - self.flops_perc_target)**2
        
        return  self.channel_loss_factor*loss_channel_perc_per_block + loss_flops_perc_per_block_bounds + loss_flops_perc_overall

class SparsityCriterion_cs(nn.Module):
    def __init__(self, flops_perc_target=1.0, num_epochs=100, full_flops=4.1, cs_loss_factor=1.0, channel_target=None, dyn_mode=['both', 'both', 'both', 'both']):
        super(SparsityCriterion_cs, self).__init__()
        self.flops_perc_target = flops_perc_target
        
        self.channel_target = math.sqrt(flops_perc_target) if channel_target is None else channel_target
        self.spatial_target = flops_perc_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops
        self.cs_loss_factor = cs_loss_factor
        self.dyn_mode = dyn_mode

    def forward(self, epoch, channel_sparsity_list, spatial_sparsity_list, flops_perc_list, flops):
        
        loss_flops_perc_per_block_bounds = 0.0
        # loss_channel_perc_per_block_bounds = 0.0
        
        # print(channel_sparsity_list)
        
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound_flops = (1 - progress*(1-self.flops_perc_target))
        lower_bound_flops = progress*(self.flops_perc_target)
        
        num_blocks = len(flops_perc_list)
        
        for i in range(num_blocks):
            loss_flops_perc_per_block_bounds += max(0, flops_perc_list[i] - upper_bound_flops)**2
            loss_flops_perc_per_block_bounds += max(0, lower_bound_flops - flops_perc_list[i])**2
        
        loss_cs_density = 0.0
        for i in range(4):
            if self.dyn_mode[i] == 'both':
                loss_cs_density += torch.mean((channel_sparsity_list[i] - self.channel_target)**2)
                loss_cs_density += torch.mean((spatial_sparsity_list[i] - self.spatial_target)**2)
        
        loss_flops_perc_per_block_bounds /= num_blocks
        
        loss_flops_perc_overall = (flops/self.full_flops - self.flops_perc_target)**2
        
        return  self.cs_loss_factor*loss_cs_density + loss_flops_perc_per_block_bounds + loss_flops_perc_overall

class SparsityCriterion_cs_v2(nn.Module):
    def __init__(self, flops_perc_target=1.0, num_epochs=100, full_flops=4.1, cs_loss_factor=1.0, channel_target=None, dyn_mode=['both', 'both', 'both', 'both']):
        super(SparsityCriterion_cs_v2, self).__init__()
        self.flops_perc_target = flops_perc_target
        
        self.channel_target = math.sqrt(flops_perc_target) if channel_target is None else channel_target
        self.spatial_target = flops_perc_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops
        self.cs_loss_factor = cs_loss_factor
        self.dyn_mode = dyn_mode

    def forward(self, epoch, channel_sparsity_list, spatial_sparsity_list, flops_perc_list, flops):
        
        loss_flops_perc_per_block_bounds = 0.0
        # loss_channel_perc_per_block_bounds = 0.0
        
        # print(channel_sparsity_list)
        
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound_flops = (1 - progress*(1-self.flops_perc_target))
        lower_bound_flops = progress*(self.flops_perc_target)
        
        num_blocks = len(flops_perc_list)
        
        for i in range(num_blocks):
            loss_flops_perc_per_block_bounds += max(0, flops_perc_list[i] - upper_bound_flops)**2
            loss_flops_perc_per_block_bounds += max(0, lower_bound_flops - flops_perc_list[i])**2
        
        all_density_c,  all_density_s = [], []
        for i in range(4):
            if self.dyn_mode[i] in ['channel', 'both']:
                all_density_c.append(channel_sparsity_list[i])
            if self.dyn_mode[i] in ['spatial', 'both']:
                all_density_s.append(spatial_sparsity_list[i])
        
        all_density_c, all_density_s = torch.cat(all_density_c), torch.cat(all_density_s)
        loss_cs_density = (torch.mean(all_density_c) - self.channel_target)**2 + (torch.mean(all_density_s) - self.spatial_target)**2
        
        loss_flops_perc_per_block_bounds /= num_blocks
        
        loss_flops_perc_overall = (flops/self.full_flops - self.flops_perc_target)**2
        
        return  self.cs_loss_factor*loss_cs_density + loss_flops_perc_per_block_bounds + loss_flops_perc_overall

class SparsityCriterion_channel_bounds(nn.Module):
    def __init__(self, flops_perc_target=1.0, num_epochs=100, full_flops=4.1, channel_loss_factor=1.0):
        super(SparsityCriterion_channel_bounds, self).__init__()
        self.flops_perc_target = flops_perc_target
        self.channel_target = math.sqrt(flops_perc_target)
        self.num_epochs = num_epochs
        self.full_flops = full_flops
        self.channel_loss_factor = channel_loss_factor

    def forward(self, epoch, channel_sparsity_list, flops_perc_list, flops):
        
        loss_flops_perc_per_block_bounds = 0.0
        loss_channel_perc_per_block_bounds = 0.0
        
        
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound_flops = (1 - progress*(1-self.flops_perc_target))
        lower_bound_flops = progress*(self.flops_perc_target)
        
        upper_bound_channel = (1 - progress*(1-self.channel_target))
        lower_bound_channel = progress*(self.channel_target)
        
        num_blocks = len(flops_perc_list)
        
        for i in range(num_blocks):
            loss_flops_perc_per_block_bounds += max(0, flops_perc_list[i] - upper_bound_flops)**2
            loss_flops_perc_per_block_bounds += max(0, lower_bound_flops - flops_perc_list[i])**2
            
            loss_channel_perc_per_block_bounds += max(0, channel_sparsity_list[i] - upper_bound_channel)**2
            loss_channel_perc_per_block_bounds += max(0, lower_bound_channel - channel_sparsity_list[i])**2
        
        # loss_channel_perc_per_block = torch.mean((channel_sparsity_list - self.channel_target)**2)
        
        loss_flops_perc_per_block_bounds /= num_blocks
        loss_channel_perc_per_block_bounds /= num_blocks
        
        loss_flops_perc_overall = (flops/self.full_flops - self.flops_perc_target)**2
        
        return  self.channel_loss_factor*loss_channel_perc_per_block_bounds + loss_flops_perc_per_block_bounds + loss_flops_perc_overall


class SparsityCriterion_channel_bounds_v2(nn.Module):
    def __init__(self, flops_perc_target=1.0, num_epochs=100, full_flops=4.1, channel_loss_factor=1.0):
        super(SparsityCriterion_channel_bounds_v2, self).__init__()
        self.flops_perc_target = flops_perc_target
        self.channel_target = math.sqrt(flops_perc_target)
        self.num_epochs = num_epochs
        self.full_flops = full_flops
        self.channel_loss_factor = channel_loss_factor

    def forward(self, epoch, channel_sparsity_list, flops_perc_list, flops):
        
        loss_flops_perc_per_block_bounds = 0.0
        loss_channel_perc_per_block_bounds = 0.0
        
        
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound_flops = (1 - progress*(1-self.flops_perc_target))
        lower_bound_flops = progress*(self.flops_perc_target)
        
        upper_bound_channel = (0.85 - progress*(0.85-self.channel_target))
        lower_bound_channel = progress*(self.channel_target)
        
        # print(epoch, upper_bound_channel, lower_bound_channel)
        
        num_blocks = len(flops_perc_list)
        
        for i in range(num_blocks):
            loss_flops_perc_per_block_bounds += max(0, flops_perc_list[i] - upper_bound_flops)**2
            loss_flops_perc_per_block_bounds += max(0, lower_bound_flops - flops_perc_list[i])**2
            
            loss_channel_perc_per_block_bounds += max(0, channel_sparsity_list[i] - upper_bound_channel)**2
            loss_channel_perc_per_block_bounds += max(0, lower_bound_channel - channel_sparsity_list[i])**2
        
        # loss_channel_perc_per_block = torch.mean((channel_sparsity_list - self.channel_target)**2)
        
        loss_flops_perc_per_block_bounds /= num_blocks
        loss_channel_perc_per_block_bounds /= num_blocks
        
        loss_flops_perc_overall = (flops/self.full_flops - self.flops_perc_target)**2
        
        return  self.channel_loss_factor*loss_channel_perc_per_block_bounds + loss_flops_perc_per_block_bounds + loss_flops_perc_overall

if __name__ == '__main__':
    a = SparsityCriterion_channel_bounds_v2(flops_perc_target=0.5)
    for epoch in range(100):
        y = a(epoch, [1.0, 1.0], [1.0, 1.0], 4.1)