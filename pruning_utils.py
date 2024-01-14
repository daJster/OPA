
import torch
import torch.nn as nn
from copy import deepcopy
from utils import get_layer_by_name, replace_layer_by_name
import numpy as np


def get_mask(weights, strategy, pruning_rate):

    if strategy == "global":

        mask = weights > pruning_rate

    elif strategy == "local":

        # find threshold that is `prune_rate`(from 0 to 1) quantile of weights.
        # use torch.quantile
        
        threshold = torch.quantile(weights, q=pruning_rate, dim=None)

        # # Your code here
        # threshold = ...

        # print(f"Local threshold: {threshold.item():.3e}")
        mask = weights > threshold
        
    else:
        mask = torch.tensor(1.)
        print(f"Unknown strategy: {strategy}")
    
    return mask

class MaskedConv2d(nn.Module):
    def __init__(self, conv_orig):
        super(MaskedConv2d, self).__init__()
        
        self.conv = deepcopy(conv_orig)
        self.register_buffer("mask", torch.tensor(1.))
        
    def compute_mask(self, strategy="local", pruning_rate=0.1):
        
        weights = self.conv.weight.data.abs()
        self.mask = get_mask(weights, strategy, pruning_rate)
            
    def forward(self, x):

        self.conv.weight.data *= self.mask
        out = self.conv(x)
        
        return out

    def prune_grad(self):

        self.conv.weight.grad *= self.mask

def replace_layers_unstructed(model, conv_list):
    
    for conv_name in conv_list:
        
        # get layer from model, turn it into masked conv and plug into model
        conv = get_layer_by_name(model, conv_name)
        conv_masked = MaskedConv2d(conv)
        replace_layer_by_name(model, conv_name, conv_masked)
        
        # Your code here

def compute_conv_masks(model, conv_list, strategy, prune_rate):

    for conv_name in conv_list:
        
        conv = get_layer_by_name(model, conv_name)
        # print(f'[{conv_name}] ', end="")
        conv.compute_mask(strategy, prune_rate)

def prune_all_gradients(model, conv_list):

    for conv_name in conv_list:
        conv = get_layer_by_name(model, conv_name)
        conv.prune_grad()
        
def compute_nonzero_ratio(model, conv_list):
    nonzero_weights = 0
    total_weights = 0
    
    for conv_name in conv_list:
        
        # get layer from model, find number of nonzero weights 
        # and total number of weights, save them
        
        conv = get_layer_by_name(model, conv_name)
        nonzero_weights += conv.mask.sum().cpu()
        total_weights += np.prod(conv.mask.size())
        
        # Your code here
        
    return nonzero_weights / total_weights, (nonzero_weights, total_weights)

def prune_unstructed(model, conv_list, strategy="local", prune_rate=0.1):

    # make copy of model
    model_pruned = deepcopy(model)

    # replace original conv2d with masked conv2d with mask == 1
    replace_layers_unstructed(model_pruned, conv_list)

    # estimate sparsification mask (0 and 1)
    compute_conv_masks(model_pruned, conv_list, strategy=strategy, prune_rate=prune_rate)
    
    return model_pruned

from utils import get_layer_by_name, replace_layer_by_name

def prune_output_channels_in_conv(conv_orig, channel_mask):
    
    n_channels = channel_mask.sum()
    
    has_bias = conv_orig.bias is not None
    
    conv_pruned = nn.Conv2d(in_channels=conv_orig.in_channels,
                            out_channels=n_channels, 
                            kernel_size=conv_orig.kernel_size,
                            stride=conv_orig.stride,
                            padding=conv_orig.padding,
                            bias=has_bias)

    conv_pruned.weight.data = conv_orig.weight.data[channel_mask].clone()
    if has_bias:
        conv_pruned.bias.data = conv_orig.bias.data[channel_mask].clone()
        
    # Create nn.Conv2d with smaller number of output channels and plug weight and bias (if any) 
    # from original convolution into it
    
    # Your code here
    
    
    return conv_pruned
    
def prune_batchnorm(bn_orig, channel_mask):
    
    n_channels = channel_mask.sum()
    
    bn_pruned = nn.BatchNorm2d(n_channels)
    
    bn_pruned.weight.data  = bn_orig.weight.data[channel_mask].clone()
    bn_pruned.bias.data    = bn_orig.bias.data[channel_mask].clone()
    bn_pruned.running_mean = bn_orig.running_mean[channel_mask].clone()
    bn_pruned.running_var  = bn_orig.running_var[channel_mask].clone()
    
    # Create nn.BatchNorm2d with smaller number of features and plug weight, bias, running_mean and running_var
    # from original batchnorm into it
    
    # Your code here

    return bn_pruned

def prune_input_channels_in_conv(conv_orig, channel_mask):
    
    n_channels = channel_mask.sum()
    
    has_bias = conv_orig.bias is not None
    
    conv_pruned = nn.Conv2d(in_channels=n_channels,
                            out_channels=conv_orig.out_channels, 
                            kernel_size=conv_orig.kernel_size,
                            stride=conv_orig.stride,
                            padding=conv_orig.padding,
                            bias=has_bias)

    conv_pruned.weight.data = conv_orig.weight.data[:, channel_mask].clone()
    if has_bias:
        conv_pruned.bias.data = conv_orig.bias.data.clone()
        
    # Create nn.Conv2d with smaller number of input channels and plug weight and bias (if any) 
    # from original convolution into it
    
    # Your code here
        
    
    return conv_pruned

    
def update_layer(basicblock, channel_mask):
    
    # update output channels in 1st convolution
    basicblock.conv1 = prune_output_channels_in_conv(basicblock.conv1, channel_mask)

    # update BatchNorm
    basicblock.bn1 = prune_batchnorm(basicblock.bn1, channel_mask)

    # update input channels in 2nd convolution
    basicblock.conv2 = prune_input_channels_in_conv(basicblock.conv2, channel_mask)
    

def prune_BasicBlock(basicblock_orig, pruning_rate, channel_selection_strategy="slimming", strategy="local"):
    
    basicblock = deepcopy(basicblock_orig)
    
    if channel_selection_strategy == "slimming":
        
        scales = basicblock.bn1.weight.data.abs()
        channel_mask = get_mask(scales, strategy, pruning_rate)
        update_layer(basicblock, channel_mask)
        
    elif channel_selection_strategy == "l1":
        
        # estimate L1 norm of conv1 output channels. 
        # Conv weight is of shape (out_channels, input_channels, kernel_size_1, kernel_size_2)
        norms = basicblock.conv1.weight.data.abs().sum(dim=(1, 2, 3))
        
        # Your code here
        selected_channels = get_mask(norms, strategy, pruning_rate)
        update_layer(basicblock, selected_channels)
        
    else:
        print("Unknown channel selection strategy")
        
    return basicblock

def replace_layers_structed(model, basicblock_list, pruning_rate, channel_selection_strategy="slimming", strategy="local"):
    
    for basicblock_name in basicblock_list:
        basicblock = get_layer_by_name(model, basicblock_name)
        basicblock_pruned = prune_BasicBlock(basicblock, pruning_rate, channel_selection_strategy, strategy)
        replace_layer_by_name(model, basicblock_name, basicblock_pruned)

def prune_structed(model, basic_block_list, prune_rate=0.1, channel_selection_strategy="slimming"):
    
    # make copy of model
    model_pruned = deepcopy(model)

    # replace original blocks with pruned blocks
    replace_layers_structed(model_pruned, basic_block_list, prune_rate, channel_selection_strategy)
    
    return model_pruned