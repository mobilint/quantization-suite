#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:16:05 2020

@author: js
"""

import torch
import torch.nn as nn
import numpy as np
import math
import os
import copy
from collections import OrderedDict


class DownResolFunction(torch.autograd.Function):
    def __init__(self):
        super(DownResolFunction, self).__init__()
    
    @staticmethod
    def forward(self, input, num_bit, scale):
        result = input.clone()
        result = result * (2.0 ** scale)
        result = result.clamp(min=-(2.0 ** (num_bit-1.0))+1.0, max=(2.0 ** (num_bit-1.0)))
        result = torch.round(result.float()) / (2.0 ** scale)
        
        return result
    
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input, None, None


class DownResolFunction_floor(torch.autograd.Function):
    def __init__(self):
        super(DownResolFunction, self).__init__()
    
    @staticmethod
    def forward(self, input, num_bit, scale):
        result = input.clone()
        result = result * (2.0 ** scale)
        result = result.clamp(min=-(2.0 ** (num_bit-1.0))+1.0, max=(2.0 ** (num_bit-1.0)))
        result = torch.floor(result.float() + 0.5) / (2.0 ** scale)
        
        return result
    
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input, None, None
    

class DownResolModule(nn.Module):
    def __init__(self, num_bit, scale):
        super(DownResolModule, self).__init__()
        self.num_bit = torch.from_numpy(np.array(num_bit))
        self.scale = torch.from_numpy(np.array(scale))
        self.down_resol = DownResolFunction.apply
        
    def forward(self, input):
        outputs = self.down_resol(input, self.num_bit, self.scale)
        
        return outputs


class DynamicDownResolModule(nn.Module):
    def __init__(self, num_bits=8):
        super(DynamicDownResolModule, self).__init__()
        self.num_bits=num_bits
        self.down_resol = DownResolFunction.apply
        
    def forward(self, input):
        scale = calcScale(input.min(), input.max(), num_bits=self.num_bits)
        outputs = self.down_resol(input, self.num_bits, scale)
        
        return outputs


def calcScale(min_val, max_val, num_bits=8):
    # Calc Scale of next 
    max_abs = max(abs(min_val), abs(max_val))
    
    scale = int(num_bits - (math.log2(max_abs) + 1))
#    scale = math.ceil(num_bits - (math.log2(max_abs) + 1))

    return scale


def weights_quant(weights_dict, num_bits=8):
    # minimize L2 between weights and new_weights
    Down_resol = DownResolFunction.apply
    new_weights_dict = copy.deepcopy(weights_dict)
    weight_scale_list = []
    for key in weights_dict.keys():
        if key[-6:] == 'weight':
            weight = weights_dict[key]
            scale = calcScale(weight.min(), weight.max(), num_bits=num_bits)
            new_weight = Down_resol(weight, num_bits, scale).type(weight.type())
            L2_min = torch.dist(weight, new_weight, 2)
            
            for i in range(scale-3, scale+4):
                weight_temp = Down_resol(weight, num_bits, i).type(weight.type())
                L2 = torch.dist(weight, weight_temp, 2)
                
                if L2 < L2_min:
                    L2_min = L2
                    new_weight = weight_temp
                    scale = i
                    
            new_weights_dict[key] = new_weight
            weight_scale_list.append(scale)
            
    return new_weights_dict, weight_scale_list


def weights_quant_b(net, val_dataset, device, weights_dict, num_bits=8):
    # minimize L2 between sample outputs
    net.eval()
    
    Down_resol = DownResolFunction.apply
    new_weights_dict = copy.deepcopy(weights_dict)
        
    num_data = len(val_dataset)
    np.random.seed(777)
    ids = np.arange(num_data)
    np.random.shuffle(ids)
    num_sample = 10
    
    sample_output_list = []
    with torch.no_grad():
        for i in range(num_sample):
            input, label = val_dataset[ids[i]]
            input = input.unsqueeze(0).to(device)
            sample_output = net(input)
            sample_output_list.append(sample_output)
    
    weight_scale_list = []
    with torch.no_grad():
        for key in weights_dict.keys():
            if key[-6:] == 'weight':
                L2_min = 100000000000000
                weight = weights_dict[key]
                scale = calcScale(weight.min(), weight.max(), num_bits=num_bits)
                new_weight = Down_resol(weight, num_bits, scale)
                
                for i in range(scale-5, scale+5):
                    L2 = 0
                    weight_temp = Down_resol(weight, num_bits, i)
                    new_weights_dict[key] = weight_temp
                    net.load_state_dict(new_weights_dict)
                    net.eval()
                    
                    for j in range(num_sample):
                        input, label = val_dataset[ids[j]]
                        input = input.unsqueeze(0).to(device)
                        output = net(input)
                        L2 += torch.dist(output, sample_output_list[j], 2)
                        
                    if L2 < L2_min:
                        L2_min = L2
                        new_weight = weight_temp
                        scale = i
                weight_scale_list.append(scale)
                new_weights_dict[key] = new_weight

    return new_weights_dict, weight_scale_list
                    

def weights_quant_with_scale(weights_dict, weight_scale_list, num_bits=8):
    # quantize with certain scale list
    Down_resol = DownResolFunction.apply
    new_weights_dict = copy.deepcopy(weights_dict)
    i = 0
    for key in weights_dict.keys():
        if key[-6:] == 'weight':
            weight = weights_dict[key]
            scale = weight_scale_list[i]
            new_weight = Down_resol(weight, num_bits, scale)
            new_weights_dict[key] = new_weight
            i += 1
            
    return new_weights_dict, weight_scale_list