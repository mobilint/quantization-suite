#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:40:47 2020

@author: js
"""

import torch
import model
import model_fused


def make_fused_weights(weight_path):
    save_path = weight_path[:-4] + '_fused.pth'
        
    checkpoint = torch.load(weight_path)
    if 'model_state_dict' in checkpoint.keys():
        weights = checkpoint['model_state_dict']
    else:
        weights = checkpoint
    
    net = model.Resnet50()
    net_fused = model_fused.Resnet50()
    
    new_weights = net_fused.state_dict()
    
    weight_dict = {}
    running_mean_list = []
    bias_list = []
    
    # weight dict
    weight_num = 0
    for key in weights.keys():
        if key[-6:] == 'weight':
            weight_dict[key[:-7]] = weight_num
            weight_num += 1
        elif key[-12:] == 'running_mean':
            running_mean_list.append(key[:-13])
            
    # new weight dict
    for key in weights.keys():
        if key[-4:] == 'bias':
            if key[:-5] not in running_mean_list:
                bias_list.append(key[:-5])

    bn_num = 0    
    for bn_name in running_mean_list:
        weight_num = weight_dict[bn_name] - 1
        conv_name = list(weight_dict.keys())[weight_num]
#        new_conv_name = list(new_weight_dict.keys())[weight_num - bn_num]
        
        conv_weight, conv_bias = batch_concate(weights, conv_name, bn_name, is_tconv=False)
        
        new_weights[conv_name + '.weight'] = conv_weight
        new_weights[conv_name + '.bias'] = conv_bias
        
        print(conv_name, bn_name)
        
        bn_num += 1
    
    for bias_name in bias_list:
        print(bias_name, bias_name)
        new_weights[bias_name + '.weight'] = weights[bias_name + '.weight'].double()
        new_weights[bias_name + '.bias'] = weights[bias_name + '.bias'].double()
    
    # compare_result
    net.load_state_dict(weights)
    net_fused.load_state_dict(new_weights)
    
    net.eval()
    net_fused.eval()
    
    # testing
    bn_num = 0
    for bn_name in running_mean_list:
        weight_num = weight_dict[bn_name] - 1
        conv_name = list(weight_dict.keys())[weight_num]
        
        conv_path = conv_name.split('.')
        
        
        bn_num += 1
    
    with torch.no_grad():
        x = torch.rand((1, 3, 224, 224))
        output = net(x)
        output_fused = net_fused(x)
    
    print(output.shape)
    print(output_fused.shape)
#    print(output - output_fused)
    print(abs(output - output_fused).max())

    torch.save({'model_state_dict': new_weights}, save_path)

        
def batch_concate(parameter_dict, conv_name, bn_name, is_tconv=False):
    eps = 0.00001
    conv_weight = parameter_dict[conv_name+'.weight'].double()
    if is_tconv:
        conv_weight = tconv_weight_conversion(conv_weight)
    bn_weight = parameter_dict[bn_name+'.weight'].double()
    bn_bias = parameter_dict[bn_name+'.bias'].double()
    bn_running_mean = parameter_dict[bn_name+'.running_mean'].double()
    bn_running_var = parameter_dict[bn_name+'.running_var'].double()
    bn_running_sigma = torch.sqrt(bn_running_var + eps)
    conv_weight = conv_weight.permute(1, 2, 3, 0)
    conv_weight = conv_weight * (bn_weight / bn_running_sigma)
    conv_weight = conv_weight.permute(3, 0, 1, 2)
    conv_bias = -bn_weight * bn_running_mean / bn_running_sigma + bn_bias
    
    return conv_weight.cpu().double(), conv_bias.cpu().double()

def tconv_weight_conversion(weight):
    shape = weight.shape
    weight_copy = weight.clone()
    for i in range(shape[2]):
        for j in range(shape[3]):
            weight_copy[:, :, i, j] = weight[:, :, shape[2]-1-i, shape[3]-1-j]
    return weight_copy.permute(1, 0, 2, 3)

if __name__ == "__main__":
    make_fused_weights("experiments/torch_fp32/pretrained.pth")