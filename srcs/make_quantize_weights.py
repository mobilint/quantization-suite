#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:25:59 2020

@author: js
"""


import os
import copy
import torch
import torchvision
from utils import AverageMeter, accuracy
from utils_down_resol import calcScale, DownResolFunction

import model_tf_fused


def dataset_test(net, val_dataset, weights):
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=3, pin_memory=True)
    
    with torch.no_grad():
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        net.load_state_dict(weights)
        net = net.cuda()
        net.eval()
        for input, label in val_loader:
            input = input.cuda()
            
            input[:, 0, :, :] = (input[:, 0, :, :] * 0.5 + 0.5) * 255 - 123.68
            input[:, 1, :, :] = (input[:, 1, :, :] * 0.5 + 0.5) * 255 - 116.78
            input[:, 2, :, :] = (input[:, 2, :, :] * 0.5 + 0.5) * 255 - 103.94
            
            output = net(input).cpu()
            label = label + 1
            
            prec1, prec5 = accuracy(output.data, label, topk=(1,5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
        print("Prec @1: ", top1.avg.item())
        
    return top1.avg.item(), top5.avg.item()


def make_quant_weights(weight_path):
    # load dataset
    val_dir = "/ssd/datasets/public/imagenet/val/"
    normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
            ]))
    
    save_path = weight_path[:-4] + '_quant.pth'
    
    num_bits = 8
    Down_resol = DownResolFunction.apply
        
    # load weights
    checkpoint = torch.load(weight_path)
    weights = checkpoint['model_state_dict']
    new_weights = copy.deepcopy(weights)
    print("weight loaded from ", weight_path)

    # initialize
    print("==================making initial scale=================")
    for key in weights.keys():
        if key[-6:] == 'weight':
            weight = weights[key]
            scale = calcScale(weight.min(), weight.max(), num_bits=num_bits)
            new_weight = Down_resol(weight, num_bits, scale)
            new_weights[key] = new_weight
            print(key, "|| scale : ", scale, "|| difference : ", (weight.float() - new_weight).abs().max())
    
    net = model_tf_fused.Resnet50()
    net.load_state_dict(weights)
    net.eval()
    with torch.no_grad():
        x, y = val_dataset[1]
        x = x.unsqueeze(0)
        output = net(x)
    
    quant_net = model_tf_fused.Resnet50()
    
    quant_net.load_state_dict(new_weights)
    quant_net.eval()
    
    with torch.no_grad():
        quant_output = quant_net(x)
    
    output_gap = abs(output - quant_output).max()
    new_weights_temp = copy.deepcopy(new_weights)
    
    print("output_gap is ", output_gap)
    
    # testing
    print("==================testing=================")
    
    top1_score = 0
    scale_list = []
    for key in weights.keys():
        if key[-6:] == 'weight':
            L2 = 1000000000
            
            weight = weights[key]
            scale = calcScale(weight.min(), weight.max(), num_bits=num_bits)
            
            
            new_weight_temp = Down_resol(weight, num_bits, scale)
            scale_temp = scale
            
            for i in range(scale-1, scale+3):
                new_weight = Down_resol(weight, num_bits, i)
                new_weights_temp[key] = new_weight
                
                # minimize maximum gap
#                quant_net.load_state_dict(new_weights_temp)
#                quant_net.eval()
#                
#                with torch.no_grad():
#                    output_temp = quant_net(x)
#                output_gap_temp = abs(output - output_temp).max()
#                if output_gap_temp < output_gap:
#                    output_gap = output_gap_temp
#                    scale_temp = i
#                    new_weight_temp = new_weight
                
                # maximize accuracy
                top1, top5 = dataset_test(net, val_dataset, new_weights_temp)
                if top1 > top1_score:
                    top1_score = top1
                    scale_temp = i
                    new_weight_temp = new_weight

                # minimize L2 norm
#                L2_temp = torch.dist(weight, new_weight, 2)
#                if L2_temp < L2:
#                    L2 = L2_temp
#                    scale_temp = i
#                    new_weight_temp = new_weight
            
            print(key, scale_temp, top1_score)
#            print(key, scale_temp, L2)
            new_weights_temp[key] = new_weight_temp
            scale_list.append(scale_temp)
    
    print(scale_list)
    
    net = model_tf_fused.Resnet50()
    quant_net = model_tf_fused.Resnet50()
    quant_net_temp = model_tf_fused.Resnet50()
    
    net.load_state_dict(weights)
    quant_net.load_state_dict(new_weights)
    quant_net_temp.load_state_dict(new_weights_temp)
    
    net.eval()
    quant_net.eval()
    quant_net_temp.eval()
    
    with torch.no_grad():
#        x = torch.rand(1, 3, 224, 224)
        output = net(x)
        quant_output = quant_net(x)
        quant_output_temp = quant_net_temp(x)
    
    print(abs(output - quant_output).max())
    print(abs(output - quant_output_temp).max())
    
    p1, p2 = dataset_test(net, val_dataset, new_weights_temp)
    
    checkpoint['model_state_dict'] = new_weights_temp
    checkpoint['top1'] = p1
    checkpoint['weight_scale'] = scale_list
    
    torch.save(checkpoint, save_path)
    print("weight saved at ", save_path)


if __name__ == "__main__":
    make_quant_weights("experiments/tf_fp32/pretrained_fused.pth")