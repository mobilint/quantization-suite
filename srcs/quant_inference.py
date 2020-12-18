#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:48:02 2020

@author: js
"""

import model_fused
import torch
import copy
import torchvision
from utils import AverageMeter, accuracy
from utils_down_resol import calcScale, DownResolFunction


def scale_inference(net, input, scale_list, num_bits):
    net.eval()
    Down_resol = DownResolFunction.apply
    
    with torch.no_grad():
        x = net.model[0].pad1(input)
        x = net.model[0].conv1(x)
        x = Down_resol(x, num_bits, scale_list[0])
        x = net.model[0].relu(x)
        
        x = net.model[1].conv1(x)
        x = Down_resol(x, num_bits, scale_list[1])
        x = net.model[1].relu(x)
        x = net.model[1].conv2(x)
        x = Down_resol(x, num_bits, scale_list[2])
        x = net.model[1].relu(x)
        
        x = net.model[2].pad1(x)
        x = net.model[2].conv1(x)
        x = Down_resol(x, num_bits, scale_list[3])
        x = net.model[2].relu(x)
        x = net.model[2].conv2(x)
        x = Down_resol(x, num_bits, scale_list[4])
        x = net.model[2].relu(x)
        
        x = net.model[3].conv1(x)
        x = Down_resol(x, num_bits, scale_list[5])
        x = net.model[3].relu(x)
        x = net.model[3].conv2(x)
        x = Down_resol(x, num_bits, scale_list[6])
        x = net.model[3].relu(x)
        
        x = net.model[4].pad1(x)
        x = net.model[4].conv1(x)
        x = Down_resol(x, num_bits, scale_list[7])
        x = net.model[4].relu(x)
        x = net.model[4].conv2(x)
        x = Down_resol(x, num_bits, scale_list[8])
        x = net.model[4].relu(x)
        
        x = net.model[5].conv1(x)
        x = Down_resol(x, num_bits, scale_list[9])
        x = net.model[5].relu(x)
        x = net.model[5].conv2(x)
        x = Down_resol(x, num_bits, scale_list[10])
        x = net.model[5].relu(x)
        
        x = net.model[6].pad1(x)
        x = net.model[6].conv1(x)
        x = Down_resol(x, num_bits, scale_list[11])
        x = net.model[6].relu(x)
        x = net.model[6].conv2(x)
        x = Down_resol(x, num_bits, scale_list[12])
        x = net.model[6].relu(x)
        
        x = net.model[7].conv1(x)
        x = Down_resol(x, num_bits, scale_list[13])
        x = net.model[7].relu(x)
        x = net.model[7].conv2(x)
        x = Down_resol(x, num_bits, scale_list[14])
        x = net.model[7].relu(x)
        
        x = net.model[8].conv1(x)
        x = Down_resol(x, num_bits, scale_list[15])
        x = net.model[8].relu(x)
        x = net.model[8].conv2(x)
        x = Down_resol(x, num_bits, scale_list[16])
        x = net.model[8].relu(x)
        
        x = net.model[9].conv1(x)
        x = Down_resol(x, num_bits, scale_list[17])
        x = net.model[9].relu(x)
        x = net.model[9].conv2(x)
        x = Down_resol(x, num_bits, scale_list[18])
        x = net.model[9].relu(x)
        
        x = net.model[10].conv1(x)
        x = Down_resol(x, num_bits, scale_list[19])
        x = net.model[10].relu(x)
        x = net.model[10].conv2(x)
        x = Down_resol(x, num_bits, scale_list[20])
        x = net.model[10].relu(x)
        
        x = net.model[11].conv1(x)
        x = Down_resol(x, num_bits, scale_list[21])
        x = net.model[11].relu(x)
        x = net.model[11].conv2(x)
        x = Down_resol(x, num_bits, scale_list[22])
        x = net.model[11].relu(x)
        
        x = net.model[12].pad1(x)
        x = net.model[12].conv1(x)
        x = Down_resol(x, num_bits, scale_list[23])
        x = net.model[12].relu(x)
        x = net.model[12].conv2(x)
        x = Down_resol(x, num_bits, scale_list[24])
        x = net.model[12].relu(x)
        
        x = net.model[13].conv1(x)
        x = Down_resol(x, num_bits, scale_list[25])
        x = net.model[13].relu(x)
        x = net.model[13].conv2(x)
        x = Down_resol(x, num_bits, scale_list[26])
        x = net.model[13].relu(x)
        
        x = torch.nn.functional.avg_pool2d(x, 7)
        
        x = net.last_conv(x)
        x = Down_resol(x, num_bits, scale_list[27])
        x = x.view(-1, 1001)
        
    return x

def dataset_test(net, val_dataset, scale_list, num_bits=8):
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=3, pin_memory=True)
    
    with torch.no_grad():
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        net = net.cuda()
        net.eval()
        for input, label in val_loader:
            input = input.cuda()
            output = scale_inference(net, input, scale_list, num_bits).cpu()
            label = label + 1
            
            prec1, prec5 = accuracy(output.data, label, topk=(1,5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
        print("Prec @1: ", top1.avg.item())
    return top1.avg.item(), top5.avg.item()


def activation_quantize(weight_path, num_bits=8):
    # load dataset
    val_dir = "/ssd/datasets/public/imagenet/val/"
    normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize, ]))
    
    # load weight
    checkpoint = torch.load(weight_path)
    weights = checkpoint['model_state_dict']
    
    # load model
    net = model_fused.Mobilenet_v1()
    net.load_state_dict(weights)
    net.eval()
    
    input, label = val_dataset[300]
    input = input.unsqueeze(0)
    
    output = net(input)
    
    scale_list = []
    
    x = net.model[0].pad1(input)
    x = net.model[0].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[0].relu(x)
    
    x = net.model[1].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[1].relu(x)
    x = net.model[1].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[1].relu(x)
    
    x = net.model[2].pad1(x)
    x = net.model[2].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[2].relu(x)
    x = net.model[2].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[2].relu(x)
    
    x = net.model[3].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[3].relu(x)
    x = net.model[3].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[3].relu(x)
    
    x = net.model[4].pad1(x)
    x = net.model[4].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[4].relu(x)
    x = net.model[4].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[4].relu(x)
    
    x = net.model[5].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[5].relu(x)
    x = net.model[5].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[5].relu(x)
    
    x = net.model[6].pad1(x)
    x = net.model[6].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[6].relu(x)
    x = net.model[6].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[6].relu(x)
    
    x = net.model[7].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[7].relu(x)
    x = net.model[7].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[7].relu(x)
    
    x = net.model[8].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[8].relu(x)
    x = net.model[8].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[8].relu(x)
    
    x = net.model[9].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[9].relu(x)
    x = net.model[9].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[9].relu(x)
    
    x = net.model[10].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[10].relu(x)
    x = net.model[10].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[10].relu(x)
    
    x = net.model[11].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[11].relu(x)
    x = net.model[11].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[11].relu(x)
    
    x = net.model[12].pad1(x)
    x = net.model[12].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[12].relu(x)
    x = net.model[12].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[12].relu(x)
    
    x = net.model[13].conv1(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[13].relu(x)
    x = net.model[13].conv2(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = net.model[13].relu(x)
    
    x = torch.nn.functional.avg_pool2d(x, 7)
    
    x = net.last_conv(x)
    scale_list.append(calcScale(x.min(), x.max(), num_bits))
    x = x.view(-1, 1001)
    
    print("model test : ", "pass" if abs(output - x).max() == 0 else "failed, check your model")
    scale_output = scale_inference(net, input, scale_list, num_bits)
    
    output_gap = abs(output - scale_output).max()
    print("scale inference gap : ", output_gap)
    
    # minimize output gap
    for i in range(len(scale_list)):
        initial_scale = scale_list[i]
        scale_list_temp = copy.deepcopy(scale_list)
        for j in range(initial_scale-2, initial_scale+2):
            scale_list_temp[i] = j
            scale_output_temp = scale_inference(net, input, scale_list_temp, num_bits)
            output_gap_temp = abs(output - scale_output_temp).max()
            if output_gap_temp < output_gap:
                scale_list[i] = j
                output_gap = output_gap_temp
        
        print(i, scale_list[i], output_gap)
    
    # maximize top1 score
    #top1_score = 0
    #for i in range(len(scale_list)):
    #    initial_scale = scale_list[i]
    #    scale_list_temp = copy.deepcopy(scale_list)
    #    for j in range(initial_scale-2, initial_scale+2):
    #        scale_list_temp[i] = j
    #        top1, top5 = dataset_test(net, val_dataset, scale_list_temp)
    #        if top1 > top1_score:
    #            top1_score = top1
    #            scale_list[i] = j
    #    print(i, scale_list[i], top1_score)
    
    top1, top5 = dataset_test(net, val_dataset, scale_list, num_bits)
    
    checkpoint['act_scale'] = scale_list
    checkpoint['top1'] = top1
    
    save_path = weight_path[:-4] + '_act.pth'
    torch.save(checkpoint, save_path)

if __name__ == "__main__":
    weight_path = 'experiments/test_new_pretrained/best.pth'
    activation_quantize(weight_path, num_bits=8)