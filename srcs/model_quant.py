#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:06:16 2020

@author: js
"""


import torch
import torch.nn as nn
from utils_down_resol import DownResolFunction, DownResolModule, DynamicDownResolModule, calcScale


num_bits = 8

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, blockstride=1):
        super(BasicBlock, self).__init__()
        self.blockstride = blockstride
        self.channel_change = in_planes != planes
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=blockstride, padding=1, bias=True)
        self.down_resol1 = DynamicDownResolModule(num_bits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.down_resol2 = DynamicDownResolModule(num_bits)
        
        if blockstride > 1 or in_planes != planes:
            self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=blockstride, padding=0, bias=True)
            self.down_resol3 = DynamicDownResolModule(num_bits)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.down_resol1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.down_resol2(out)
        
        if self.blockstride > 1 or self.channel_change:
            residual = self.conv3(residual)
            residual = self.down_resol3(residual)

        out += residual
        out = self.relu(out)

        return out


class ResBlock(nn.Module):

    def __init__(self, in_planes, inter_planes, planes, blockstride=1):
        
        super(ResBlock, self).__init__()
        self.blockstride = blockstride
        self.channel_change = in_planes != planes
        
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.down_resol1 = DynamicDownResolModule(num_bits)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=blockstride, padding=1, bias=True)
        self.down_resol2 = DynamicDownResolModule(num_bits)
        self.conv3 = nn.Conv2d(inter_planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.down_resol3 = DynamicDownResolModule(num_bits)
        
        if blockstride > 1 or in_planes != planes:
            self.conv4 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=blockstride, padding=0, bias=True)
            self.down_resol4 = DynamicDownResolModule(num_bits)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.down_resol1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.down_resol2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.down_resol3(out)
        
        if self.blockstride > 1 or self.channel_change:
            residual = self.conv4(residual)
            residual = self.down_resol4(residual)

        out += residual
        out = self.relu(out)

        return out


class Resnet34(nn.Module):
    
    def __init__(self, unet=False):
        super(Resnet34, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.unet = unet
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Block1
        self.block1_1 = BasicBlock(64, 64)
        self.block1_2 = BasicBlock(64, 64)
        self.block1_3 = BasicBlock(64, 64)
        # Block2
        self.block2_1 = BasicBlock(64, 128, blockstride=2)
        self.block2_2 = BasicBlock(128, 128)
        self.block2_3 = BasicBlock(128, 128)
        self.block2_4 = BasicBlock(128, 128)
        # Block3
        self.block3_1 = BasicBlock(128, 256, blockstride=2)
        self.block3_2 = BasicBlock(256, 256)
        self.block3_3 = BasicBlock(256, 256)
        self.block3_4 = BasicBlock(256, 256)
        self.block3_5 = BasicBlock(256, 256)
        self.block3_6 = BasicBlock(256, 256)
        # Block4
        self.block4_1 = BasicBlock(256, 512, blockstride=2)
        self.block4_2 = BasicBlock(512, 512)
        self.block4_3 = BasicBlock(512, 512)
        
        self.avgpool =nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512, 1000, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        c1 = self.relu(x)
        
        x = self.maxpool(c1)
        x = self.block1_1(x)
        x = self.block1_2(x)
        c2 = self.block1_3(x)
        
        x = self.block2_1(c2)
        x = self.block2_2(x)
        x = self.block2_3(x)
        c3 = self.block2_4(x)
        
        x = self.block3_1(c3)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        c4 = self.block3_6(x)

        x = self.block4_1(c4)
        x = self.block4_2(x)
        c5 = self.block4_3(x)
        
        x = self.avgpool(c5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.unet:
            return c1, c2, c3, c4, c5
        else:
            return x


class Resnet50(nn.Module):
    
    def __init__(self, unet=False):
        super(Resnet50, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.unet=unet
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.down_resol1 = DynamicDownResolModule(num_bits)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # BLock1
        self.block1_1 = ResBlock(64, 64, 256)
        self.block1_2 = ResBlock(256, 64, 256)
        self.block1_3 = ResBlock(256, 64, 256)
        # Block2
        self.block2_1 = ResBlock(256, 128, 512, blockstride=2)
        self.block2_2 = ResBlock(512, 128, 512)
        self.block2_3 = ResBlock(512, 128, 512)
        self.block2_4 = ResBlock(512, 128, 512)
        # Block3
        self.block3_1 = ResBlock(512, 256, 1024, blockstride=2)
        self.block3_2 = ResBlock(1024, 256, 1024)
        self.block3_3 = ResBlock(1024, 256, 1024)
        self.block3_4 = ResBlock(1024, 256, 1024)
        self.block3_5 = ResBlock(1024, 256, 1024)
        self.block3_6 = ResBlock(1024, 256, 1024)
        # Block4
        self.block4_1 = ResBlock(1024, 512, 2048, blockstride=2)
        self.block4_2 = ResBlock(2048, 512, 2048)
        self.block4_3 = ResBlock(2048, 512, 2048)
        
        self.avgpool =nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(2048, 1000, bias=True)
        self.down_resol2 = DynamicDownResolModule(num_bits)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.down_resol1(x)
        c1 = self.relu(x)
        
        x = self.maxpool(c1)
        x = self.block1_1(x)
        x = self.block1_2(x)
        c2 = self.block1_3(x)
        
        x = self.block2_1(c2)
        x = self.block2_2(x)
        x = self.block2_3(x)
        c3 = self.block2_4(x)
        
        x = self.block3_1(c3)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        c4 = self.block3_6(x)

        x = self.block4_1(c4)
        x = self.block4_2(x)
        c5 = self.block4_3(x)
        
        x = self.avgpool(c5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.down_resol2(x)
        
        if self.unet:
            return c1, c2, c3, c4, c5
        else:
            return x


if __name__ == "__main__":
    net = Resnet34()
    x = torch.rand((2,3,224,224))
    output = net(x)
    print(output.shape)

    net = Resnet50()
    x = torch.rand((2,3,224,224))
    output = net(x)
    print(output.shape)
