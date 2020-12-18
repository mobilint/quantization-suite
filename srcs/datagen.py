#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:03:03 2020

@author: js
"""

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

IMAGE_NET_DATA_PATH = '/ssd/datasets/public/imagenet/'


def get_data_loader(config):
    batch_size = config['batch_size']
    traindir = os.path.join(IMAGE_NET_DATA_PATH, 'train')
    valdir = os.path.join(IMAGE_NET_DATA_PATH, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                     std=[0.5, 0.5, 0.5])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
#    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=3, pin_memory=True)
#        num_workers=3, pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=3, pin_memory=True)
    
    return train_loader, val_loader