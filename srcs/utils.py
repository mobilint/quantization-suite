#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:01:57 2020

@author: js
"""

import torch
import os
import json


def load_config(exp_name):
    path = os.path.join('./experiments', exp_name, 'config.json')
    with open(path) as file:
        config = json.load(file)
        
    assert config['name'] == exp_name
    
    return config


def get_model_path(config, model_name):
    name = config['name']
    
    folder = os.path.join('./experiments', name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if config['mGPUs']:
        path = os.path.join(folder, model_name + "_mGPUs.pth")
    else: 
        path = os.path.join(folder, model_name + ".pth")

    if os.path.isfile(path):
        print("model from : {}".format(path))
        exist_best_model = True
        return path, exist_best_model
    else:
        print("There is no " + model_name + " model")
        exist_best_model = False
        return path, exist_best_model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res