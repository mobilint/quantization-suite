#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:15:44 2020

@author: js
"""

import argparse
import os
import time
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
from warmup_scheduler import GradualWarmupScheduler
from datagen import get_data_loader
from utils import load_config, AverageMeter, accuracy, get_model_path
from utils_down_resol import weights_quant, weights_quant_b, weights_quant_with_scale
from quant_inference import scale_inference
import model, model_fused, model_quant
import model_tf, model_tf_fused


def build_model(config, device, train=True):
    # load model
    if config['model'] == 'default':
        net = model.Resnet50()
    elif config['model'] == 'fused':
        net = model_fused.Resnet50()
    elif config['model'] == 'quant':
        net = model_quant.Resnet50()
    elif config['model'] == 'tf':
        net = model_tf.Resnet50()
    elif config['model'] == 'tf_fused':
        net = model_tf_fused.Resnet50()
    else:
        raise ValueError('cannot load model, check config file')
    # load loss
    if config['loss'] == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError('cannot load loss, check config file')
    
    net = net.to(device)
    loss_fn = loss_fn.to(device)
    
    if not train:
        return net, loss_fn
    # load optimizer
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), 
                                    lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                                     lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError('cannot load optimizer, check config file')
    # load scheduler
    if config['scheduler'] == 'cosine':
        scheduler_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'])
    elif config['scheduler'] == 'step':
        scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_every'], gamma=config["lr_decay"])
    else:
        raise ValueError('cannot load scheduler, check config file')
    scheduler = GradualWarmupScheduler(optimizer, multiplier=config['lr_multiplier'], total_epoch=config['lr_epoch'], after_scheduler=scheduler_step)
    
    return net, loss_fn, optimizer, scheduler
    

def evaluation(config, net, loss_fn, loader, device):
    net.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    t_fwd = 0
    
    for input, target in tqdm(loader):
        input_var = input.to(device)
        target_var = target.to(device)
        
        if config['model'] == 'tf' or config['model'] == 'tf_fused':
            input_var[:, 0, :, :] = (input_var[:, 0, :, :] * 0.229 + 0.485) * 255 - 123.68
            input_var[:, 1, :, :] = (input_var[:, 1, :, :] * 0.224 + 0.456) * 255 - 116.78
            input_var[:, 2, :, :] = (input_var[:, 2, :, :] * 0.225 + 0.406) * 255 - 103.94
        
        # compute output
        tic = time.time()
        output = net(input_var)
        if output.shape[1] == 1001:
            target_var += 1
        t_fwd += time.time() - tic
        loss = loss_fn(output, target_var)
#        print(torch.argmax(output, dim=1), target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    metrics = {}
    metrics['top1'] = top1.avg.item()
    metrics['top5'] = top5.avg.item()
    metrics['Forward Pass Time'] = t_fwd / len(loader.dataset)
    metrics['loss'] = losses.avg
    
    return metrics


def quant_evaluation(net, loss_fn, loader, device, scale_list, num_bits=8):
    net.eval()
    
    with torch.no_grad():
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        t_fwd = 0
        
        
        for input, label in loader:
            input = input.to(device)
            label = label.to(device)
            
            # compute output
            tic = time.time()
            output = scale_inference(net, input, scale_list, num_bits).cpu()
            t_fwd += time.time() - tic
            loss = loss_fn(output, label)
                        
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
    metrics = {}
    metrics['top1'] = top1.avg.item()
    metrics['top5'] = top5.avg.item()
    metrics['Forward Pass Time'] = t_fwd / len(loader.dataset)
    metrics['loss'] = losses.avg
    
    return top1.avg.item(), top5.avg.item()


def train(exp_name, device, epoch):
    config = load_config(exp_name)
    max_epochs = config['max_epochs']
    
    print("make data loader")
    train_data_loader, val_data_loader = get_data_loader(config)
    
    net, loss_fn, optimizer, scheduler = build_model(config, device, train=True)
    
    best_ckpt_path, exist_best_model = get_model_path(config, "best")
    best_top1 = 0
    if exist_best_model:
        best_checkpoint = torch.load(best_ckpt_path)
        best_top1 = best_checkpoint['top1']
        print("best top1 score is {:.3f}".format(best_top1))
    
    if config['resume_training']:
        saved_ckpt_path = get_model_path(config, "epoch")
        checkpoint = torch.load(saved_ckpt_path, map_location=device)
        if config['mGPUs']:
            net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded trained ckpt at {}".format(saved_ckpt_path))
        st_epoch = config['resume_from']
    else:
        st_epoch = 0

    for g in optimizer.param_groups:
        g['lr'] = config['learning_rate']

    for epoch in range(st_epoch, max_epochs):
        start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        net.train()
        
        print("Epoch {}, learning rate : {}".format(epoch+1, scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
        
        # train
        start = time.time()
        for input, target in tqdm(train_data_loader):
            # measure data loading time
            data_time.update(time.time() - start)
            
            input_var = input.to(device)
            target_var = target.to(device)
            
            # compute output
            output = net(input_var)
            loss = loss_fn(output, target_var)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            # compute gradient and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

        scheduler.step()
        print("Epoch {}|Time {:.3f}|Training Loss: {:.5f}".format(epoch + 1, time.time() - start_time, losses.avg))
        print("\033[32m training   || Prec@1: {:.3f} | Prec@5: {:.3f} \033[37m".format(top1.avg, top5.avg))
        
        # validation
        tic_val = time.time()
        val_metrics = evaluation(config, net, loss_fn, val_data_loader, device)
        print("Epoch {}|Time {:.3f}|Validation Loss: {:.5f}".format(epoch +1, time.time() - tic_val, val_metrics['loss']))
        print("\033[32m validation || Prec@1: {:.3f} | Prec@5: {:.3f} \033[37m".format(val_metrics['top1'], val_metrics['top5']))
        
        # save best model
        if val_metrics['top1'] > best_top1:
            if config['mGPUs']:
                torch.save({'model_state_dict': net.module.state_dict(),
                            'top1': val_metrics['top1']}, best_ckpt_path)
            else:
                torch.save({'model_state_dict': net.state_dict(),
                            'top1': val_metrics['top1']}, best_ckpt_path)
            print("\033[32m Best model saved at {}. Prec@1 is {} \033[37m".format(best_ckpt_path, val_metrics['top1']))
            best_top1 = val_metrics['top1']


def quant_train(exp_name, device, epoch):
    num_bits = 8
    
    config = load_config(exp_name)
    config['resume_training'] = True
    config['resume_from'] = 0
    max_epochs = config['max_epochs']

    print("make data loader")
    train_data_loader, val_data_loader = get_data_loader(config)

    net, loss_fn, optimizer, scheduler = build_model(config, device, train=True)

    ckpt_path, exist_model = get_model_path(config, epoch)
    best_top1 = 0
    if exist_model:
        checkpoint = torch.load(ckpt_path)
        if 'top1' in checkpoint.keys():
            best_top1 = checkpoint['top1']
            print("best top1 score is {:.3f}".format(best_top1))
            best_top1 = best_top1 * 0.9     # 90% accuracy
    else:
        print()

    saved_ckpt_path = ckpt_path
    checkpoint = torch.load(saved_ckpt_path, map_location=device)
    
    if 'model_state_dict' in checkpoint.keys():
        weights = checkpoint['model_state_dict']
    else:
        weights = checkpoint
    
    if config['mGPUs']:
        net.module.load_state_dict(weights)
    else:
        net.load_state_dict(weights)

    print("Successfully loaded trained ckpt at {}".format(saved_ckpt_path))
    st_epoch = 0

    for g in optimizer.param_groups:
        g['lr'] = config['learning_rate']
        
    # quant weight
    quant_weights, weight_scale_list = weights_quant(weights, num_bits=num_bits)
#    quant_weights, weight_scale_list = weights_quant_b(net, val_data_loader.dataset, device, weights, num_bits=num_bits)
    net.load_state_dict(quant_weights)

    for epoch in range(st_epoch, max_epochs):
        start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        net.train()
        
        print("Epoch {}, learning rate : {}".format(epoch+1, scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
        
        # train
        start = time.time()
        for input, target in tqdm(train_data_loader):
            # measure data loading time
            data_time.update(time.time() - start)
            
            input_var = input.to(device)
            target_var = target.to(device)
            
            if config['model'] == 'tf' or config['model'] == 'tf_fused':
                input_var[:, 0, :, :] = (input_var[:, 0, :, :] * 0.229 + 0.485) * 255 - 123.68
                input_var[:, 1, :, :] = (input_var[:, 1, :, :] * 0.224 + 0.456) * 255 - 116.78
                input_var[:, 2, :, :] = (input_var[:, 2, :, :] * 0.225 + 0.406) * 255 - 103.94
            
            # compute output
            output = net(input_var)
            if output.shape[1] == 1001:
                target_var += 1
            loss = loss_fn(output, target_var)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            # compute gradient and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()
            
        scheduler.step()
        print("Epoch {}|Time {:.3f}|Training Loss: {:.5f}".format(epoch + 1, time.time() - start_time, losses.avg))
        print("\033[32m training   || Prec@1: {:.3f} | Prec@5: {:.3f} \033[37m".format(top1.avg, top5.avg))
        
        # validation before quantize
        tic_val = time.time()
        val_metrics = evaluation(config, net, loss_fn, val_data_loader, device)
        print("Epoch {}|Time {:.3f}|Validation Loss: {:.5f}".format(epoch +1, time.time() - tic_val, val_metrics['loss']))
        print("\033[32m validation || Prec@1: {:.3f} | Prec@5: {:.3f} \033[37m".format(val_metrics['top1'], val_metrics['top5']))
        
        # save recent model
        
        # quant weight
        if epoch == 0:
            weights = net.state_dict()
            quant_weights, weight_scale_list = weights_quant(weights, num_bits=num_bits)
#            quant_weights, weight_scale_list = weights_quant_b(net, val_data_loader.dataset, device, weights, num_bits=num_bits)
            net.load_state_dict(quant_weights)
        else:
            weights = net.state_dict()
#            quant_weights, weight_scale_list = weights_quant_with_scale(weights, weight_scale_list, num_bits)
            quant_weights, weight_scale_list = weights_quant(weights, num_bits=num_bits)
            net.load_state_dict(quant_weights)
        
        # validation after quantize
        tic_val = time.time()
        val_metrics = evaluation(config, net, loss_fn, val_data_loader, device)
        print("Epoch {}|Time {:.3f}|Validation Loss: {:.5f}".format(epoch +1, time.time() - tic_val, val_metrics['loss']))
        print("\033[32m validation || Prec@1: {:.3f} | Prec@5: {:.3f} \033[37m".format(val_metrics['top1'], val_metrics['top5']))
        
        # save best model
        if val_metrics['top1'] > best_top1:
            if config['mGPUs']:
                torch.save({'model_state_dict': net.module.state_dict(),
                            'top1': val_metrics['top1'],
                            'weight_scale': weight_scale_list}, ckpt_path[:-4] + '_quant.pth')
            else:
                torch.save({'model_state_dict': net.state_dict(),
                            'top1': val_metrics['top1'],
                            'weight_scale': weight_scale_list}, ckpt_path[:-4] + '_quant.pth')
            print("\033[32m Best model saved at {}. Prec@1 is {} \033[37m".format(ckpt_path[:-4] + '_quant.pth', val_metrics['top1']))
            best_top1 = val_metrics['top1']


def experiment(exp_name, device, epoch):
    config = load_config(exp_name)
    config['augmentation'] = False
    net, loss_fn = build_model(config, device, train=False)
    
    model_path, exist_best_model = get_model_path(config, epoch)
    assert exist_best_model, "There is no model"
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint.keys():
        weights = checkpoint['model_state_dict']
    else:
        weights = checkpoint
    
    if config['mGPUs']:
        net.module.load_state_dict(weights)
    else:
        net.load_state_dict(weights)
    train_loader, val_loader = get_data_loader(config)

#    train_metrics = evaluation(config, net, loss_fn, train_loader, device)
#    print("------Training Result------")
#    print("Prec@1           : ", train_metrics['top1'])
#    print("Prec@2           : ", train_metrics['top5'])
#    print("Forward Pass Time: ", train_metrics['Forward Pass Time'])
#    print("loss             : ", train_metrics['loss'])
    
    val_metrics = evaluation(config, net, loss_fn, val_loader, device)
    print("------Validation Result------")
    print("Prec@1           : ", val_metrics['top1'])
    print("Prec@5           : ", val_metrics['top5'])
    print("Forward Pass Time: ", val_metrics['Forward Pass Time'])
    print("loss             : ", val_metrics['loss'])

    if 'model_state_dict' not in checkpoint.keys():
        torch.save({'model_state_dict': weights, 'top1': val_metrics['top1']}, model_path)
        print("model saved at ", model_path)


def quant_experiment(exp_name, device, epoch):
    config = load_config(exp_name)
    config['augmentation'] = False
    num_bits = config['num_bits']
    net, loss_fn = build_model(config, device, train=False)
    
    model_path, exist_best_model = get_model_path(config, epoch)
    assert exist_best_model, "There is no model"
    
    checkpoint = torch.load(model_path, map_location=device)
    weights = checkpoint['model_state_dict']
    if 'weight_scale' in checkpoint.keys():
        weight_scale_list = checkpoint['weight_scale']
        print("weight scale loaded")
        weights, weight_scale_list = weights_quant_with_scale(weights, weight_scale_list, num_bits)
        print("weight scale completed")
    
    if 'act_scale' in checkpoint.keys():
        act_scale_list = checkpoint['act_scale']
        print("act scale loaded")
    
    if config['mGPUs']:
        net.module.load_state_dict(weights)
    else:
        net.load_state_dict(weights)
        
    train_loader, val_loader = get_data_loader(config)
    
#    train_metrics = evaluation(config, net, loss_fn, train_loader, device)
#    print("------Training Result------")
#    print("Prec@1           : ", train_metrics['top1'])
#    print("Prec@2           : ", train_metrics['top5'])
#    print("Forward Pass Time: ", train_metrics['Forward Pass Time'])
#    print("loss             : ", train_metrics['loss'])
    
    if 'act_scale' in checkpoint.keys():
        val_metrics = quant_evaluation(net, loss_fn, val_loader, device, act_scale_list, num_bits)
    else:
        val_metrics = evaluation(config, net, loss_fn, val_loader, device)
        
    print("------Validation Result------")
    print("Prec@1           : ", val_metrics['top1'])
    print("Prec@5           : ", val_metrics['top5'])
    print("Forward Pass Time: ", val_metrics['Forward Pass Time'])
    print("loss             : ", val_metrics['loss'])    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MOBINET custom implementation')
    parser.add_argument('mode', choices=['train', 'val', 'run', 'quant_train', 'quant_val'], help='name of the experiment')
    parser.add_argument('--name', default='default', help="name of the experiment")
    parser.add_argument('--device', default='cuda', help='device to train on')
    parser.add_argument('--epoch', default='best', help='epoch for model')
    args = parser.parse_args()

    torch.set_num_threads(10)
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    print("\033[37m Using device", device)

    if args.mode=='train':
        train(args.name, device, args.epoch)
    if args.mode=='val':
        experiment(args.name, device, args.epoch)
    if args.mode=='quant_train':
        quant_train(args.name, device, args.epoch)
    if args.mode=='quant_val':
        quant_experiment(args.name, device, args.epoch)
#    if args.mode=='run':
#        inference(args.name, device, args.epoch)
