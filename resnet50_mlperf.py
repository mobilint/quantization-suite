#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:25:51 2020

@author: js
"""

import numpy as np
import onnx
import onnxruntime
import os
import cv2


val_dir = "/ssd/datasets/public/imagenet/val/"
onnx_model_path = "/home/js/models/tf_compiling/onnx_model/resnet50_v1.onnx"


def resize_center_crop(img, out_size=(224, 224)):
    scale = 100 / 87.5
    out_height, out_width = out_size
    resize_height = int(out_height * scale)
    resize_width = int(out_width * scale)
    
    height, width, _ = img.shape
    if height > width:
        resize_width = resize_width
        resize_height = int(resize_height * height / width)
    else:
        resize_width = int(resize_width * width / height)
        resize_height = resize_height
    
    img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    
    left = int((resize_width - out_width) / 2)
    right = int((resize_width + out_width) / 2)
    top = int((resize_height - out_height) / 2)
    bottom = int((resize_height + out_height) / 2)
    
    img = img[top:bottom, left:right]
    
    return img


def normalize_mobilenet(img):
    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5
    
    return img

def normalize_resnet_tf(img):
    img = img.astype(np.float32)
    img[:, :, 0] = img[:, :, 0] - 123.68
    img[:, :, 1] = img[:, :, 1] - 116.78
    img[:, :, 2] = img[:, :, 2] - 103.94
    
    return img

def normalize_resnet_torch(img):
    img = img.astype(np.float32)
    img[:, :, 0] = (img[:, :, 0] / 255 - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] / 255 - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] / 255 - 0.406) / 0.225
    
    return img

# load val map
with open(os.path.join(val_dir, 'val_map.txt'), 'r') as f:
    lines = f.read().splitlines()

# load model
onnx_model = onnx.load(onnx_model_path)
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

correct = 0
for line in lines:
    filename, label = line.split(' ')
    
    filepath = os.path.join(val_dir, 'val', filename)
    label = int(label)
    
    img = cv2.imread(filepath)
    img = np.asarray(img)
    if len(img.shape) < 3 or img.shape[2] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = resize_center_crop(img, out_size=(224, 224))
    img = normalize_resnet_tf(img)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    
    onnx_input = {onnx_session.get_inputs()[0].name: img}
    onnx_output = onnx_session.run(None, onnx_input)
    
    if label + 1 == onnx_output:
        correct += 1

print(correct / len(lines))









