#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhanghuangzhao
"""

import argparse
import os
import pickle
from fmnist_dataset import load_fashion_mnist
from model import CNN

import torch
from torch.utils.data import DataLoader
import random
    
    
    
    
def gettensor(x, y, device):
    
    return x.to(device), y.to(device)
        
            
def evaluate(classifier, dataset, device):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    for x, y in dataset:
        
        with torch.no_grad():
            x, y = gettensor(x, y, device)
            logits = classifier(x)
            res = torch.argmax(logits, dim=1) == y
            testcorrect += torch.sum(res)
            testnum += len(y)
    
    acc = float(testcorrect) * 100.0 / testnum
    return acc
    
def generate_data_for_black_box_attack(classifier, dataset, device, max_num=1000):
    
    classifier.eval()
    image, label = [], []
    
    for x, y in dataset:
        
        with torch.no_grad():
            x, y = gettensor(x, y, device)
            logits = classifier(x)
            res = torch.argmax(logits, dim=1) == y
            
        for i in range(x.shape[0]):
            if res[i].item():
                image.append(x[i].numpy())
                label.append(y[i].numpy())
            if len(image) >= max_num:
                return image, label
            
    return image, label
            
            
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--save_path', type=str, default='../model/cnn.ckpt')
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--gen_data_path', type=str, default='../attack_data/correct_1k.pkl')
    
    opt = parser.parse_args()
    
    if int(opt.gpu) < 0:
        device = torch.device('cpu')
        torch.manual_seed(opt.rand_seed)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        torch.manual_seed(opt.rand_seed)
        torch.cuda.manual_seed(opt.rand_seed)
        
    random.seed(opt.rand_seed)
    
    _, __, test = load_fashion_mnist("../data", random=random)
    test_dataloader = DataLoader(test, batch_size=opt.eval_batch_size)

    
    classifier = CNN()
    classifier.load_state_dict(torch.load(opt.save_path))
    
    acc = evaluate(classifier, test_dataloader, device)
    print ('test acc = %.2f%%' % acc)
    
    img, label = generate_data_for_black_box_attack(classifier, test_dataloader, device)
    with open(opt.gen_data_path, "wb") as f:
        pickle.dump([img, label], f)
    