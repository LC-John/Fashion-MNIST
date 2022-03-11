#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhanghuangzhao
"""

import argparse
import os
from fmnist_dataset import load_fashion_mnist
from model import CNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import random
    
    
    
    
def gettensor(x, y, device):
    
    return x.to(device), y.to(device)
    
    

    
def trainEpochs(classifier, optimizer, loss_fn, epochs, training_set, dev_set,
                print_each, save_dir, device):
    
    for ep in range(1, epochs + 1):
        
        classifier.train()
        print_loss_total = 0
        
        print ('Ep %d' % ep)
        
        for i, (x, y) in enumerate(training_set):
            
            optimizer.zero_grad()
            x, y = gettensor(x, y, device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            print_loss_total += loss.item()
            if (i + 1) % print_each == 0: 
                print_loss_avg = print_loss_total / print_each
                print_loss_total = 0
                print('    %.4f' % print_loss_avg)
                
        acc = evaluate(classifier, dev_set, device)
        print ('  dev acc = %.2f%%' % acc)
        torch.save(classifier.state_dict(),
                   os.path.join(save_dir, 'ep_' + str(ep) + '_devacc_' + str(acc) + '_.pt'))
        
            
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
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--save_dir', type=str, default='../model')
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--log_per_step', type=int, default=100)
    args = parser.parse_args()
    
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
    
    train, dev, _ = load_fashion_mnist("../data", random=random)
    train_dataloader = DataLoader(train, batch_size=opt.batch_size, drop_last=True)
    dev_dataloader = DataLoader(dev, batch_size=opt.eval_batch_size)

    
    classifier = CNN()
    optimizer = optim.Adam(classifier.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    
    trainEpochs(classifier, optimizer, criterion, opt.num_epochs,
                train_dataloader, dev_dataloader,
                opt.log_per_step, opt.save_dir, device)
