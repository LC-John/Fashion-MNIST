#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhanghuangzhao
"""


import torch
from torch import nn

class CNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(1, 24, 5, 1, 2),
                nn.ReLU(),
                nn.Conv2d(24, 48, 5, 2, 2),
                nn.ReLU(),
                nn.Conv2d(48, 64, 5, 3, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(5 * 5 * 64, 200),
                nn.ReLU(),
                nn.Linear(200, 10)
            )

    def forward(self, x):
        
        x = x.reshape((-1, 1, 28, 28))
        return self.layers(x)



        

if __name__ == "__main__":
    
    from fmnist_dataset import load_fashion_mnist
    from torch.utils.data import DataLoader

    train, dev, test = load_fashion_mnist("../data")
    train_dataloader = DataLoader(train, batch_size=1)
    
    m = CNN()
    
    for x, y in train_dataloader:
        
        l = m(x)
        break