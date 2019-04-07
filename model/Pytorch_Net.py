# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
def conv3_3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False):
        super(residual_block, self).__init__()
        self.downsample = downsample
        stride = 2 if self.downsample else 1

        self.forwardPath = nn.Sequential(
            conv3_3(in_channel, out_channel, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            conv3_3(out_channel, out_channel),
            nn.BatchNorm2d(out_channel),
        )
        if self.downsample:
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.forwardPath(x)
        if self.downsample:
            x = self.conv3(x)
        return nn.functional.relu(x + out, True)


class residual_block2(nn.Module):
    def __init__(self, in_channel, out_channel, down_channel, downsample=False):
        super(residual_block2, self).__init__()

        self.downsample = downsample
        stride = 2 if self.downsample else 1  # downsample or not

        self.x_change = in_channel != out_channel

        self.forwardPath = nn.Sequential(
            nn.Conv2d(in_channel, down_channel, 1, stride=1),  
            nn.BatchNorm2d(down_channel),
            nn.ReLU(True),
            conv3_3(down_channel, down_channel, stride=stride), 
            nn.BatchNorm2d(down_channel),
            nn.ReLU(True),
            nn.Conv2d(down_channel, out_channel, 1, stride=1),  
            nn.BatchNorm2d(out_channel),
        )

        self.convX = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),  # x dim
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.forwardPath(x)
        if self.downsample or self.x_change:
            x = self.convX(x)
        return nn.functional.relu(x + out, True)
class ResNet18_Mnist(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(ResNet18_Mnist, self).__init__()
        self.verbose = verbose
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv2_x = nn.Sequential(
            self.make_layer(2, 1, 16, 16),
        )
        self.conv3_x = nn.Sequential(
            residual_block(16, 32,True),
            self.make_layer(2, 1, 32, 32),
        )
        self.conv4_x = nn.Sequential(
            residual_block(32, 64, True),
            self.make_layer(2, 1, 64, 64),
        )
        self.conv5_x = nn.Sequential(
            nn.AvgPool2d(7)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        if self.verbose:
            print('conv1 output:{}'.format(x.shape))
        x = self.conv2_x(x)
        if self.verbose:
            print('conv2_x output:{}'.format(x.shape))
        x = self.conv3_x(x)
        if self.verbose:
            print('conv3_x output:{}'.format(x.shape))
        x = self.conv4_x(x)
        if self.verbose:
            print('conv4_x output:{}'.format(x.shape))
        x = self.conv5_x(x)
        if self.verbose:
            print('conv5_x output:{}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def make_layer(self, layer_number, block_class, in_channel, out_channel, down_channel=0):
        layers = []
        for i in range(layer_number):
            if block_class == 1:
                layers.append(residual_block(in_channel, out_channel))
            elif block_class == 2: 
                layers.append(residual_block2(in_channel, out_channel, down_channel))
        return nn.Sequential(*layers)