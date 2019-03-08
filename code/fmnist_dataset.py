#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:43:37 2019

@author: zhanghuangzhao
"""

import numpy
import random
from mnist import MNIST

class dataset(object):
        
    def __init__(self, x, y):
            
        self.__x = numpy.asarray(x).reshape((-1, 28, 28, 1))
        self.__y = numpy.asarray(y)
        self.size = self.__x.shape[0]
        self.__idx = []
        self.__shuffle()
        self.images = self.__x
        self.labels = self.__y
            
    def __shuffle(self):
            
        self.__idx = random.sample(range(self.size), self.size)
        
    def reset_epoch(self):
        
        self.__shuffle()
            
    def next_batch(self, batch_size, dtype=numpy.float32):
            
        if len(self.__idx) < batch_size:
            self.__shuffle()
        assert batch_size <= self.size, \
            "Batch size %d is larger than dataset size %d" % (batch_size, self.size)
                
        idx = self.__idx[:batch_size]
        self.__idx = self.__idx[batch_size:]
        x, y = ([], [])
        for i in idx:
            x.append(self.__x[i])
            y.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            y[-1][self.__y[i]] = 1
        return numpy.asarray(x, dtype=dtype), numpy.asarray(y, dtype=dtype)

class Fashion_MNIST(object):
    
    def __init__(self, validation_size=10000, data_dir="../data"):
        
        fmnist = MNIST(data_dir, return_type="lists")
        train = fmnist.load_training()
        test = fmnist.load_testing()
        
        assert validation_size >= 0 and validation_size <= len(train[0]), \
            "Invalid validattion ratio %.3f, should be within 0 to %d" \
            % (validation_size, len(train[0]))

        idx = random.sample(range(len(train[0])), len(train[0]))
        tr, va = ([[], []], [[], []])
        for i in idx[:validation_size]:
            va[1].append(train[1][i])
            va[0].append(train[0][i])
        for i in idx[validation_size:]:
            tr[1].append(train[1][i])
            tr[0].append(train[0][i])
        self.train = dataset(tr[0], tr[1])
        self.valid = dataset(va[0], va[1])
        self.test = dataset(test[0], test[1])
        
        self.__label_dict = ["t-shirt/top", "trouser", "pullover", "dress",
                             "coat", "sandal", "shirt", "sneaker", "bag",
                             "ankle boot"]

    def get_label(self, label_idx):
        
        return self.__label_dict[label_idx]
    
    def get_labels(self):
        
        return self.__label_dict

if __name__ == "__main__":
    
    fmnist = Fashion_MNIST()
    
    try:
        import matplotlib.pyplot as plt
        for i in range(10):
            x, y = fmnist.train.next_batch(1)
            while not numpy.argmax(y) == i:
                x, y = fmnist.train.next_batch(1)
            plt.imshow(x.reshape([28, 28]))
            plt.imsave("../images/fmnist_%d.jpg"%i, x.reshape([28, 28]))
    except Exception as e:
        print(e)