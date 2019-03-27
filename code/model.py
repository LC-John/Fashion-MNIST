#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:47:32 2019

@author: zhanghuangzhao
"""


import tensorflow as tf
import numpy

class CNN(object):
    
    def __init__(self, dtype="fp32", scope_name="", is_inference=False,
                 lr=[1e-5, 2e-2], lr_decay=2000, grad_clip=5):
        
        # Data type
        if dtype == "fp16":
            self.__dtype = numpy.float16
        elif dtype == "fp32":
            self.__dtype = numpy.float32
        elif dtype == "fp64":
            self.__dtype = numpy.float64
        else:
            assert False, "Invalid data type (%s). Use \"fp16\", \"fp32\" or \"fp64\" only" % dtype
        
        # Hyper-parameters
        self.__K = 24      # Conv-1 depth
        self.__stride1 = 1 # Conv-1 stride
        self.__L = 48      # Conv-2 depth
        self.__stride2 = 2 # Conv-2 stride
        self.__M = 64      # Conv-3 depth
        self.__stride3 = 3 # Conv-3 stride
        self.__N = 200     # FC width
        self.__min_lr = lr[0]   # Minimum learning rate
        self.__max_lr = lr[1]   # Maximum learning rate
        self.__decay_step = lr_decay    # Learning rate exponentional decay
        self.__grad_clipping = grad_clip    # Gradient clipping by absolute value
        
        # Add placeholders
        self.__add_placeholders()
        # Add variables
        self.__add_variables()

        # Build graph
        (self.__Ylogits, \
         self.__Y,        \
         self.__update_ema) = self.__build_graph(self.__X)
        self.__trainables = [x for x in tf.global_variables() \
                                if x.name.startswith(scope_name)]
        
        if not is_inference:
            (self.__loss,    \
             self.__accuracy, \
             self.__train_step) = self.__build_training_graph()
        else:
            (self.__loss,    \
             self.__accuracy, \
             self.__train_step) = self.__build_inference_graph()
            
        self.__saver = tf.train.Saver(var_list=self.__trainables,
                                      max_to_keep=1)
        
        # Gradient on input image
        self.__grad = tf.gradients(self.__loss, self.__X)
        
    def __add_placeholders(self):
        
        self.__X = tf.placeholder(self.__dtype, [None, 28, 28, 1])
        self.__Y_ = tf.placeholder(self.__dtype, [None, 10])
        self.__tst = tf.placeholder(tf.bool)           # test flag for batch norm
        self.__iter = tf.placeholder(tf.int32)
        self.__pkeep = tf.placeholder(self.__dtype)      # dropout probability
        self.__pkeep_conv = tf.placeholder(self.__dtype)
        
    def __add_variables(self):
        
        # Conv-1 weights
        self.__W1 = tf.Variable(tf.truncated_normal([6, 6, 1, self.__K],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B1 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__K]))
        # Conv-2 weights
        self.__W2 = tf.Variable(tf.truncated_normal([5, 5, self.__K, self.__L],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B2 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__L]))
        # Conv-3 weights
        self.__W3 = tf.Variable(tf.truncated_normal([4, 4, self.__L, self.__M],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B3 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__M]))
        # FC weights
        self.__W4 = tf.Variable(tf.truncated_normal([5 * 5 * self.__M, self.__N],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B4 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__N]))
        # Softmax weights
        self.__W5 = tf.Variable(tf.truncated_normal([self.__N, 10],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B5 = tf.Variable(tf.constant(0.1, self.__dtype, [10]))
        
    def __build_graph(self, X):
        
        # output shape is 28x28
        Y1l = tf.nn.conv2d(X, self.__W1,
                           strides=[1, self.__stride1, self.__stride1, 1],
                           padding='SAME')
        Y1bn, update_ema1 = self.__batchnorm(Y1l, self.__tst, self.__iter,
                                             self.__B1, convolutional=True)
        Y1r = tf.nn.relu(Y1bn)
        Y1 = tf.nn.dropout(Y1r, self.__pkeep_conv,
                           self.__compatible_convolutional_noise_shape(Y1r))
        # output shape is 14x14
        Y2l = tf.nn.conv2d(Y1, self.__W2,
                           strides=[1, self.__stride2, self.__stride2, 1],
                           padding='SAME')
        Y2bn, update_ema2 = self.__batchnorm(Y2l, self.__tst, self.__iter,
                                             self.__B2, convolutional=True)
        Y2r = tf.nn.relu(Y2bn)
        Y2 = tf.nn.dropout(Y2r, self.__pkeep_conv,
                           self.__compatible_convolutional_noise_shape(Y2r))
        # outputshape is 5x5
        Y3l = tf.nn.conv2d(Y2, self.__W3,
                           strides=[1, self.__stride3, self.__stride3, 1],
                           padding='SAME')
        Y3bn, update_ema3 = self.__batchnorm(Y3l, self.__tst, self.__iter,
                                             self.__B3, convolutional=True)
        Y3r = tf.nn.relu(Y3bn)
        Y3 = tf.nn.dropout(Y3r, self.__pkeep_conv,
                           self.__compatible_convolutional_noise_shape(Y3r))
        YY = tf.reshape(Y3, shape=[-1, 5 * 5 * self.__M])
        Y4l = tf.matmul(YY, self.__W4)
        Y4bn, update_ema4 = self.__batchnorm(Y4l, self.__tst,
                                             self.__iter, self.__B4)
        Y4r = tf.nn.relu(Y4bn)
        Y4 = tf.nn.dropout(Y4r, self.__pkeep)
        Ylogits = tf.matmul(Y4, self.__W5) + self.__B5
        Y = tf.nn.softmax(Ylogits)
        update_ema = tf.group(update_ema1, update_ema2,
                              update_ema3, update_ema4)
        
        return Ylogits, Y, update_ema

    def __build_training_graph(self):
        
        loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.__Ylogits,
                                                        labels=self.__Y_)
        loss = tf.reduce_mean(loss_)
        
        correct_prediction = tf.equal(tf.argmax(self.__Y, 1),
                                      tf.argmax(self.__Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.__dtype))
        lr = self.__min_lr + tf.train.exponential_decay(self.__max_lr,
                                                        self.__iter,
                                                        self.__decay_step,
                                                        1/numpy.e)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads_and_vars = opt.compute_gradients(loss=loss,
                                               var_list=self.__trainables)
        grads_and_vars_ = []
        for g, v in grads_and_vars:
            grads_and_vars_.append((tf.clip_by_value(g,
                                                     -self.__grad_clipping,
                                                     self.__grad_clipping), v))
        train_step = opt.apply_gradients(grads_and_vars_)
        return loss, accuracy, train_step
    
    def __build_inference_graph(self):
        
        loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.__Ylogits,
                                                        labels=self.__Y_)
        loss = tf.reduce_mean(loss_)
        
        correct_prediction = tf.equal(tf.argmax(self.__Y, 1),
                                      tf.argmax(self.__Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.__dtype))
        return loss, accuracy, None

    def __batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
        
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def __no_batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
        
        return Ylogits, tf.no_op()
    
    def __compatible_convolutional_noise_shape(self, Y):
        
        noiseshape = tf.shape(Y)
        noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
        return noiseshape
    
    def train_op(self, sess, x, y, iter_, pkeep=1.0, pkeep_conv=1.0):
        
        (_, acc, loss) = sess.run([self.__train_step, self.__accuracy, self.__loss],
                                  feed_dict={self.__X: x,
                                             self.__Y_: y,
                                             self.__iter: iter_,
                                             self.__tst: False,
                                             self.__pkeep: pkeep,
                                             self.__pkeep_conv: pkeep_conv})
        (_) = sess.run([self.__update_ema], feed_dict={self.__X: x,
                                                       self.__Y_: y,
                                                       self.__iter: iter_,
                                                       self.__tst: False,
                                                       self.__pkeep: 1.0,
                                                       self.__pkeep_conv: 1.0})
        return acc, loss
    
    def eval_op(self, sess, x, y):
        
        (acc, loss) = sess.run([self.__accuracy, self.__loss],
                               feed_dict={self.__X: x,
                                          self.__Y_: y,
                                          self.__iter: 0,
                                          self.__tst: True,
                                          self.__pkeep: 1.0,
                                          self.__pkeep_conv: 1.0})
        return acc, loss
    
    def infer_op(self, sess, x):
        
        (y) = sess.run([self.__Y],
                       feed_dict={self.__X: x,
                                  self.__iter: 0,
                                  self.__tst: True,
                                  self.__pkeep: 1.0,
                                  self.__pkeep_conv: 1.0})
        return y
        
    def grad_op(self, sess, x, y):
        
        (grad) = sess.run([self.__grad],
                          feed_dict={self.__X: x,
                                     self.__Y_: y,
                                     self.__iter: 0,
                                     self.__tst: True,
                                     self.__pkeep: 1.0,
                                     self.__pkeep_conv: 1.0})
        return grad
    
    def save(self, sess, path):
        
        self.__saver.save(sess, path)
    
    def restore(self, sess, path):
        
        self.__saver.restore(sess, path)
    
    def get_trainables(self):
        
        return self.__trainables
        

if __name__ == "__main__":
    
    from fmnist_dataset import Fashion_MNIST
    
    with tf.variable_scope("fmnist_cnn") as vs:
        m = CNN(scope_name="fmnist_cnn", dtype="fp64")
    d = Fashion_MNIST()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    x, y = d.train.next_batch(1)
    print (m.eval_op(sess, x, y))
    print (m.infer_op(sess, x))
    print (m.train_op(sess, x, y, 0, 0.9, 1.0))
    print (m.eval_op(sess, x, y))
    print (m.infer_op(sess, x))
    