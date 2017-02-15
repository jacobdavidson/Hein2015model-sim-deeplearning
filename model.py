#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:43:40 2017

@author: jdavidson
"""

import tensorflow as tf
import numpy as np
import scipy

def weight_variable(shape):
#  initial = tf.truncated_normal(shape, stddev=0.5)
  initial = tf.constant(0.5, shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
  
def atan2(y, x, epsilon=1.0e-12):
    # taken from:  https://github.com/tensorflow/tensorflow/issues/6095
    # Add a small number to all zeros, to avoid division by zero:
    x = tf.select(tf.equal(x, 0.0), x+epsilon, x)
    y = tf.select(tf.equal(y, 0.0), y+epsilon, y)

    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
    return angle  
  
# fixed parameters to use with this, for now  
kmax=25  

rn_input = tf.placeholder(tf.float32,shape=[None, kmax, 3])
s_input = tf.placeholder(tf.float32, shape=[None])
ds_input = tf.placeholder(tf.float32, shape=[None])
da_input = tf.placeholder(tf.float32, shape=[None])

ds_output = tf.placeholder(tf.float32, shape=[None])
da_output = tf.placeholder(tf.float32, shape=[None])

#LAYERS:  rotated neighbors (rn) input
distance_input=rn_input[:,:,2]
mask=tf.nn.sigmoid((-distance_input+100)*100)

# social common
Cr=tf.Variable(100.0)   # 150
Ca=tf.Variable(100.0)    # 75
lr=tf.Variable(7.5)     # 5
la=tf.Variable(7.5)    # 10
social_common=(Cr/lr*tf.exp(-distance_input*mask/lr)-Ca/la*tf.exp(-distance_input*mask/la))/distance_input

# mulitply the tanh by the sigmoid, to activate:
scm=social_common*mask

# form GX and GY
GX=tf.reduce_sum(scm*rn_input[:,:,0], axis=1)
GY=tf.reduce_sum(scm*rn_input[:,:,1], axis=1)

# DVY (easy calc)
Wy=tf.Variable(0.1)  # actual:  0.05
DVY=Wy*GY

# Speed calculation
alpha=tf.Variable(1.0)  # actual:  2
eta=tf.Variable(1.0)  # actual:  1
hspeed=(alpha-eta*tf.square(s_input))

# DVX calc
Wx=tf.Variable(0.1)  # actual:  0.05
DVX = Wx*(GX + hspeed)

# Speed and Angle calc
ds_output = tf.sqrt(tf.square(s_input+DVX)+tf.square(DVY)) - s_input
da_output = atan2(DVY,s_input+DVX)
  