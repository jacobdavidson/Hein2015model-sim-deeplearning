#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:43:40 2017

@author: jdavidson
"""

import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
#  initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.constant(0.1,shape=shape)  
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
numpixels=41;

img_input = tf.placeholder(tf.float32,shape=[None, numpixels,numpixels])
#rn_input = tf.placeholder(tf.float32,shape=[None, kmax,3])
s_input = tf.placeholder(tf.float32, shape=[None])
ds_input = tf.placeholder(tf.float32, shape=[None])
da_input = tf.placeholder(tf.float32, shape=[None])

ds_output = tf.placeholder(tf.float32, shape=[None])
da_output = tf.placeholder(tf.float32, shape=[None])

#LAYERS:

#distances
weights1=weight_variable([numpixels,numpixels])
bias1=weight_variable([numpixels,numpixels])
h1=tf.nn.tanh(img_input*weights1+bias1)

weights2=weight_variable([numpixels,numpixels])
bias2=weight_variable([numpixels,numpixels])
bias2_2=weight_variable([numpixels,numpixels])
h2=tf.nn.tanh(h1*weights2+bias2)+bias2_2

#dx calculation:  linear
weights_x=weight_variable([numpixels,numpixels])
weights_y=weight_variable([numpixels,numpixels])
GX=h2*weights_x*img_input #multiple by image input, since its like a mask             
GY=h2*weights_y*img_input #multiple by image input, since its like a mask

# speed processing
sweights1 = weight_variable([1])
sbias1 = bias_variable([1])
GS = tf.square(s_input)*sweights1+sbias1
              
# change in X and Y velocities
weights4 = weight_variable([2])
DVY=tf.reduce_sum(tf.reduce_sum(GY,axis=1),axis=1)
DVX = weights4[0]*(tf.reduce_sum(tf.reduce_sum(GX, axis=1),axis=1)+weights4[1]*GS)

# Speed and Angle calc
ds_output = tf.sqrt(tf.square(s_input+DVX)+tf.square(DVY)) - s_input
da_output = atan2(DVY,s_input+DVX)
  