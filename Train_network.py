

#%% reformat to a good shape for input
#COPY THESE VALUES FROM 'Simulation to image data.py'#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Take input 'image' data from the model, and put it into a classifier
Created on Fri Jan 13 14:59:59 2017

@author: jdavidson
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model

#%%
#Import saved simulation data,

# params
numsims=4
maxsteps=(200*20)
numparticles=100
kmax=25

# make list of filenames for these simulations
rnfiles=[("sim"+str(y)+"-"+str(x)+".rn"+".npy") for y in range(numsims) for x in range(1,maxsteps)]
iofiles=[("sim"+str(y)+"-"+str(x)+".io"+".npy") for y in range(numsims) for x in range(1,maxsteps)]
savedir="/home/jdavidson/Documents/py/Hein2015simdata/"


def LoadBatch(batch_size,startpointer):
    # batch_size is the number of time steps to load, not the number of individual inferences
    x_out=np.empty([numparticles*batch_size,kmax,3])
    s_out=np.empty([numparticles*batch_size])
    y_out=np.empty([numparticles*batch_size,2])
    for i in range(batch_size):
        rn=np.load(savedir+rnfiles[startpointer+i])
        io=np.load(savedir+iofiles[startpointer+i])
        x_out[i*numparticles:(i+1)*numparticles]=rn
        s_out[i*numparticles:(i+1)*numparticles]=io[0,:]
        y_out[i*numparticles:(i+1)*numparticles,:]=np.transpose(io[1:3,:]) # delta s, delta a
    return x_out, s_out, y_out


# create training and validation set indices
num=(maxsteps-1)*numsims
allsteps=list(range(num))
trainsteps=allsteps[:int(num * 0.9)]
valsteps=allsteps[-int(num * 0.1):]
    
#%% initiate session and define loss computation
LOGDIR = './save'
sess = tf.InteractiveSession()
train_vars = tf.trainable_variables()         

def lossdiff(x,y):
    return tf.reduce_mean(tf.square(tf.subtract(x,y)))
    
loss = lossdiff(model.ds_output,model.ds_input) + lossdiff(model.da_output,model.da_input)
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
  
sess.run(tf.initialize_all_variables())

# create a summary to monitor cost tensor
tf.scalar_summary("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()

saver = tf.train.Saver()

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

epochs = 100  #  orignal value is epochs=30
batch_size = 10
val_batch_size = len(valsteps)

# train over the dataset
for epoch in range(epochs):  # this goes through everything in the dataset
  print('epoch ',epoch)
  for i in range(int(len(trainsteps)/batch_size)):
    loadpointer = batch_size*i
    #loadpointer = trainsteps[np.random.randint(len(trainsteps)-batch_size-1)]
    rndata, sdata, ydata = LoadBatch(batch_size,loadpointer)
    optimizer.run(feed_dict={model.rn_input: rndata, model.s_input: sdata, model.ds_input: ydata[:,0], model.da_input: ydata[:,1]})
    if i % 500 == 1:
      rndata, sdata, ydata = LoadBatch(val_batch_size,valsteps[0])
      loss_value = loss.eval(feed_dict={model.rn_input: rndata, model.s_input: sdata, model.ds_input: ydata[:,0], model.da_input: ydata[:,1]})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))
      
#      rndata, sdata, ydata = LoadBatch(val_batch_size,valsteps[np.random.randint(len(valsteps)-val_batch_size)])
      feed = feed_dict={model.rn_input: rndata, model.s_input: sdata, model.ds_input: ydata[:,0], model.da_input: ydata[:,1]}
      validdscalc = model.ds_output.eval(feed_dict=feed)
      validdsreal = ydata[:,0]
      validdacalc = model.da_output.eval(feed_dict=feed)
      validdareal = ydata[:,1]
        
      fig = plt.figure(1)  
      fig.clf()
      xyrange=0.2
      ax = fig.add_subplot(121)
      scatter, = ax.plot(validdsreal,validdscalc,'bo')
      plt.xlabel('Simulated')
      plt.ylabel('Model')
      plt.title('Delta speed')
      ax.set_xlim(-xyrange, xyrange)
      ax.set_ylim(-xyrange, xyrange)
      ax.plot([-1,1],[-1,1])
      ax = fig.add_subplot(122)
      scatter, = ax.plot(validdareal,validdacalc,'bo')
      plt.xlabel('Simulated')
      plt.ylabel('Model')
      plt.title('Delta angle')
      ax.set_xlim(-xyrange, xyrange)
      ax.set_ylim(-xyrange, xyrange)
      ax.plot([-1,1],[-1,1])
      plt.show
      plt.pause(0.001)
     

    # write logs at every iteration
    #summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    #summary_writer.add_summary(summary, epoch * batch_size + i)

#    if i % batch_size == 0:
#      if not os.path.exists(LOGDIR):
#        os.makedirs(LOGDIR)
#      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
#      filename = saver.save(sess, checkpoint_path)
#  print("Model saved in file: %s" % filename)


