# -*- coding: utf-8 -*-
"""
This version had a different LoadBatch function, which makes each into an "image".  Then it uses these in the network calculation

@author: jdavidson
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model_fullconn as model
import random


# to make it print the whole numpy array
np.set_printoptions(threshold=np.inf)


#%%
#Import saved simulation data,

# params
numsims=50
maxsteps=(200*20)
numparticles=100
kmax=25
lmax=25

numpixels=41 # make this an odd number!  

# make list of filenames for these simulations
rnfiles=[("sim"+str(y)+"-"+str(x)+".rn"+".npy") for y in range(numsims) for x in range(1,maxsteps)]
iofiles=[("sim"+str(y)+"-"+str(x)+".io"+".npy") for y in range(numsims) for x in range(1,maxsteps)]
savedir="/home/jdavidson/Documents/py/Hein2015simdata/"
# shuffle the filenames
c = list(zip(rnfiles, iofiles))
random.shuffle(c)
rnfiles, iofiles = zip(*c)

def ToImage(rnsingle,numpixels):
    imgarray=np.zeros([numpixels,numpixels])            
    neighbors=rnsingle
    neighbors=neighbors[neighbors[:,2]<2*lmax]   
    img=np.transpose([np.rint(neighbors[:,0]/(2*lmax)*(numpixels-1))+(numpixels-1)/2, 
                              np.rint(neighbors[:,1]/(2*lmax)*(numpixels-1))+(numpixels-1)/2])
    img=img.astype(int)
    imgarray[img[:,0],img[:,1]] = 1     
    return imgarray


def LoadBatch(batch_size,startpointer,numpixels):
    # batch_size is the number of time steps to load, not the number of individual inferences
#    x_out=np.empty([numparticles*batch_size,kmax,3])
    img_out=np.empty([numparticles*batch_size,numpixels,numpixels])
    s_out=np.empty([numparticles*batch_size])
    y_out=np.empty([numparticles*batch_size,2])
    for i in range(batch_size):
        rn=np.load(savedir+rnfiles[startpointer+i])
        imgs=list(map(lambda x: ToImage(x,numpixels),rn))
        io=np.load(savedir+iofiles[startpointer+i])
       # x_out[i*numparticles:(i+1)*numparticles]=rn
        img_out[i*numparticles:(i+1)*numparticles]=imgs
        s_out[i*numparticles:(i+1)*numparticles]=io[0,:]
        y_out[i*numparticles:(i+1)*numparticles,:]=np.transpose(io[1:3,:]) # delta s, delta a
    return img_out, s_out, y_out


# create training and validation set indices
num=(maxsteps-1)*numsims
allsteps=list(range(num))
trainsteps=allsteps[:int(num * 0.995)]
valsteps=allsteps[-int(num * 0.005):]
print(len(valsteps))
    
#%% define loss computation

def lossdiff(x,y):
    return tf.reduce_mean(tf.square(tf.subtract(x,y)))
    
loss = lossdiff(model.ds_output,model.ds_input) + lossdiff(model.da_output,model.da_input)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

  
epochs = 30  #  orignal value is epochs=30
batch_size = 20
val_batch_size = len(valsteps)

#%% initial session and do computations
sess = tf.InteractiveSession()
sess.run(   tf.global_variables_initializer())
#with tf.Session() as session: #Create graph session
 
# train over the dataset
for epoch in range(epochs):  # this goes through everything in the dataset
  for i in range(int(len(trainsteps)/batch_size)):
    loadpointer = batch_size*i
    #loadpointer = trainsteps[np.random.randint(len(trainsteps)-batch_size-1)]
    imgdata, sdata, ydata = LoadBatch(batch_size,loadpointer,numpixels)
    optimizer.run(feed_dict={model.img_input: imgdata, model.s_input: sdata, model.ds_input: ydata[:,0], model.da_input: ydata[:,1]})
    if i % 100 == 1:
      imgdata, sdata, ydata = LoadBatch(val_batch_size,valsteps[0],numpixels)
      feed = feed_dict={model.img_input: imgdata, model.s_input: sdata, model.ds_input: ydata[:,0], model.da_input: ydata[:,1]}
      loss_value = sess.run(loss, feed_dict=feed)          
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))
      
      [validdscalc,validdacalc] = sess.run([model.ds_output,model.da_output],feed_dict=feed)
#          validdscalc = model.ds_output.eval(feed_dict=feed)
#          validdacalc = model.da_output.eval(feed_dict=feed)          

      validdsreal = ydata[:,0]
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
#         

    

session.close()
