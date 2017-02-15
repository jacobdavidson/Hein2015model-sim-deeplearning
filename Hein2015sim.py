#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:43:47 2017

@author: jdavidson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:43:26 2017

@author: jdavidson
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# to make it print the whole numpy array
np.set_printoptions(threshold=np.inf)

# for output
numsims=1
savedir="/home/jdavidson/Documents/py/Hein2015simdata/"

#%% INITITALIZE PARAMETERS AND STARTING POSITIONS
numparticles=100;
# parameters for social interactions
Cr=150 # weighting of repulsive force
lr=5 # length scale of repulsive force
Ca=75 # weighting of attractive force
la=10 # length scale of attractive force
lmax=25 # max distance to weight nearest neighbors
kmax=25 # max number of nearest neighbors to consider in the calculation

# other parameters
tau = 0.05; # time step spacing
sigma=np.sqrt(tau)*0.0
xsize, ysize = [500, 500]; # size of the simulation box
mass=1;

# preferred speeds
alpha = 2*np.ones([numparticles]) ## 
eta = 1 #damping

# simulation time
numsimsteps=200*20; # multiply by 1/tau, because the first number is amount of time;  keeping this as an integer


#%% Define functions to find particles and define forces
def correctdiff(dd): # Corrects the distance calculation for the periodic box size
    newdd=dd;
    if dd[0]<(-xsize/2): 
        newdd[0]=newdd[0]+xsize
    elif dd[0]>=(xsize/2): 
        newdd[0]=newdd[0]-xsize
    if dd[1]<(-ysize/2): 
        newdd[1]=newdd[1]+ysize
    elif dd[1]>=(ysize/2): 
        newdd[1]=newdd[1]-ysize      
    return newdd

def fixanglerange(angle): # Puts all angles into a range of [-Pi,Pi] 
    if angle>np.pi:
        return -(2*np.pi-angle);
    elif angle<-np.pi:
        return (2*np.pi+angle)
    else:
        return angle
    
def Vlen(vec):  # vector length
    return np.sqrt(vec[0]*vec[0]+vec[1]*vec[1])   
        
def getdist(ptcls):  # calculates the distances and zones.  This uses the above functions
    pos=ptcls[:,0:2];
    angles=ptcls[:,2];
    distdiff=np.empty([numparticles,numparticles,3]);
    for focus in range(numparticles):
        # get difference correct it for periodic boundaries
        diffs=pos[focus]-pos;
        for q in range(numparticles):
            diffs[q]=correctdiff(diffs[q])
        # calculate distances and zones
        dists=list(map(Vlen,diffs))
        # save the results
        distdiff[focus]=np.concatenate((diffs,np.transpose([dists])),axis=1);        
        # returns:  [diff_x, diff_y, dist]
        #           [  0   ,   1   ,   2 ]
    return distdiff

#%% INITIALIZE STARTING POSITIONS AND THEN RUN THE SIMULATION a bunch of times
for simnum in range(numsims):
    
    print(numsimsteps)
    

         

    
    
    #                     0    1      2      3
    # storing things:  [xpos, ypos, theta, speed]
    allparticles = np.empty([numsimsteps, numparticles, 4])
    # keep x positions between [0,xsize], y positions between [0, ysize]
    # keep angles between -Pi and Pi
    
    startparticles = np.transpose([xsize*np.random.rand(numparticles), 
                                   ysize*np.random.rand(numparticles), 
                                   2*np.pi*np.random.rand(numparticles)-np.pi,
                                   alpha
                                   ]);
    allparticles[0]=startparticles;
    
    
    # for visualization
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    scatter, = ax.plot(allparticles[0,:,0],allparticles[0,:,1],'bo')
    ax.set_xlim(0, xsize)
    ax.set_ylim(0, ysize)
    #scatter, = ax.plot(np.array(range(numparticles)),np.zeros(numparticles),'bo')
    #ax.set_xlim(0, numparticles)
    #ax.set_ylim(-1, 1)
    plt.ion()
    
    
    for step in range(1,numsimsteps):
        currentparticles=allparticles[step-1]
        positions=currentparticles[:,0:2]
        angles=currentparticles[:,2] 
        speeds=currentparticles[:,3]
        if step>0:
            vx=speeds*np.cos(angles)
            vy=speeds*np.sin(angles)    
            
        dz=getdist(currentparticles); 
        
        ## calculate new angles and new speeds
        vxchange=np.empty([numparticles])
        vychange=np.empty([numparticles])
        
        allrotatedneighbors=np.zeros([numparticles,kmax,3])
        for q in range(numparticles):
            # get kmax nearest neighbors, then keep ones within lax
            sortindices=np.argsort(dz[q,:,2])
            sortindices=sortindices[1:(kmax+1)] # dont keep the self-particle, which will be first in the list
            neighbors=dz[q,sortindices]
            neighbors=neighbors[neighbors[:,2]<lmax]
            
            # sum over neighbors to get new directions
            # returns:  [diff_x, diff_y, dist]
            #           [  0   ,   1   ,   2 ]        
            social_commonterm=np.array(list(map(lambda x: (Cr/lr*np.exp(-x/lr)-Ca/la*np.exp(-x/la))/x,neighbors[:,2])))
            social_x=sum(social_commonterm*neighbors[:,0])
            social_y=sum(social_commonterm*neighbors[:,1])
            
            # calculate new angles and speed
            vxchange[q] = social_x + (alpha[q]-eta*np.square(speeds[q]))/speeds[q] * vx[q]
            vychange[q] = social_y + (alpha[q]-eta*np.square(speeds[q]))/speeds[q] * vy[q]
            
            # TO OUTPUT for learning
            # rotate to put in the frame of an individual
            to_rotate=-angles[q] # rotate the opposite direction of the focus individual
            if len(neighbors)>0:
                rotationmatrix=np.array([[np.cos(to_rotate),-np.sin(to_rotate)], [np.sin(to_rotate),np.cos(to_rotate)]])
                rotatedneighbors=list(map(lambda x: np.dot(rotationmatrix,x),neighbors[:,0:2]))
                rotatedneighbors=np.concatenate((rotatedneighbors,np.transpose([neighbors[:,2]])),axis=1)
                allrotatedneighbors[q,0:len(rotatedneighbors)]=rotatedneighbors
            # set to large number for distance for other entries
            largedist=10000
            allrotatedneighbors[q,len(neighbors):,2]=largedist
        
        # make angle, speed, and position updates
        vxnew = vx +tau/mass*vxchange
        vynew = vy +tau/mass*vychange
        
        newspeeds=np.sqrt(np.square(vxnew)+np.square(vynew))
        newangles=list(map(fixanglerange,np.arctan2(vynew,vxnew))) + sigma*np.random.randn(numparticles)
        
    #    newspeeds=alpha
        newx = np.mod(positions[:,0] + tau*vxnew,xsize);    
        newy = np.mod(positions[:,1] + tau*vynew,ysize);
        ## update the positions according the new angles
        allparticles[step] = np.transpose([newx, newy, newangles, newspeeds]);    
                
        # save quantities for learning
        #inputs:   speeds,     allrotatedneighbors
        #outputs    vxchange,  vychange
        np.save(savedir+"sim"+str(simnum)+"-"+str(step)+".rn",allrotatedneighbors)
        np.save(savedir+"sim"+str(simnum)+"-"+str(step)+".io",[speeds,newspeeds-speeds,list(map(fixanglerange,newangles-angles))])
        
        # this keeps it a bit more exact, instead of recalculating from angles
        vx=vxnew
        vy=vynew
        
    #     view figure
        if np.mod(step,100)==0:
            print(step)
            scatter.set_xdata(allparticles[step,:,0])
            scatter.set_ydata(allparticles[step,:,1])  
            plt.pause(0.0001)
            plt.show();
            
            
# end of loop for doing multiple simulations

#%% VISUALIZE RESULTS
fig = plt.figure(2)
fig.clf()
ax = fig.add_subplot(111)
scatter, = ax.plot(allparticles[0,:,0],allparticles[0,:,1],'bo')
ax.set_xlim(0, xsize)
ax.set_ylim(0, ysize)


def animate(step):
    scatter.set_xdata(allparticles[step,:,0])
    scatter.set_ydata(allparticles[step,:,1])    
    return scatter

ani = animation.FuncAnimation(fig, animate, np.arange(numsimsteps), interval=1, repeat_delay=1000)
plt.show()

#ani.save('fullsim.mp4')

#%% individual angles
fig=plt.figure(3);
fig.clf()
ax = fig.add_subplot(121)
plt.title('angles')
for focus in range(10):
    plt.plot(allparticles[:,focus,2])
    plt.plot(allparticles[:,focus,2])
ax=fig.add_subplot(122)
plt.title('speeds')
plt.plot(allparticles[:,:,3])
#ax.set_ylim(0, 4)
