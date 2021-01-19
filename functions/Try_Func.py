# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:46:52 2019

@author: Talha Ilyas
"""
'''
In this script I functions for easy use of following 

    1. convolution
    2. pooling
    3. back convolution
    4. back pooling

'''

import numpy as np

#---------------Single Layer/channel  Conv 1st filter----------------------------------------------------
def sc_conv(image, flter):
    X = flter 
    Y = image
    k,l=X.shape
    i,j=Y.shape
    Op = np.zeros((i-k+1,i-k+1))
    for b in range(j-l+1): 
        for a in range(i-k+1):
            Op[b,a] = np.sum(np.multiply(X,Y[b:k+b,a:a+3]),axis=(0,1))
    return Op, X
#---------------Multi Layer/channel  Conv 1st filter----------------------------------------------------
def mc_conv(image, flter):
    X = flter
    Y = image
    ch_f,k,l=X.shape
    ch_i,i,j=Y.shape
    Op = np.zeros((i-k+1,i-k+1))
    for b in range(j-l+1):
      for a in range(i-k+1):
          Op[b,a] = np.sum(np.multiply(X[:,:,:],Y[:,b:k+b,a:a+3]))    
    return Op
#----------------multi filter output-----------------------  
'''Op_f = np.stack((Op,Op), axis=0)'''
#---------------multi filter pooling---------------------------------
def mf_pooling(Op_f):
    ip = Op_f
    cd,c,d = ip.shape
    e = c-c//2
    pool = np.zeros((cd,e,e))
    locations = np.zeros((cd,e,e))
    for cd in range(cd):
        for g in range(c-e):
            for f in range(c-e):
                pool[cd,g,f] = np.max(ip[cd,g*2:g+g+2,f*2:f+f+2])
                locations[cd,g,f] = np.argmax(ip[cd,g*2:g+g+2,f*2:f+f+2])
    return locations,pool
#---------------Single filter pooling---------------------------------
def sf_pooling(Op):
    op = Op
    c,d = op.shape
    e = c-c//2
    pool = np.zeros((e,e))
    locations = np.zeros((e,e))
    for g in range(c-e):
        for f in range(c-e):
            pool[g,f] = np.max(op[g*2:g+g+2,f*2:f+f+2])
            locations[g,f] = np.argmax(Op[g*2:g+g+2,f*2:f+f+2])
    return locations,pool
#----------------------input to FC_NN----------------------------
'''X1 = pool.flatten()'''#FWD Passs
#***________________________________Backward Pass Started_____________________________________________***
#------------------------Back Pool Single Layer/Channel---------------------------
def b_sf_pooling(before_pooling_shape,locations,pool_values):
    cache1 = pool_values
    cache2 = before_pooling_shape.shape
    cache3 = locations.astype(int)
    b_Op = np.zeros(cache2)
    t,u = cache1.shape
    for j in range(u):
        for i in range(t):
           a = cache3[j,i]
           np.put(b_Op[j*2:j+j+2,i*2:i+i+2],a, cache1[j,i])
    return b_Op
#------------------------Back Conv Single Layer/Channel---------------------------
def b_sc_conv(b_pool_op, image):  
        X = b_pool_op
        Y = image
        #Y = numpy.rot90(image, k=2)
        #X =  numpy.rot90(X, k=2)
        k,l=X.shape
        i,j=Y.shape
        f_g = np.zeros((i-k+1,i-k+1))
        for b in range(j-l+1):
            for a in range(i-k+1):
                f_g[b,a] = np.sum(np.multiply(X,Y[b:k+b,a:a+k]),axis=(0,1))
        return f_g
#------------------------Back Pool Multi Layer/Channel---------------------------
def b_mf_pooling(before_pooling_shape,locations,pool_values):
    cache1 = pool_values
    cache2 = before_pooling_shape.shape
    cache3 = locations.astype(int)
    b_Op = np.zeros(cache2)
    cd,t,u = cache1.shape
    for cd in range(cd):
        for j in range(u):
            for i in range(t):
               a = cache3[cd,j,i]
               np.put(b_Op[cd,j*2:j+j+2,i*2:i+i+2],a, cache1[cd,j,i])
    return b_Op
#-----------------------Back Conv Multi channel--------------------------------
def b_mc_conv(b_pool_op, image):  
        X = b_pool_op
        Y = image
        #Y = numpy.rot90(image, k=2)
        #X =  numpy.rot90(X, k=2)
        ch_f,k,l=X.shape
        ch_i,i,j=Y.shape
        f_g = np.zeros((i-k+1,i-k+1))
        for b in range(j-l+1):
            for a in range(i-k+1):
                f_g[b,a] = np.sum(np.multiply(X[:,:,:],Y[:,b:k+b,a:a+k]))
        return f_g
#---------------One by One Convolution----------------------------------------------------
def onebyone_conv(image, flter):
    X = flter
    Y = image
    ch_f,k,l=X.shape
    ch_i,i,j=Y.shape
    Op = np.zeros((i,j))
    for b in range(j-l+1):
      for a in range(i-k+1):
          Op[b,a] = np.sum(np.multiply(X[:,:,:],Y[:,b:b+1,a:a+1]))
    #Op = Op / ch_f
    return Op
#--------------------Channel wise conv----------------------------------------------------
def ch_conv(self, f_g, pool):
        d,e,f = f_g.shape 
        a,b,c = pool.shape
        f_grad = np.empty((0,b-e+1,c-f+1))
        for i in range(a):
              temp = pool[i,:,:]
              temp = np.array(temp, ndmin=3)
              f_gad = self.b_mc_conv(f_g, temp)
              f_gad = np.array(f_gad, ndmin=3)
              f_grad = np.append(f_grad,f_gad,axis=0)
        return f_grad
















