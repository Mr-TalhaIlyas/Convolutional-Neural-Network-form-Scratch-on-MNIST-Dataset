"""
Created on Fri Nov  8 17:56:19 2019

@author: Talha Ilyas
"""

'''
In this script I tried some code for implementing 

    1. convolution
    2. pooling
    3. back convolution
    4. back pooling

for "Multi Channel" inputs like CIFAR 10 by using ony Nnmpy
'''

import numpy as np

Y = np.array([[[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]],[[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]],[[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]]])
X1 = np.array([[[1,0,-1],[1,0,-1],[1,0,-1]] , [[1,0,-1],[1,0,-1],[1,0,-1]] , [[1,0,-1],[1,0,-1],[1,0,-1]]])#Std flter
X2 = np.array([[[1,2,-1],[1,2,-1],[1,2,-1]] , [[1,2,-1],[1,2,-1],[1,2,-1]] , [[1,2,-1],[1,2,-1],[1,2,-1]]])#Sobel flter
X3 = np.array([[[3,0,-3],[10,0,-10],[3,0,-3]] , [[3,0,-3],[10,0,-10],[3,0,-3]] , [[3,0,-3],[10,0,-10],[3,0,-3]]])#Schor flter
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
#------------------------------Back pooling Multi Layer-----------------------------------
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
#----------------multi filter output----------------------- 
c11_op = mc_conv(Y,X1)
c12_op = mc_conv(Y,X2)
c13_op = mc_conv(Y,X3)
Op_c1 = np.stack((c11_op,c12_op,c13_op), axis=0)
locations_c1,Op_c1P = mf_pooling(Op_c1)
BP_c1 = b_mf_pooling(Op_c1, locations_c1, Op_c1P)
print(Op_c1.shape)
print(Op_c1)
print(Op_c1P.shape)
print(Op_c1P)
print(locations_c1.shape)
print(locations_c1)
print(BP_c1.shape)
print(BP_c1)

#------------------------------Back Conv Multi Layer--------------------------------------
