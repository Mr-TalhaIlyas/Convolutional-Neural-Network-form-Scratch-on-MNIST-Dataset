# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:11:50 2019

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
X = np.array([[[1,0,-1],[1,0,-1],[1,0,-1]] , [[1,0,-1],[1,0,-1],[1,0,-1]] , [[1,0,-1],[1,0,-1],[1,0,-1]]])

ch_f,k,l=X.shape
ch_i,i,j=Y.shape
Op1 = np.zeros((i-k+1,i-k+1))
Op2 = np.zeros((i-k+1,i-k+1))
#---------------Multi Layer  Conv 1st filter----------------------------------------------------

for b in range(j-l+1):
  for a in range(i-k+1):
      Op1[b,a] = np.sum(np.multiply(X[:,:,:],Y[:,b:k+b,a:a+3]))
print(Op1)    
#---------------Multi Layer  Conv 2nd filter----------------------------------------------------

for b in range(j-l+1):
  for a in range(i-k+1):
      Op2[b,a] = np.sum(np.multiply(X[:,:,:],Y[:,b:k+b,a:a+3]))
print(Op2) 
#----------------multi filter output-----------------------   
Op_f = np.stack((Op1,Op2), axis=0)
#---------------------Multi filter Pooling---------------------------------------------
cd,c,d = Op_f.shape
e = c-c//2
pool = np.zeros((cd,e,e))
for cd in range(cd):
    for g in range(c-e):
        for f in range(c-e):
            pool[cd,g,f] = np.max(Op_f[cd,g*2:g+g+2,f*2:f+f+2])
print(pool)
#------------------------------Back pooling Multi Layer--------------------------------------


#------------------------------Back Conv Multi Layer--------------------------------------





































#-----------------Softmax-------------------------------------------------
X1 = pool.flatten()
nodes = 4
W = np.random.randn(len(X1) , nodes)
exp = np.exp(np.dot(X1,W))
total = np.sum(exp,axis=0)
SOftmax = exp/total

shiftx = X1 - np.max(X1)
exps = np.exp(np.dot(shiftx,W))
SoftStable = exps / np.sum(exps,axis=0)