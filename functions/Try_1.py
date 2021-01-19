# -*- coding: utf-8 -*-
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

for "Single Channel" inputs like MNIST by using ony Nnmpy
'''
import numpy as np
#X = np.ones([3,3]) * 5
#Y = np.ones([15,15])
#Y = np.random.randint(5, size=(28,28))

#X = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
#Y = np.array([[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]])

X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Y = np.array([[0,50,0,29],[0,80,31,2],[33,90,0,75],[0,9,0,95]])



k,l=X.shape
i,j=Y.shape
Op = np.zeros((i-k+1,i-k+1))
#---------------Conv----------------------------------------------------
for b in range(j-l+1):
    for a in range(i-k+1):
        Op[b,a] = np.sum(np.multiply(X,Y[b:k+b,a:a+k]),axis=(0,1))
        #print(Op)    
#---------------------Pooling---------------------------------------------
#here we need to save 3 vars. for back pooling conv_image(shape), pool (shape) and location of max pooled values
c,d = Op.shape
cache2 = Op.shape
e = c-c//2
pool = np.zeros((e,e))
z = np.zeros((e,e))
temp = np.zeros((e,e))
for g in range(c-e):
    for f in range(c-e):
        pool[g,f] = np.max(Op[g*2:g+g+2,f*2:f+f+2])
        z[g,f] = np.argmax(Op[g*2:g+g+2,f*2:f+f+2])
cache1 = pool #pool values and shape
cache3 = z.astype(int)  #max pooled locations thats why int32 type
#------------Back Pool-----------------------------------------
b_Op = np.zeros(cache2)
t,u = cache1.shape
for j in range(u):
    for i in range(t):
       a = cache3[j,i]
       np.put(b_Op[j*2:j+j+2,i*2:i+i+2],a, cache1[j,i])
#------------------------Back Conv Single Layer/Channel---------------------------
def b_sc_conv(flter, b_pool_op, image):
    filter_old = flter
    cachef,_ = filter_old.shape
    X = np.rot90(b_pool_op,k=2)
    Y = image
    k,l=X.shape
    i,j=Y.shape
    f_grad = np.zeros((cachef,cachef))
    for b in range(cachef):
        for a in range(cachef):
            f_grad[b,a] = np.sum(np.multiply(X,Y[b:k+b,a:a+k]),axis=(0,1))
    return f_grad
f_g = b_sc_conv(X, X, Y)