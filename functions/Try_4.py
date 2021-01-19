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

#Test = np.ones([8,8])


#X1 = np.array([[2,1],[4,4]])
#X = np.pad(X1,(2,2),'constant', constant_values=(0))
#Y = np.array([[1,4,1],[1,4,3],[3,3,1]])
#Y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#X = np.array([[0,50,0,29],[0,80,31,2],[33,90,0,75],[0,9,0,95]])
Y = np.ones([3,3]) * 5
X = np.random.randint(5, size=(28,28))
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
            cachef = flter
    return Op, cachef
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
    return op,locations,pool
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
def b_sc_conv( b_pool_op, image):  
        X = b_pool_op
        Y = image
        #X =  np.rot90(X, k=2)
        k,l=X.shape
        i,j=Y.shape
        f_g = np.zeros((i-k+1,i-k+1))
        for b in range(j-l+1):
            for a in range(i-k+1):
                f_g[b,a] = np.sum(np.multiply(X,Y[b:k+b,a:a+k]),axis=(0,1))
        return f_g
#--------------Anohter way of Conv---------------------------
def Conv_layer(image,fil1):
    image_h,image_w=np.shape(image)
    fil_h,fil_w=np.shape(fil1)
    h=0
    l=np.empty((0,1))
    while(h<=image_h-fil_h):
        w=0
        while(w<=image_w-fil_w):
            dumy=image[h:h+fil_h,w:w+fil_w].flatten()
            dumy=np.array(dumy.reshape(fil_h*fil_w,1))
            l=np.append(l,[dumy])
            w=w+1
        h=h+1 #this+1 is stride
    i=np.reshape(l,[(image_h-fil_h+1)*(image_w-fil_w+1),fil_h*fil_w])
    i=i.T
    fil_1=fil1.flatten()
    fil_1=np.reshape(fil_1,[1,fil_h*fil_w])
    after_conv=np.dot(fil_1,i)
    after_conv=np.reshape(after_conv,[image_h-fil_h+1,image_h-fil_h+1])
    return after_conv
abbas = Conv_layer(X,Y)
talha =  b_sc_conv(Y,X)
