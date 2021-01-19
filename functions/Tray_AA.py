'''
In this script I tried some code for implementing 

    1. convolution
    2. back convolution

'''

import numpy as np


X = np.array([[5,-4,0,8],[-10,-2,2,3],[0,-2,-4,-7],[-3,-2,-3,-16]])
Y = np.array([[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]])

cachef = 3 #filter size
Fliped = np.rot90(X,k=2)
k,_=X.shape
i,j=Y.shape
Op = np.zeros((cachef,cachef))
#---------------Back Conv----------------------------------------------------
for b in range(cachef):
    for a in range(cachef):
        Op[b,a] = np.sum(np.multiply(Fliped,Y[b:k+b,a:a+k]),axis=(0,1))
X2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
Op2 = X2 + Op*0.0001

k,l=X2.shape
i,j=Y.shape
Op3 = np.zeros((i-k+1,i-k+1))
#---------------Conv----------------------------------------------------
for b in range(j-l+1):
    for a in range(i-k+1):
        Op3[b,a] = np.sum(np.multiply(Op2,Y[b:k+b,a:a+3]),axis=(0,1))



