
import numpy as np


def exchangeMatrix(p):
    out = np.insert(np.zeros((p-1)), p-1 , 1)
    for c in range(2,p+1):
        out = np.vstack((out, np.insert(np.zeros((p-1)), p-c , 1)))
    return out


def differenceMatrix(n):
    D = np.identity(n)
    for i in range(1,n):
        D[i,i-1] = -1
    return D

def differenceMatrixInverse(n):
    D = np.tri(n)
    return D


def ARsplit(X, p):
    out = np.insert(X[:-1],0,np.zeros((1)))
    
    if p > 1 :
        for c in range(2,p+1):
            out = np.vstack((out, np.insert(X[:-c],0,np.zeros((c)))))
        out = out.T @ exchangeMatrix(p)
        return np.expand_dims(out,2)
    else :
        return np.expand_dims(np.expand_dims(out,1),1)

    