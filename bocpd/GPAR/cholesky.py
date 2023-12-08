
import numpy as np
from scipy.linalg import solve_triangular
import copy

def hypot( x,y):
    x = abs(x)
    y = abs(y)
    t = min(x,y)
    x = max(x,y)
    t = t/x
    return x*np.sqrt(1+t*t)

def rypot(x,y):
    x = abs(x)
    y = abs(y)
    t = min(x,y)
    x = max(x,y)
    t = t/x
    return x*np.sqrt(1-t*t)



def cholupdate(L, x):
    
    x = copy.copy(x)
    L = copy.copy(L)
    n = np.size(x)

    for i in range(n):
        #r = np.sqrt(L[i, i]**2 + x[i]**2)
        r = hypot(L[i, i], x[i])
        c = r / L[i,i]
        s = x[i]/ L[i,i]
        L[i,i]= r
  
        L[i+1:, i] = (L[i+1:, i] + s * x[i+1:]) / c
        x[i+1:] = c * x[i+1:] - s * L[i+1:, i]

    return L

def choldowndate(L, x):

    x = copy.copy(x)
    L = copy.copy(L)
    n = np.size(x)

    for i in range(n):
        r = rypot(L[i, i], x[i])
        #r = np.sqrt(max(L[i, i]**2 - x[i]**2, epsilon))
        c = r / L[i,i]
        s = x[i]/ L[i,i]
        L[i,i]= r

        L[i+1:, i] = (L[i+1:, i] - s * x[i+1:]) / c
        x[i+1:] = c * x[i+1:] - s * L[i+1:, i]

    return L


class CholeskyMatrixUpdate():
    
    def __init__(self, L = None, T = None):
        
        self._L = None
        self._t = 0
        
        if L is not None :
            n = L.shape[0]
            self._L = np.zeros((T, T))
            self._L[0:n, 0:n] = L
            self._t = n
            
    def prune(self, t) :
        self._t = min(self._t, t)
        
    @property
    def L(self) :
        return self._L[0: self._t, 0: self._t]
        
    def update(self, t, new_col) :
        #at time step t :
            #=> cholesky L is updated from size t-2 to size t-1
            #=> store a (t-1)x(t-1) L matrix
        
        #if already size (t-1) i.e self.t == t then no update
        if t == self._t : #no_update
            return self.L
        
        #else : do column update
        assert t == new_col.shape[0] == self.L.shape[0] + 1
        
        chol_copy = np.zeros((t, t))
        chol_copy[0:t-1, 0:t-1] = self.L
        chol_copy[t-1, 0:t-1] = solve_triangular(self.L, new_col[:-1], lower=True, check_finite=False)
        chol_copy[t-1, t-1] = np.sqrt(new_col[-1] - chol_copy[t-1, 0:t-1].T @ chol_copy[t-1, 0:t-1])
        self._L[0:t, 0:t] = chol_copy
        self._t +=1
        return chol_copy


    
class CholeskyMatrixDowndate():
    
    def __init__(self, chol = None, T = None):
        
        self._L = None
        self._t = 0
        
        if chol is not None :
            n = chol.shape[0]
            self._L = np.zeros((T, T))
            self._L[0:n, 0:n] = chol
            self._t = n
        
    @property
    def L(self) :
        return self._L[0: self._t, 0: self._t]
    
    def prune(self, t) :
        self._t = min(self._t, t)
        
    def update(self, t, new_col) :
        
        if t == self._t : #no_update
            return self.L

        assert t == new_col.shape[0] == self.L.shape[0] + 1
        
        chol_copy = np.zeros((t, t))
        k= np.sqrt(new_col[0])
        v = new_col[1:]
        
        chol_copy[0, 0] = k
        chol_copy[1:t, 0] = v / k 
        chol_copy[1: t, 1 : t] = choldowndate(self.L, v)
        self._L[0:t, 0:t] = chol_copy
        self._t +=1
        return chol_copy
