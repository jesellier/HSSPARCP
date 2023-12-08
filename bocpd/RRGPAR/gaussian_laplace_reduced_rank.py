import numpy as np
import numbers


def check_random_state_instance(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
    
    
class GaussianLaplaceReducedRank():
    
    def __init__(self, variance, lengthscale , n_features = 10, L = 5):
        self.n_features = n_features
        self.n_dimension = 1
        
        self._L = L
        self._lengthscale = lengthscale 
        self._variance = variance
        self._is_fitted = False

    def fit(self):
        self._is_fitted = True
        self.sqrt_eigenval = self.j *np.pi/2/self._L  #( np.pi * self.j / (2 * self._L))**2
        self.Λ = self.S(self.sqrt_eigenval)
        self.num_params = len(self.theta)
        return self
        
    @property
    def variance(self) :
     return self._variance
 
    @property
    def lengthscale (self) :
     return self._lengthscale
 
    @property
    def theta(self):
        return np.append( self._variance, self._lengthscale)
    
    @property
    def parameters(self):
        return self.theta
    
    def set_theta(self, params):
        print()
        self._variance = params[0]
        self._lengthscale = params[1]
        
        
    @property
    def j(self) :
     return np.expand_dims(np.array(range(1, self.n_features + 1)),1)[:,0]
 
    @property
    def invΛ(self) :
     return 1 / self.Λ

    def feature(self, X):
        """ Transforms the data X (n_samples, n_dimension) 
        to feature map space Z(X) (n_samples, n_features)"""
        #out = (np.expand_dims(X + self._L, 1) * np.expand_dims(self.j, 0))[:,:,0]
        #features = np.sin(np.pi * out / (2 * self._L)) / np.sqrt(self._L)
        features = self._L**(-1/2)*np.sin(np.kron(self.j.T *np.pi,(X + self._L)/2/self._L))
        return features
 
    
    def __call__(self, X, X2 = None, eval_gradient = False):

        if eval_gradient is True :
            return self.__eval_with_grads(X, X2)

        if X2 is None :
            Z = self.feature(X)
            return (self.Λ * Z)  @ Z.T
        return  (self.Λ * self.feature(X))  @  self.feature(X2).T
    
    
    def __eval_with_grads(self, X, X2 = None):

        _, dΛ  = self.Λ_with_grad()
        
        Z1 = self.feature(X)
        
        if X2 is None :
            n = len(X)
            
            dK = np.zeros((n, n, self.num_params))
            for ii in range(self.num_params) :
                dK[:,:, ii] = (dΛ[ii] * Z1)  @ Z1.T
            
            return (self.Λ * Z1)  @ Z1.T, dK
        
        else :
            Z2 = self.feature(X2)
            
            dK = np.zeros((len(X), len(X2), self.num_params))
            for ii in range(self.num_params) :
                dK[:,:, ii] = (dΛ[ii] * Z1)  @ Z2.T
            
            return (self.Λ * Z1)  @  Z2.T,  dK
    
    
    def S(self, v) :
        tmp =  self._variance * np.sqrt(2 * np.pi) * self._lengthscale * np.exp( - 0.5 *  (v * self._lengthscale)**2)  
        return  tmp 

    
    def Λ_with_grad(self):
        #compute gradient in the logspace (dvariance, dlengthscales)
        dΛ = (self.Λ,  self.Λ * (1 - (self.sqrt_eigenval * self._lengthscale)**2))
        return self.Λ, np.array(dΛ)  
    
    def invΛ_with_grad(self):
        invΛ = self.invΛ
        dinvΛ = (invΛ, invΛ * (1 - (self.sqrt_eigenval * self._lengthscale)**2))
        return invΛ, np.array(dinvΛ)
        




class GaussianLaplaceReducedRankND():
    
    def __init__(self, variance, lengthscales , n_features = 10, n_dimension = 1, L = 5):
        self.n_features = n_features
        self.n_dimension = n_dimension
        
        self._L = L
        self._lengthscales = np.array(lengthscales) 
        
        if self._lengthscales.shape[0] == 1 and  n_dimension > 1 :
            self._lengthscales = np.array(2 * [self._lengthscales[0]] )
            
            
        self._variance = variance
        self._is_fitted = False

    def fit(self):
        self._is_fitted = True
        self.sqrt_eigenval =  self.j *np.pi/2/ self._L #2d
        self.Λ = self.S(self.sqrt_eigenval)
        self.num_params = len(self.theta)
        
        if (self.Λ < 1e-50).any() :
            raise ValueError("Λ is numerically instable")
        
        return self
  
    @property
    def variance(self) :
     return self._variance
 
    @property
    def lengthscales (self) :
     return self._lengthscales 
 
    @property
    def parameters(self):
        return self.theta
 
    @property
    def theta(self):
        return np.insert(self._lengthscales, 0, self._variance)
    
    def set_theta(self, params):
         self._variance = params[0]
         self._lengthscales = params[1:]
        
        
        
    @property
    def j(self) :
        j = np.expand_dims(np.array(range(1, self.n_features + 1)),1)[:,0]

        jnd = np.zeros((self.n_features**self.n_dimension, self.n_dimension))
        for ii in range(self.n_dimension):
            jnd[:,ii] = np.tile(np.repeat(j, self.n_features**(self.n_dimension-1-ii)), self.n_features**(ii)) 
        
        return jnd
 
    @property
    def invΛ(self) :
     return 1 / self.Λ

    def feature(self, X):
        """ Transforms the data X (n_samples, n_dimension) 
        to feature map space Z(X) (n_samples, n_features)"""
        tmp = np.expand_dims(X + self._L, 1) * np.expand_dims(self.sqrt_eigenval , 0)
        features = self._L**(-self.n_dimension/2) * np.prod(np.sin(tmp),2)
        return features
 
    def __call__(self, X, X2 = None, eval_gradient = False):

        if X2 is None :
            Z = self.feature(X)
            return (self.Λ * Z)  @ Z.T
        return  (self.Λ * self.feature(X))  @  self.feature(X2).T
    
    
    def S(self, v) :
        tmp =  np.sqrt(2 * np.pi) * self._lengthscales * np.exp( - 0.5 *  (v * self._lengthscales)**2)  
        return self._variance *  np.prod(tmp, 1)

    
    def Λ_with_grad(self):
        #compute gradient in the logspace (dvariance, dlengthscales)
        #dΛ = (self.Λ,  self.Λ * (1 - self.sqrt_eigenval**2 * self._lengthscales**2))
        dΛ = np.zeros((self.Λ.shape[0], self.num_params))
        dΛ[:,0] = self.Λ
        dΛ[:,1:] = np.expand_dims(self.Λ, 1) * (1 - self.sqrt_eigenval**2 * self._lengthscales**2)

        return self.Λ, dΛ.T  
    
    def invΛ_with_grad(self):
        invΛ = self.invΛ
        #dinvΛ = (invΛ, invΛ * (1 - self.sqrt_eigenval**2 * self._lengthscales**2))
        dinvΛ = np.zeros((invΛ.shape[0], self.num_params))
        dinvΛ[:,0] = invΛ
        dinvΛ[:,1:] = np.expand_dims(invΛ, 1) * (1 - self.sqrt_eigenval**2 * self._lengthscales**2)
        return invΛ, dinvΛ.T
        









if __name__ == '__main__':


    rng = np.random
    
    x = 3*np.pi* rng.uniform(size = [200, 1])
    xt = np.expand_dims(np.linspace(-1,10,128),1)

    mid = (min(x) + max(x))/2
    x = x - mid
    xt = xt - mid 
    
    
    L = 4/3*max(abs(x))  # Boundary, x \in [-Lt,Lt]
    k1 =  GaussianLaplaceReducedRank(variance =2, lengthscale=0.5 , n_features = 15, L = L).fit()
    X = xt
    
    f1 = k1.feature(X)
    K1 = k1(X)
    Λ, dΛ = k1.Λ_with_grad()
    

    #########################################################
    n_dimension = 2
    lengthscales =  np.array([0.5, 0.5])
    X = 3*np.pi* rng.uniform(size = [200, n_dimension])
    
    k2 =  GaussianLaplaceReducedRankND(variance =2, lengthscales=  lengthscales , n_features = 15, n_dimension = n_dimension, L = L).fit()
    #k3 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
    
    f2 = k2.feature(X)
    K2 = k2(X)
    Λ2, dΛ2 = k2.Λ_with_grad()


   



