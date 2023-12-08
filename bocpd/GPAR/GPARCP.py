
import numpy as np

from bocpd.model_cp import GPCPBase, SPCPBase
from bocpd.GPAR.arsplit import ARsplit, exchangeMatrix, differenceMatrixInverse
from scipy.linalg import cholesky, solve_triangular




class ARPredictionCPMixin :
    
    def prior(self):
        mu = 0.0
        sigma2 = self.kernel(self.lagMatrix[0].T) 
        return np.atleast_1d(mu), np.atleast_1d(sigma2[0][0])
    
    def prior_with_grads(self):

        #mu = np.zeros((1,1))
        mu = np.zeros((1))
        dmu = np.zeros((1,len(self.kernel.theta)))

        (sigma2, dsigma2) = self.kernel(self.lagMatrix[0].T, eval_gradient = True)
        dsigma2 = dsigma2[0]
        
        if self.noise_trainable :
            dsigma2 = np.append(dsigma2, 0.0)
            dmu = np.append(dmu, 0.0)
            
        return (mu, np.atleast_2d(dmu)), (sigma2[0], np.atleast_2d(dsigma2))
    
     
    def prediction(self, t):
         
        if self.eval_gradient is False  :
            return self.__prediction(t)
        else :
            return self.__prediction_with_grads(t)

        
    def __prediction(self, t):
        
         if t == 0 :
             
            return self.prior()
        
         MRC = self.MRC(t) #:= min(t, MaxLen)
            
         E = exchangeMatrix(MRC)
         Dinv = differenceMatrixInverse(MRC)

         xt = self.lagMatrix[t]
         yp = E @ self.X[t- MRC : t ]
         xp = np.atleast_2d(E @ self.lagMatrix[t- MRC:t,:,0])
         
         kt = self.kernel(xt.T)
         Kss =  self.kernel(xp)
         Ks = self.kernel( xp, xt.T)
         Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
         
         L =  cholesky(Kss, lower=True, check_finite=False) 
         alpha = solve_triangular(L, yp, lower=True, check_finite=False)
         v = solve_triangular(L, Ks, lower=True, check_finite=False)
         
         # adjust for change point i.e add prior distribution on top of the vector
         mu = np.insert(Dinv @ (v * alpha), 0, 0)
         sigma2 = np.insert(kt - Dinv @ (v * v), 0, kt)
  
         if self.compute_SSE :
            self.SSE_t = np.insert(Dinv @ (alpha * alpha), 0, 0) 
            
         # Extend the mu prediction for the older (> MRC) run length hypothesis
         if MRC < t:  
            mu = np.append(mu, mu[-1] * np.ones(t - MRC))  # t - MRC x 1. [x]
            sigma2 =  np.append(sigma2 , sigma2[-1] * np.ones(t - MRC))

            if self.compute_SSE :
                 self.SSE_t = np.append(self.SSE_t, self.SSE_t[-1] * np.ones(t - MRC))  # t - MRC x 1. [x]

    
         return mu, sigma2

     
 
    def __prediction_with_grads(self, t):
        
        if t == 0 : 
            return self.prior_with_grads()
        
        MRC = self.MRC(t) #:= min(t, MaxLen)

        yp = exchangeMatrix(MRC) @ self.X[t- MRC : t ]
        xt = self.lagMatrix[t]
        
        (K, dK) = self.kernel(exchangeMatrix(MRC + 1) @ self.lagMatrix[t - MRC :t+1, :, 0], eval_gradient = True)
        kt = self.kernel(xt.T)
        Kss = K[1:,1:]
        Ks = np.expand_dims(K[0, 1:],1)
        Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
                
        dKss =  dK[1:, 1:, :]
        dKs = dK[0, 1:, :]
        #dkt = dK[-1,-1]
        dkt = dK[0,0]
        
        if self.noise_trainable :
            dKss = np.append(dKss, 2 * self._noise_scale**2 * np.atleast_3d(np.eye(MRC)), axis=2) #adjust to return grad wrt log(noise)
            dKs = np.hstack((dKs, np.zeros((MRC,1))))   
            dkt = np.append(dkt,0.0)
        
        L =  cholesky(Kss, lower=True, check_finite=False)   
        
        alpha = solve_triangular(L, yp, lower=True, check_finite=False)
        v = solve_triangular(L, Ks, lower=True, check_finite=False)
        
        Dinv = differenceMatrixInverse(MRC)
        mu = Dinv @ (alpha * v)
        sigma2 = kt - Dinv @ (v * v)

        self.L = L
        self.alpha = alpha
        self.v = v

        #compte dL following Sarkka (2013) i.e dL = L Φ(L^{-1} dK L^{-1}.T )
        F = Dinv - 0.5 * np.eye(MRC)
        Linv = solve_triangular(L, np.identity(MRC), lower=True, check_finite=False)

        dL = np.einsum('ijk,jl-> ilk', dKss, Linv.T)
        dL = Linv @ np.moveaxis(dL, -1, 0) 
        dL = np.expand_dims(F,0) * dL
        dL = L @ dL

        #compute dalpha
        #tmp = np.moveaxis((dL @ alpha)[:,:,0], -1, 0) 
        tmp = np.atleast_3d(dL @ alpha)[:,:,0].T
        tmp =  solve_triangular(L, tmp, lower=True, check_finite=False)
        dalpha = - tmp
        
        #compute dv
        Linv_dKs = solve_triangular(L, dKs, lower=True, check_finite=False)
        tmp = np.moveaxis((dL @ v)[:,:,0], -1, 0) 
        dLinv_Ks = - solve_triangular(L, tmp, lower=True, check_finite=False)
        dv = dLinv_Ks  + Linv_dKs
        
        dmu = Dinv @ (alpha * dv + v * dalpha)
        dsigma2 = dkt - 2 * Dinv @ (v * dv)
        
        # adjust for change point i.e add prior distribution on top of the vector
        mu = np.insert(mu, 0, 0)
        sigma2 = np.insert(sigma2, 0, kt)
        
        zero_np = np.zeros((dmu.shape[1]))
        dmu = np.insert(dmu, 0, zero_np, axis = 0)
        dsigma2 = np.insert(dsigma2, 0, dkt, axis = 0)

        #compute_SSE with prior adjustment
        if self.compute_SSE :
            self.SSE_t = np.insert(Dinv @ (alpha * alpha), 0, 0) 
            self.dSSE_t = np.insert(2 * Dinv @ (alpha * dalpha), 0, zero_np, axis = 0)
    
        # Extend the mu prediction for the older (> MRC) run length hypothesis
        if MRC < t:  
            mu = np.append(mu, mu[-1] * np.ones(t - MRC))  # t - MRC x 1. [x]
            sigma2 =  np.append(sigma2 , sigma2 [-1] * np.ones(t - MRC))
            
            dmu = np.concatenate((dmu, np.tile(dmu[-1, :], (t - MRC, 1))))
            dsigma2 = np.concatenate((dsigma2, np.tile(dsigma2[-1, :], (t - MRC, 1))))
            
            if self.compute_SSE :
                 self.SSE_t = np.append(self.SSE_t, self.SSE_t[-1] * np.ones(t - MRC))  # t - MRC x 1. [x]
                 self.dSSE_t = np.concatenate((self.dSSE_t, np.tile(self.dSSE_t[-1, :], (t - MRC, 1))))    

        return (mu, dmu), (sigma2, dsigma2)





class GPARCP(ARPredictionCPMixin, GPCPBase):
    
    def __init__(self, kernel, p = 1, noise_parameter = 0.1):
        super().__init__(kernel,  noise_parameter )
        assert p > 0
        self.p = int(p)
        self._jitter = 0.0
        self.compute_SSE = False

    @property
    def total_noise_variance(self):
        return   self._noise_scale**2 + self._jitter
        
    def precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)
       
        
        

class SPARCP(ARPredictionCPMixin, SPCPBase):
    
    def __init__(self, kernel, p = 1,  prior_parameter = 0.1,  noise_parameter  =0.0):
        super().__init__(kernel, prior_parameter,  noise_parameter )
        assert p > 0
        self.p = int(p)
        
        if self.is_noise :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10
        
        self.compute_SSE = True
        self.SSE_t = 0
        self.dSSE_t = 0
        

    @property
    def total_noise_variance(self):
        return   self._noise_scale**2 + self._jitter
        
    def precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)
        self.out = []


        
   