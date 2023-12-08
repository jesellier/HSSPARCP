#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from bocpd.GPTS.reverseYuleWalkerES import reverseYuleWalkerES
from bocpd.GPTS.dreverseYuleWalkerES import dreverseYuleWalkerES
from bocpd.model_ts import SPToeplitzTimeSerieBase, GPTimeSerieBase


class ToeplitzPrecompMixin():
    
    def precompute(self):
        
        (T, D) =  self.X.shape
 
        RYW_epsilon = 1e-8
        RYW_minLen = 50  # Must be at least 2, otherwise indexing errors will result.
        
        if hasattr(self, 'RYW_epsilon') :
            RYW_epsilon = self.RYW_epsilon

        # Precompute the matrix A and vector sigma2 (see thesis Turner algo 14. lalbeled respectively alpha and sigma)
        if self.eval_gradient is False :
              # Precompute the matrix A and vector σ (see thesis Turner algo 14. lalbeled respectively alpha and sigma)
              Kss = self.kernel(self.grid[:1])
              Ks = self.kernel(self.grid[1:], np.zeros((1, 1)))
              A = reverseYuleWalkerES(Kss, Ks, RYW_minLen,  RYW_epsilon ) 
              pruneLen = A.shape[0]
        else :
              (K, dK) = self.kernel(self.grid, eval_gradient = True)
              Kss = np.resize(K[0][0], (1,1))
              Ks = np.expand_dims(K[1:, 0],1)
             
              #dKss => (1, kernel.n_dims)
              dKss = np.expand_dims(dK[0,0,:],0)
              #dKss => (T, kernel.n_dims)
              dKs = dK[1:,0,:]
    
              (A, dA) = dreverseYuleWalkerES(
                  Kss,
                  Ks,
                  dKss,
                  dKs,
                  RYW_minLen,
                  RYW_epsilon 
                  )
    
              pruneLen = A.shape[0]

              self.dKss = dKss
             
              dsigma2 = np.zeros((pruneLen + 1,  self.kernel.n_dims))
              for ii in range(self.kernel.n_dims):
                  dsigma2[1:, ii] = dKss[0, ii] - dA[:, :, ii] @ Ks[:pruneLen, 0] - A @ dKs[:pruneLen, ii]

              dA = np.concatenate((np.zeros((1, pruneLen, self.kernel.n_dims)), dA))
              dsigma2[0, :] = dKss[0, :]

              self.dA = dA
              self.dSig2 = dsigma2
             
        #Precompute sigma2
        sigma2 = Kss - A @ Ks[:pruneLen, 0]
        
        # Add in the prior preditictive in the first row, are these memory ineffiecient
        A = np.concatenate((np.zeros((1, pruneLen)), A), axis=0)
        sigma2 = np.append(Kss, sigma2)
        
        if not (sigma2 > 0).all() :
            raise ValueError("sigma2 non strictly positive")
        
        self.maxLen = pruneLen
        self.A = A
        self.Sig2 = np.atleast_2d(sigma2).T
 
    
 
class ToeplitzPredictionMixin():

    
    def mu_t(self, t) :
        
         MRC = self.MRC(t)
         mu = self.A[ MRC,: MRC] @ self.X[ t- MRC :t, 0][::-1]
 
         if self.eval_gradient is True : 
            dmu = self.dA[MRC, : MRC, :].T @ self.X[ t- MRC :t, 0][::-1]
            return mu, dmu
     
         return mu
     
        
    def sigma2_t(self, t) :
        
         MRC = self.MRC(t)
         sigma2 = self.Sig2[MRC , 0]
         
         if self.eval_gradient is True : 
             dsigma2 = self.dSig2[MRC, :] 
             return sigma2, dsigma2
     
         return sigma2
     
     
    def prediction(self, t):
         
        if self.eval_gradient is False  :
            mu = self.mu_t(t)
            sigma2 = self.sigma2_t(t)
            return (mu, sigma2)
        else :
            mu, dmu = self.mu_t(t)
            sigma2,  dsigma2 = self.sigma2_t(t)
            return (mu, dmu), (sigma2, dsigma2)
        


class GPTS(ToeplitzPrecompMixin, ToeplitzPredictionMixin, GPTimeSerieBase):
    
    def __init__(self, kernel):
        nulle_noise_scale = 0.0
        super().__init__(kernel, nulle_noise_scale)
        self.noise_trainable = False
        self.RYW_epsilon = 1e-8
        

class SPTS(ToeplitzPrecompMixin, ToeplitzPredictionMixin, SPToeplitzTimeSerieBase):
    
    def __init__(self, kernel, prior_parameter):
        super().__init__(kernel, prior_parameter)
        self.RYW_epsilon = 1e-8






