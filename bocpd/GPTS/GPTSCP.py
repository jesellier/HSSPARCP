
import numpy as np

from bocpd.model_cp import GPCPBase, SPToeplitzCPBase
from bocpd.GPTS.GPTS import ToeplitzPrecompMixin


class ToeplitzPredictionCPMixin :
    
    
     def prediction(self, t):
         
        if self.eval_gradient is False  :
            mu = self.mu_t(t)
            sigma2 = self.sigma2_t(t)
            return (mu, sigma2)
        else :
            mu, dmu = self.mu_t(t)
            sigma2,  dsigma2 = self.sigma2_t(t)
            return (mu, dmu), (sigma2, dsigma2)
    

     def mu_t(self, t) :
        
         MRC = self.MRC(t) #:= min(t, MaxLen)
         mu = self.A[ : MRC + 1,  : MRC] @ self.X[ t- MRC :t, 0][::-1]
         
         # Extend the mu prediction for the older (> MRC) run length hypothesis
         if MRC < t:  
            #mu = np.append(mu, mu[-1] * np.ones(t + 1 - mu.shape[0]))  # t - MRC x 1. [x]
            mu = np.append(mu, mu[-1] * np.ones(t - MRC))  # t - MRC x 1. [x]
            
         if self.eval_gradient is True : 
             
             dmu = np.zeros((t + 1, self.kernel.n_dims))
             for ii in range(self.kernel.n_dims):
                dmu[: MRC + 1, ii] = self.dA[: MRC + 1, : MRC, ii] @ self.X[ t- MRC :t, 0][::-1]
                
             if MRC < t :
                 dmu = np.concatenate((dmu, [dmu[MRC]] * np.ones((t + 1 - dmu.shape[0], 1))))
            
             return mu, dmu
     
        
         return mu
     
    
     
     def sigma2_t(self, t) :
        
         MRC = self.MRC(t) #:= min(t, MaxLen)
         sigma2 = self.Sig2[: t + 1, 0]
         
         # Extend the sigma2 prediction for the older (> MRC) run length hypothesis
         if MRC < t:  
            #sigma2 = np.append(sigma2, sigma2[-1] * np.ones(t + 1 - sigma2.shape[0])) 
            sigma2 = np.append(sigma2, sigma2[-1] * np.ones(t - MRC)) 
  
         if self.eval_gradient is True : 
             
             dsigma2 = self.dSig2[:t + 1, :] 
             if MRC < t :
                 dsigma2 = np.concatenate((dsigma2, np.tile(dsigma2[-1, :], (t - MRC, 1))))
            
             return sigma2, dsigma2
     
         return sigma2
     
        
      
    
class SPTSCP(ToeplitzPrecompMixin, ToeplitzPredictionCPMixin, SPToeplitzCPBase):
    
    def __init__(self, kernel, prior_parameter):
        super().__init__(kernel, prior_parameter )
        self.RYW_epsilon = 1e-8
        


class GPTSCP(ToeplitzPrecompMixin, ToeplitzPredictionCPMixin, GPCPBase):
    
    def __init__(self, kernel):
        nul_noise_parameter = 0.0 
        super().__init__(kernel, nul_noise_parameter)
        self.noise_trainable = False
        self.RYW_epsilon = 1e-8
