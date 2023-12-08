import numpy as np
from abc import abstractmethod

from bocpd.model_base import GaussianProcessBase, StudentProcessBase, model_class
from bocpd.Utils.studentpdf import  studentpdf, studentlogpdf
from bocpd.Utils.gaussianpdf import gaussianpdf, gaussianlogpdf
from bocpd.Utils.rmult import rmult


        
class GPCPBase(GaussianProcessBase):
    
    def __init__(self, kernel, noise_parameter = 0.1):
        super().__init__(kernel, noise_parameter)
        self.mtype = model_class.GPCP
        self.MRC_past_0 = False

    @property
    def _noise_scale(self):
        return self._noise_parameter


    @abstractmethod
    def precompute(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def prediction(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )
        

    def initialize(self):
         
        if self.isSet is False :
            raise ValueError("data not set")
            
        self.t = 0
        self.precompute()
        
        
        
    def sigma2_adj_t(self, sigma2 ):
        if self.process_0_values :
            tmp = sigma2
            tmp[tmp <= 0] = 1e-22
            return tmp
        return sigma2
        

    def pdf(self, t):

        assert self.eval_gradient is False

        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        mu, sigma2 = self.prediction(t)
        sigma2 = self.sigma2_adj_t(sigma2)
        
        
        predvar = sigma2 + self._noise_scale ** 2
        predprobs = gaussianpdf(self.X[t, 0], mu, predvar)
        
        if self.MRC_past_0 : 
           MRC = self.MRC(t) 
           if t > MRC : predprobs[MRC+1:] = 0
        
        if self.compute_metrics  is True :
            self.abs_error_t = np.abs(self.X[t, 0] - mu)
            self.abs_error_2_t = self.abs_error_t **2
        
        self.t = t
        
        return predprobs
 
    
    def logpdf(self, t):

        if self.eval_gradient is True :

            # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
            (mu, dmu), (sigma2, dsigma2) = self.prediction(t)
            sigma2 = self.sigma2_adj_t(sigma2)
            
            predvar = sigma2 + self._noise_scale ** 2
            dpredvar = dsigma2
            
            if self.noise_trainable :
                dpredvar[:, -1] += 2 * self._noise_scale ** 2
            
            (logpredprobs, dlogpredprobs) = gaussianlogpdf(self.X[t, 0], mu, predvar, 2)
            dlogpredprobs = rmult(dmu, dlogpredprobs[:, 0]) + rmult(dpredvar[:t + 1, :], dlogpredprobs[:, 1])

            #Adjust the final grad output to the transformed space i.e. dlogpredprobs  = dlogpredprobs * self.gradient_factor
            return logpredprobs,  dlogpredprobs * self.gradient_factor
        else :
            mu = self.mu_t(t)
            sigma2 = self.sigma2_t(t)
            predvar = sigma2 + self._noise_scale ** 2
            predprobs = gaussianlogpdf(self.X[t, 0], mu, predvar, 1)
            return predprobs
        
        self.t = t

    def update(self, t):
        pass
  


        

class SPToeplitzCPBase(StudentProcessBase):
    
    def __init__(self, kernel, prior_parameter):
        super().__init__(kernel, prior_parameter)
        self.noise_trainable = False
        self.prior_trainable = True
        self._noise_parameter = 0.0
        self.mtype = model_class.GPCP
        self.MRC_past_0 = False
        
    @property
    def _noise_scale(self):
        return self._noise_parameter
    
    @property
    def _beta0(self):
        return 1.0
    
    @property
    def _alpha0(self):
        return self._prior_parameter

    
    @abstractmethod
    def precompute(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def prediction(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )
        

    def initialize(self):
         
        if self.isSet is False :
            raise ValueError("data not set")
        
        (T, D) =  self.X.shape

        self.t = -1
        self.SSE = np.zeros((T + 1, D))
        self.SSE[0] = 2 * self._beta0 # Initialize first SSE to contribution from gamma prior.
        
        if self.eval_gradient :
            self.dSSE = np.zeros((T + 1, self.kernel.n_dims))
            
        self.precompute()
        
    

    def pdf(self, t):
        
        assert t == self.t + 1 

        assert self.eval_gradient is False

        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        t_alpha = self.grid[ : t + 1][:,0]
        df = np.asarray([2 * self._alpha0]) + t_alpha
        
        mu, sigma2 = self.prediction(t)

         
        pred_var = sigma2 * self.SSE[:t + 1, 0] / df
        predprobs = studentpdf(self.X[t, 0], mu, pred_var, df, 1)
        
        if self.MRC_past_0 : 
           MRC = self.MRC(t) 
           if t > MRC : predprobs[MRC+1:] = 0
        
        if self.compute_metrics  is True :
             self.abs_error_t = np.abs(self.X[t, 0] - mu)
             self.abs_error_2_t = self.abs_error_t **2

        self.mu = mu
        self.sigma2 = sigma2
        self.t = t
        
        return predprobs
 
    
    def logpdf(self, t):
        
        assert t == self.t + 1 

        if self.eval_gradient is True :
 
            # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
            t_alpha = self.grid[ : t + 1][:,0]
            df = np.asarray([2 * self._alpha0]) + t_alpha
            (mu, dmu), (sigma2, dsigma2) = self.prediction(t)

            pred_var = sigma2 * self.SSE[:t + 1, 0] / df
            (logpredprobs, dlogpredprobs_k) = studentlogpdf(self.X[t, 0], mu, pred_var, df, 2)

            #Compute Gradient
            ddf_a = 2
            dpredvar_a = ddf_a  * np.atleast_2d(- sigma2 * self.SSE[:t + 1, 0] / df ** 2).T
    
            dpredvar = np.zeros((t + 1, self.kernel.n_dims))
            for ii in range(self.kernel.n_dims):
                # Use the product rule. t x 1. [x^2/theta_m]
                dpredvar[:, ii] = (dsigma2[:, ii] * self.SSE[:t + 1, 0] + sigma2 * self.dSSE[:t + 1, ii]) / df
 
            dlogpredprobs = rmult(dmu, dlogpredprobs_k[:, 0]) + rmult(dpredvar[:t + 1, :], dlogpredprobs_k[:, 1])
            dlogpredprobs_a = np.atleast_2d(dpredvar_a[:t + 1, 0] * dlogpredprobs_k[:, 1] + ddf_a * dlogpredprobs_k[:, 2]).T
            dlogpredprobs_a = self._alpha0 * dlogpredprobs_a  #adjust to return grad wrt log(alpha)
            dlogpredprobs = np.concatenate((dlogpredprobs, dlogpredprobs_a), axis = 1)
            
            self.mu = mu
            self.dmu = dmu
            self.sigma2 = sigma2
            self.dsigma2 = dsigma2
            self.t = t
            
            #Adjust the final grad output to the transformed space i.e. dlogpredprobs  = dlogpredprobs * self.gradient_factor
            return logpredprobs,  dlogpredprobs * self.gradient_factor
        
        else :
            self.mu = mu
            self.sigma2 = sigma2
             
            return np.log(self.pdf(t))
    
    
    
    def update(self, t):
        
        assert self.t == t 
        
        # update the SSE vector
        self.SSE[ 1 : t + 2, 0] = self.SSE[ : t + 1, 0] + (self.mu - self.X[t, 0]) ** 2 / self.sigma2
        self.SSE[0, 0] = 2 * self._beta0  # 1 x 1. []
        
        if self.eval_gradient is True :
        
            for ii in range(self.kernel.n_dims):
                self.dSSE[1:t + 2, ii] = self.dSSE[:t+1, ii] + 2 * (self.mu - self.X[t, 0]) \
                    / self.sigma2 * self.dmu[:, ii] + -(self.mu - self.X[t, 0]) ** 2 \
                    / self.sigma2 ** 2 * self.dsigma2[:, ii]
                self.dSSE[0, ii] = 0

    
  

class SPCPBase(StudentProcessBase):
    
    def __init__(self, kernel, prior_parameter, noise_parameter = 0.0):
        super().__init__(kernel, prior_parameter, noise_parameter)
        self.unit_beta = True 
        self.prior_trainable = True
        self.mtype = model_class.GPCP
        
        self.add_noise_scale = False
        self.process_0_values = False
        self.MRC_past_0 = False
        

    @property
    def _beta0(self):
        if self.unit_beta is True :
            return 1.0
        else :
            return self._prior_parameter
    
    @property
    def _alpha0(self):
        return self._prior_parameter
    
    @property
    def _noise_scale(self):
        return self._noise_parameter

    
    @abstractmethod
    def precompute(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def prediction(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )
        

    def initialize(self):
         
        if self.isSet is False :
            raise ValueError("data not set")

        self.t = -1
        self.SSE_t = np.atleast_2d(0)
        
        if self.eval_gradient : 
            self.dSSE_t = np.zeros((1, self.num_trainable_params - 1))
            
        self.precompute()
        
    
    @property
    def SSE_adj_t(self ):
        if self.process_0_values :
            tmp = self.SSE_t
            tmp[tmp < 0] = 1e-05
            return tmp
        return self.SSE_t
    
    
    def sigma2_adj_t(self, sigma2 ):
        if self.process_0_values :
            tmp = sigma2
            tmp[tmp <= 0] = 1e-22
            return tmp
        return sigma2
        

    def pdf(self, t):

        assert self.eval_gradient is False

        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        t_alpha = self.grid[ : t + 1][:,0]
        df = np.asarray([2 * self._alpha0])  + t_alpha

        mu, sigma2 = self.prediction(t)
        sigma2 = self.sigma2_adj_t(sigma2)
        
        SSE = 2 * self._beta0 + self.SSE_adj_t
        predvar = sigma2 * SSE[:t + 1] / df
        
        if self.add_noise_scale :
            predvar += self._noise_scale ** 2

        predprobs = studentpdf(self.X[t, 0], mu, predvar, df, 1)
        
        if self.MRC_past_0 : 
           MRC = self.MRC(t) 
           if t > MRC : predprobs[MRC+1:] = 0
        
        if self.compute_metrics  is True :
             self.abs_error_t = np.abs(self.X[t, 0] - mu)
             self.abs_error_2_t = self.abs_error_t **2

        self.t = t
        
        return predprobs
 
    
    def logpdf(self, t):

        if self.eval_gradient is True :
            
            # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
            t_alpha = self.grid[ : t + 1][:,0]
            df = np.asarray([2 * self._alpha0]) + t_alpha
            (mu, dmu), (sigma2, dsigma2) = self.prediction(t)
            sigma2 = self.sigma2_adj_t(sigma2)
            
            SSE = 2 * self._beta0 + self.SSE_adj_t
            predvar = sigma2 * SSE / df
            
            if self.add_noise_scale :
                predvar += self._noise_scale ** 2
 
            (logpredprobs, dlogpredprobs_k) = studentlogpdf(self.X[t, 0], mu, predvar, df, 2)
            
            #Compute Gradient
            dSSE = self.dSSE_t

            dpredvar = np.zeros((t + 1, dsigma2.shape[1]))
            for ii in range(dsigma2.shape[1]):
                # Use the product rule. t x 1. [x^2/theta_m]
                dpredvar[:, ii] = (dsigma2[:, ii] * SSE + sigma2 * dSSE[:, ii]) / df

            if self.noise_trainable and self.add_noise_scale  :
                dpredvar[:, self._noise_param_idx] += 2 * self._noise_scale ** 2

            dlogpredprobs = rmult(dmu, dlogpredprobs_k[:, 0]) + rmult(dpredvar[:t + 1, :], dlogpredprobs_k[:, 1])
                
            if self.prior_trainable is True :           
                dpredvar_a = - 2 * np.atleast_2d(predvar / df ).T

                if self.unit_beta is False :
                    dpredvar_a += 2 * np.expand_dims(sigma2 / df, 1)

                dlogpredprobs_a = np.atleast_2d(dpredvar_a[:t + 1, 0] * dlogpredprobs_k[:, 1] + 2 * dlogpredprobs_k[:, 2]).T
                dlogpredprobs_a = self._alpha0 * dlogpredprobs_a  #adjust to return grad wrt log(alpha)
                dlogpredprobs = np.concatenate((dlogpredprobs, dlogpredprobs_a), axis = 1)
  
            self.t = t
            
            #Adjust the final grad output to the transformed space i.e. dlogpredprobs  = dlogpredprobs * self.gradient_factor
            return logpredprobs,  dlogpredprobs * self.gradient_factor
        
        else :
            return np.log(self.pdf(t))
    
    
    
    def update(self, t):
        pass
  

