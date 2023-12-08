
import numpy as np

from bocpd.test.GP_base_tf import STPBase_TF, GPBase_TF
from bocpd.test.reverseYuleWalkerES_tf import reverseYuleWalkerES_tf

import tensorflow as tf
from gpflow.config import default_float

import tensorflow_probability as tfp
tfd = tfp.distributions





class PrecompMixin_TF():
    
    def precompute(self):
        
        (T, D) =  self.X.shape
        epsilon = 1e-8
        minLen = 500  # Must be at least 2, otherwise indexing errors will result.

        # Precompute the matrix A and vector sigma2 (see thesis Turner algo 14. lalbeled respectively alpha and sigma)
        Kss = self.kernel(self.grid[:1])
        Ks = self.kernel(self.grid[1:], np.zeros((1, 1)))
        A = reverseYuleWalkerES_tf(Kss, Ks, minLen, epsilon) 
        pruneLen = A.shape[0]

        #Precompute sigma2
        sigma2 = Kss - A @ Ks[:pruneLen, :]
        
        # Add in the prior preditictive in the first row, are these memory ineffiecient
        A = tf.experimental.numpy.vstack([tf.zeros((1, pruneLen), dtype= default_float()), A])
        sigma2 = tf.experimental.numpy.vstack([Kss, sigma2])
        
        if not (sigma2.numpy() > 0).all() :
            raise ValueError("sigma2 non strictly positive")
        
        self.A = A
        self.maxLen = tf.constant(pruneLen)
        self.Sig2 = sigma2
        
 

class SPTS_TF(PrecompMixin_TF, STPBase_TF):
    
    def __init__(self, kernel, prior_parameter ):
        super().__init__(kernel, prior_parameter )

        
    def initialize(self):
        self.t = 0
        self.SSE = 2 * self._beta0 # Initialize first SSE to contribution from gamma prior.
        self.precompute()
        

    def tf_prediction(self, t) :
        
        MRC = min(self.maxLen, t)
        
        y = self.X[t - MRC : t]
        xp = self.grid[t-MRC : t]
        xt = tf.expand_dims(self.grid[t],1)
        
        K = self.kernel(xp, xp)
        Kss = self.kernel(xt, xt)
        Ks = self.kernel(xp, xt)
        Kinv = tf.linalg.inv(K)
        
        mu = tf.transpose(Ks) @ Kinv @ y
        sigma2 = Kss - tf.transpose(Ks) @ Kinv @ Ks
        return(mu, sigma2)
    
    def addTrainableNoise(self, noise_parameter ) :
         raise ValueError("cannot add trainable noise on 'SPTS_TF'")
         pass

    
    def mu_t(self, t) :
         MRC = min(self.maxLen, t)
         
         if t == 0 : 
             return 0.0
         
         mu = tf.expand_dims(self.A[ MRC,: MRC],0) @ self.X[ t- MRC :t, :][::-1]
         return mu
     
     
    def sigma2_t(self, t) :
         MRC = min(self.maxLen, t)
         sigma2 = self.Sig2[MRC,0]
         return sigma2
         

    def pdf(self, t):
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        self.t = t
        df = 2 * self._alpha0 + self.MRC(t)
        
        if self.fast_computation is True : 
              self.mu = self.mu_t(t)
              self.sigma2 = self.sigma2_t(t)
        else : 
            self.mu, self.sigma2 = self.tf_prediction(t)

        pred_var = self.sigma2 * self.SSE / df
        pred_scale = tf.math.sqrt(pred_var)
        predprobs = tfp.distributions.StudentT(df, self.mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
    

    def update(self, t):
        self.SSE = self.SSE + (self.mu - self.X[t, 0]) ** 2 / self.sigma2
        
    
    def run(self):

        self.computeGradient(False)
        self.initialize()
        
        (T,D) = self.X.shape
        Z = tf.zeros([1, 1], dtype= default_float())

        for t in range(T) :
            Z = tf.experimental.numpy.vstack([Z, self.logpdf(t)])
            self.update(t)
            
        nlml = - tf.math.reduce_sum(Z, 0)
        return nlml, Z[1:].numpy()
    
    


class GPTS_TF(PrecompMixin_TF, GPBase_TF):
    
    def __init__(self, kernel, noise_parameter  = 0.0):
        super().__init__(kernel, noise_parameter )

        
    def initialize(self):
        self.t = 0
        self.precompute()
        

    def tf_prediction(self, t) :
        
        MRC = min(self.maxLen, t)
        
        y = self.X[t - MRC : t]
        xp = self.grid[t-MRC : t]
        xt = tf.expand_dims(self.grid[t],1)
        
        K = self.kernel(xp, xp)
        Kss = self.kernel(xt, xt)
        Ks = self.kernel(xp, xt)
        Kinv = tf.linalg.inv(K)
        
        mu = tf.transpose(Ks) @ Kinv @ y
        sigma2 = Kss - tf.transpose(Ks) @ Kinv @ Ks
        return(mu, sigma2)
    

    
    def mu_t(self, t) :
         MRC = min(self.maxLen, t)
         
         if t == 0 : 
             return 0.0
         
         mu = tf.expand_dims(self.A[ MRC,: MRC],0) @ self.X[ t- MRC :t, :][::-1]
         return mu
     
     
    def sigma2_t(self, t) :
         MRC = min(self.maxLen, t)
         sigma2 = self.Sig2[MRC,0]
         return sigma2
         

    def pdf(self, t):
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        self.t = t

        if self.fast_computation is True : 
              mu = self.mu_t(t)
              sigma2 = self.sigma2_t(t)
        else : 
            mu, sigma2 = self.tf_prediction(t)

        pred_scale = tf.math.sqrt(sigma2 )
        predprobs = tfp.distributions.Normal(mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
    

    def update(self, t):
        pass
    
    def run(self):

        self.computeGradient(False)
        self.initialize()
        
        (T,D) = self.X.shape
        Z = tf.zeros([1, 1], dtype= default_float())

        for t in range(T) :
            Z = tf.experimental.numpy.vstack([Z, self.logpdf(t)])
            self.update(t)
            
        nlml = - tf.math.reduce_sum(Z, 0)
        return nlml, Z[1:].numpy()
        
        



class SPTSCP_TF(PrecompMixin_TF, STPBase_TF):
    
    def __init__(self, kernel, prior_parameter  = 0.0):
        super().__init__(kernel, prior_parameter )

        
    def initialize(self):
         
        if self.isSet is False :
            raise ValueError("data not set")
        
        self.t = 0
        self.SSE0 = self.SSE = 2 * self._beta0 *  tf.ones((1, 1), dtype= default_float()) # Initialize first SSE to contribution from gamma prior.
        self.precompute()
        
    def addTrainableNoise(self, noise_scale) :
         raise ValueError("cannot add trainable noise on 'SPTSCP_TF'")
         pass

        
    def mu_t(self, t) :
        
         MRC = min(self.maxLen, t)
         mu = self.A[ : MRC + 1,  : MRC] @ self.X[ t- MRC :t, :][::-1]
         
         # Extend the mu prediction for the older (> MRC) run length hypothesis
         if MRC < t:  
            mu = tf.experimental.numpy.vstack([mu , mu[-1] * tf.ones((t - MRC, 1), dtype= default_float())])
     
         return mu
     
        
    def sigma2_t(self, t) :
        
         self.t = t
        
         MRC = min(self.maxLen, t)
         sigma2 = self.Sig2[: t + 1, :]
         
         # Extend the sigma2 prediction for the older (> MRC) run length hypothesis
         if MRC < t:  
            sigma2 = tf.experimental.numpy.vstack([sigma2 , sigma2[-1] * tf.ones((t - MRC, 1), dtype= default_float())])

         return sigma2
         
        

    def pdf(self, t):
        
        self.t = t
        
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        df = 2 * self._alpha0 + tf.constant(self.grid[ : t + 1])
        self.mu = self.mu_t(t)
        self.sigma2 = self.sigma2_t(t)
        
        pred_var = self.sigma2 * self.SSE[:t + 1, :] / df
        pred_scale = tf.math.sqrt(pred_var)
        predprobs = tfp.distributions.StudentT(df, self.mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
 
    
    def update(self, t):
        assert self.t == t
        # update the SSE vector
        y = (self.mu - self.X[t, 0]) ** 2 / self.sigma2
        self.SSE = tf.experimental.numpy.vstack([self.SSE0, self.SSE[ : t + 1, :] + y])






class GPTSCP_TF(PrecompMixin_TF, GPBase_TF):
    
    def __init__(self, kernel, noise_parameter  = 0.0):
        super().__init__(kernel, noise_parameter )

        
    def initialize(self):
         
        if self.isSet is False :
            raise ValueError("data not set")
        
        self.t = 0
        self.precompute()
        
        
        
    def mu_t(self, t) :
        
         MRC = min(self.maxLen, t)
         mu = self.A[ : MRC + 1,  : MRC] @ self.X[ t- MRC :t, :][::-1]
         
         # Extend the mu prediction for the older (> MRC) run length hypothesis
         if MRC < t:  
            mu = tf.experimental.numpy.vstack([mu , mu[-1] * tf.ones((t - MRC, 1), dtype= default_float())])
     
         return mu
     
        
    def sigma2_t(self, t) :
        
         self.t = t
        
         MRC = min(self.maxLen, t)
         sigma2 = self.Sig2[: t + 1, :]
         
         # Extend the sigma2 prediction for the older (> MRC) run length hypothesis
         if MRC < t:  
            sigma2 = tf.experimental.numpy.vstack([sigma2 , sigma2[-1] * tf.ones((t - MRC, 1), dtype= default_float())])

         return sigma2
         
        

    def pdf(self, t):
        
        self.t = t
        
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        mu = self.mu_t(t)
        sigma2 = self.sigma2_t(t)
  
        pred_scale = tf.math.sqrt(sigma2 )
        predprobs = tfp.distributions.Normal(mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
 
    
    def update(self, t):
        pass
    
    
