
import numpy as np

from bocpd.test.GP_base_tf import STPBase_TF, GPBase_TF
from bocpd.GPAR.arsplit import ARsplit, exchangeMatrix

import tensorflow as tf
from gpflow.config import default_float

import tensorflow_probability as tfp
tfd = tfp.distributions




class SPAR_TF(STPBase_TF):
    
    def __init__(self, kernel, p, prior_parameter, noise_parameter = 0.0):
        super().__init__(kernel, prior_parameter, noise_parameter)
        self._jitter = 1e-10
        self.p = int(p)
        
    def initialize(self):
        self.t = 0
        self.SSE_t = 0
        self._precompute()

    def _precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)
        

    def prior(self):
        tf_zero = tf.zeros((1,1), dtype = default_float())
        mu = tf_zero
        sig2 = self.kernel(tf_zero)
        return mu, sig2

    
    def prediction(self, t):
        
        if t < 1 :
            return self.prior()
        
            
        if self.maxLen is None:
            MRC = t
        else :
            MRC = min(self.maxLen, t)
 
        # xt = self.lagMatrix[t]
        # yp = self.X[ : t ]
        # xp = self.lagMatrix[:t,:,0]
        
        xt = self.lagMatrix[t]
        yp = self.X[ t- MRC : t ]
        xp = self.lagMatrix[t- MRC:t,:,0]
 
        ks = self.kernel(xp, tf.transpose(xt))
        kss = self.kernel(xp)
        kss += (self._jitter + self._noise_scale**2) * tf.eye(kss.shape[0], dtype = default_float() )
 
        L = tf.linalg.cholesky(kss)
        alpha = tf.linalg.triangular_solve(L, yp, lower=True)
        v = tf.linalg.triangular_solve(L, ks, lower=True)
        
        mu = tf.transpose(v) @ alpha
        sigma2 = self.kernel(tf.transpose(xt)) - tf.transpose(v) @ v
        self.SSE_t = tf.transpose(alpha) @ alpha

        return  mu, sigma2
    
    
    def pdf(self, t):
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        self.t = t
        df = 2 * self._alpha0 + self.MRC(t)
        mu, sigma2 = self.prediction(t)

        SSE = 2 * self._beta0 + self.SSE_t
        pred_var = sigma2 * SSE / df
        pred_scale = tf.math.sqrt(pred_var)
        predprobs = tfp.distributions.StudentT(df, mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

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
    
    
    


class GPAR_TF(GPBase_TF):
    
    def __init__(self, kernel, p, noise_parameter = 0.1):
        super().__init__(kernel, noise_parameter )
        self._jitter = 0.0
        self.p = int(p)
        
    def initialize(self):
        self.t = 0
        self._precompute()
        
    def _precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)
        

    def prior(self):
        tf_zero = tf.zeros((1,1), dtype = default_float())
        mu = tf_zero
        sig2 = self.kernel(tf_zero)
        return mu, sig2


    def prediction(self, t):
        
        if t < 1 :
            return self.prior()
        
        if self.maxLen is None:
            MRC = t
        else :
            MRC = min(self.maxLen, t)
            
        xt = self.lagMatrix[t]
        yp = self.X[ t- MRC : t ]
        xp = self.lagMatrix[t- MRC:t,:,0]

        #xt = self.lagMatrix[t]
        #yp = self.X[ : t ]
        #xp = self.lagMatrix[:t,:,0]
 
        ks = self.kernel(xp, tf.transpose(xt))
        kss = self.kernel(xp)
        kss += (self._jitter + self._noise_scale**2) * tf.eye(kss.shape[0], dtype = default_float() )
        
        L = tf.linalg.cholesky(kss)
        alpha = tf.linalg.triangular_solve(L, yp, lower=True)
        v = tf.linalg.triangular_solve(L, ks, lower=True)
        
        mu = tf.transpose(v) @ alpha
        sigma2 = self.kernel(tf.transpose(xt)) - tf.transpose(v) @ v

        return  mu, sigma2


    def pdf(self, t):
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        self.t = t

        mu, sigma2 = self.prediction(t)
        pred_scale = tf.math.sqrt(sigma2 + self._noise_scale**2)
        predprobs = tfp.distributions.Normal(mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
    

    def update(self, t):
        pass

    def run(self):

        self.initialize()
        
        (T,D) = self.X.shape
        Z = tf.zeros([1, 1], dtype= default_float())

        for t in range(T) :
            Z = tf.experimental.numpy.vstack([Z, self.logpdf(t)])
            self.update(t)
            
        nlml = - tf.math.reduce_sum(Z, 0)
        return nlml, Z[1:].numpy()
        



class AR_CP_Prediction_Mixin :

          
    def prior(self):
        tf_zero = tf.zeros((1,1), dtype = default_float())
        mu = tf_zero
        sig2 = self.kernel(tf_zero)
        return mu, sig2
        
    
    def prediction(self, t):
        
        if t == 0 :
            return self.prior()
        
        if self.maxLen is None:
            MRC = t
        else :
            MRC = min(self.maxLen, t)
        
        Dinv = tf.experimental.numpy.tri(MRC, dtype = default_float())
        E = tf.convert_to_tensor(np.atleast_2d(exchangeMatrix(MRC)), dtype = default_float())

        xt = self.lagMatrix[t]
        yp = E @ self.X[t- MRC : t ]
        xp = E @ self.lagMatrix[t- MRC:t,:,0]
 
        ks = self.kernel(xp, tf.transpose(xt))
        kss = self.kernel(xp)
        kss += (self._jitter + self._noise_scale**2) * tf.eye(kss.shape[0], dtype = default_float() )
        
        L = tf.linalg.cholesky(kss)
        alpha = tf.linalg.triangular_solve(L, yp, lower=True)
        v = tf.linalg.triangular_solve(L, ks, lower=True)

        k0 = self.kernel(tf.transpose(xt))
        mu = Dinv @ (v * alpha)
        sigma2 = k0 - Dinv @ (v * v)
        
        self.SSE_t = Dinv @ (alpha * alpha)
        
        # adjust for change point i.e add prior distribution on top of the vector
        zeros_tf = tf.zeros([1,1], dtype = default_float())
        mu = tf.experimental.numpy.vstack([zeros_tf, mu])
        sigma2 = tf.experimental.numpy.vstack([k0, sigma2])
        self.SSE_t = tf.experimental.numpy.vstack([zeros_tf, self.SSE_t])
        
        if MRC < t :
            mu = tf.experimental.numpy.vstack([mu, mu[-1] * tf.ones((t - MRC,1), dtype = default_float() )])
            sigma2 = tf.experimental.numpy.vstack([sigma2, sigma2[-1] * tf.ones((t - MRC,1), dtype = default_float() )])
            self.SSE_t = tf.experimental.numpy.vstack([self.SSE_t, self.SSE_t[-1] * tf.ones((t - MRC,1), dtype = default_float() )])
            

        return  mu, sigma2





class GPARCP_TF(AR_CP_Prediction_Mixin, GPBase_TF):
    
    def __init__(self, kernel, p, noise_parameter ):
        super().__init__(kernel, noise_parameter )
        self._jitter = 0.0
        self.p = int(p)
        
    def initialize(self):
        self.t = 0
        self._precompute()
        
    def _precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)

    def pdf(self, t):
        
        self.t = t
        
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        mu, sigma2 = self.prediction(t)
        pred_scale = tf.math.sqrt(sigma2 + self._noise_scale**2)
        predprobs = tfp.distributions.Normal(mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
 
    
    def update(self, t):
        pass



class SPARCP_TF(AR_CP_Prediction_Mixin, STPBase_TF):
    
    def __init__(self, kernel, p, prior_parameter , noise_parameter  = 0.0):
        super().__init__(kernel, prior_parameter , noise_parameter )
        self.p = int(p)
        
        if noise_parameter  != 0.0 :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10

        
    def initialize(self):
         
        if self.isSet is False :
            raise ValueError("data not set")
        
        self.t = 0
        self._precompute()
        
    
    def _precompute(self):
        self.SSE_t = 0
        self.lagMatrix = ARsplit(self.X, self.p)


    def pdf(self, t):
        
        self.t = t
        
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        t_alpha = tf.Variable(self.grid[ : t + 1], dtype = default_float() )
        df = 2 * self._alpha0 + t_alpha
        mu, sigma2 = self.prediction(t)
        SSE = 2 * self._beta0 + self.SSE_t
        pred_var = sigma2 * SSE / df
        pred_scale = tf.math.sqrt(pred_var)
        predprobs = tfp.distributions.StudentT(df, mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
 
    
    def update(self, t):
        assert self.t == t
        pass



