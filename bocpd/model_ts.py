import numpy as np
from abc import abstractmethod
from sklearn.model_selection import ParameterGrid

from bocpd.model_base import Model, GaussianProcessBase, StudentProcessBase, model_class

from bocpd.Utils.studentpdf import  studentpdf, studentlogpdf
from bocpd.Utils.gaussianpdf import gaussianpdf, gaussianlogpdf

import scipy.optimize
import time
import scipy.stats as ss

from GPTS.rt_minimize import rt_minimize




class Simple_TS_Mixin():
    
    def run(self, X = None, grid = None, verbose = False, compute_metrics = False):
        
        if X is not None :
            self.setData(X, grid)
            
        self.compute_metrics = compute_metrics
        self.computeGradient(False)
        self.initialize()
        
        (T,D) = self.X.shape
        Z = np.zeros((T, 1))
        
        if self.compute_metrics : 
            M0 = np.zeros((T, 1))
            M2 = np.zeros((T, 1))
        
        for t in range(T) :
            Z[t] = np.log(self.pdf(t))
            
            if self.compute_metrics :
                M0[t] = self.abs_error_t
                M2[t] = self.abs_error_t**2
            
            self.update(t)
        
        Z = - Z
        nlml = sum(Z)
        
        if self.compute_metrics :
               return nlml, (Z, M2, M0)
        
        else :
            return nlml, Z
    
    def run_with_gradients(self):
        
        self.computeGradient(True)
        self.initialize()
        
        (T,D) = self.X.shape
        Z = np.zeros((T, 1))
        dZ = np.zeros((T, self.num_trainable_params))
        
        for t in range(T) :
            #print("t:=" + str(t) + "\;")
            (Z[t], dZ[t]) = self.logpdf(t)
            self.update(t)

        nlml = - sum(Z)
        dnlml = - sum(dZ)

        return nlml, dnlml, Z, dZ
    
    
    
class TS_Optimizer():
    
    def __init__(self, X, model, grid = None):

        self.model = model
        self.model.setData(X, grid)
    
        self.min_func = np.inf
        self.min_theta = None
        self.iter = 0
        self.verbose = True
        self.do_callback  = True

        if self.model.mtype != model_class.GP :
            raise ValueError("TS_Optimizer only accept GP models")

        
    def optimize(self, optimizer = "", maxiter = 50, tol = None, bounds = None) :
          
          self.optimizer = optimizer
          self.bounds = bounds
          
          if self.optimizer == 'scipy' :
              return self.__optimize_scipy(maxiter, tol)
          else : 
              return self.__optimize_rt_minimize(maxiter)


    def __optimize_rt_minimize(self, maxiter):
        
        initial_params = self.model.trainable_parameters
        
        def func(theta , obj = self):
            
            t0 = time.time()
            self.model.set_trainable_params(theta)
            out = self.model.run_with_gradients()
    
            if out[0] < self.min_func :
                self.min_func = out[0]
                self.min_theta = theta
               
            if self.verbose :
                print("model_{}".format(self.model.name) + "_iter_{}".format(self.iter))
                #print("tetha:= : {}".format(theta))
                #print("grad:= : {}".format(out[1]))
                #print("params:= : {}".format(self.model.parameters))
                print("func:= : {}".format(out[0]))
                print("time:= : {}".format(time.time() - t0))
                print("")
            
            self.iter +=1
            return out
        
        self.result = rt_minimize(initial_params, func, -maxiter,)
        self.model.set_trainable_params(self.min_theta)
        return self.result #(theta, nlml, i)


    def __optimize_scipy(self, maxiter, tol):
        
        initial_params = self.model.trainable_parameters
        self.t0 = time.time()
        
        def func(theta, obj = self):

            self.model.set_trainable_params(theta)
            out = self.model.run_with_gradients()
            self.fout = out[0]

            if self.fout < self.min_func :
                self.min_func = self.fout
                self.min_theta = theta
       
            return out
        
        def callback(theta, obj = self):

            if self.verbose :
                print("model_{}".format(self.model.name) + "_iter_{}".format(self.iter))
                print("func:= : {}".format(self.fout))
                print("time:= : {}".format(time.time() -  self.t0))
                print("")
            
            self.iter +=1
            self.t0 = time.time()
       
            pass

        #method = "L-BFGS-B"
        method = None
        options =  {"maxiter": maxiter} #, 'disp': True} 
        
        if self.do_callback is True :
            self.result = scipy.optimize.minimize(fun = func, x0 = initial_params, tol = tol, jac=True, callback=callback, method=method, options = options)
        else :
            self.result = scipy.optimize.minimize(fun = func, x0 = initial_params, tol = tol, jac=True, method=method, options = options)
            
        self.model.set_trainable_params(self.min_theta )
        return self.result


        


class TS_GridSearch():
    
    def __init__(self, X, model):
         
         self.X = X
         self.model = model
         self.model.setData(X, None)
         
         self.min_func = np.inf
         self.min_theta = None
         self.verbose = True
         self.iter = 0
         
         if self.model.mtype != model_class.TS :
            raise ValueError("TS_Optimizer only accept TIM models")
            
    
    def set_param_grid(self, entry_range = np.arange(1e-01, 2, 0.5), param_grid_dic = None ) :

        if param_grid_dic is None :
            param_grid_dic = {}
            entry_dic = self.model.trainable_parameters
            
            for key in entry_dic.keys() : 
                param_grid_dic[key] = entry_range
        
        self.param_grid = ParameterGrid(param_grid_dic)

  
    def search(self, entry_range = np.arange(1e-01, 2, 0.5), param_grid_dic = None ) :

        self.set_param_grid(entry_range, param_grid_dic)
        
        for p in self.param_grid :
    
            t0 = time.time()
            self.model.set_trainable_params(p)
            nlml, _ = self.model.run(compute_metrics = False)
            
            if nlml < self.min_func :
                self.min_func = nlml
                self.min_theta = p
                
            if self.verbose :
                self.iter +=1
                #print("model_{}".format(self.bocpd.model.name) + "_iter_{}".format(self.iter))    
                print("model_{}".format(self.model.name) + "_iter_{}".format(self.iter))   
                print("func:= : {}".format(nlml))
                print("time:= : {}".format(time.time() - t0))
                print("")
                
        self.model.set_trainable_params(self.min_theta)

                 

class TIM(Model, Simple_TS_Mixin):
    
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0):
        super().__init__()
        self.alpha0 = alpha
        self.beta0 = beta
        self.kappa0 = kappa
        self.mu0 = mu
        self.mtype = model_class.TS
        self.trainable_entries = ['alpha0', 'beta0']
        
    @property
    def parameters(self):
        return  np.array([self.alpha0, self.beta0, self.mu0, self.kappa0])
    
    @property
    def num_params(self) :
        return 4
    
    @property
    def num_trainable_params(self) :
        return len(self.trainable_parameters)
        
    @property
    def trainable_parameters(self) :
        tmp = {}
        for e in self.trainable_entries :
            tmp[e] = getattr(self, e)
        return tmp

    def set_trainable_params(self, entry_dict):
        for key in entry_dict :
            setattr(self, key, entry_dict[key]) 
        pass
 
    def initialize(self):
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.kappa = self.kappa0
        self.mu = self.mu0


    @property
    def scale2(self):
        return self.beta * (self.kappa + 1) / (self.alpha * self.kappa)

    def pdf(self, t):
        """
        Return the pdf function of the t distribution

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        
        if self.compute_metrics  is True :
            self.abs_error_t = np.abs(self.X[t, 0] - self.mu)
            self.abs_error_2_t = self.abs_error_t **2

        return ss.t.pdf(
            x= self.X[t],
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.scale2),
        )
    
    
    def logpdf(self, t):
        return np.log(self.pdf(t))
    

    def update(self, t):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """

        self.mu = (self.kappa * self.mu + self.X[t]) / (self.kappa + 1)
        self.kappa = self.kappa + 1.0
        self.alpha = self.alpha + 0.5
        self.beta =  self.beta + self.kappa * ((self.X[t] - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))
        

        
class GPTimeSerieBase(GaussianProcessBase, Simple_TS_Mixin):
    
    def __init__(self, kernel, noise_parameter = 0.1):
        super().__init__(kernel, noise_parameter)
        self.mtype = model_class.GP
        
    
    @property
    def _noise_scale(self):
        return self._noise_parameter
    
    
    @abstractmethod
    def precompute(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def prediction(self, t):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )
        
    def initialize(self):
        
        if self.isSet is False :
            raise ValueError("data not set")
        
        
        self.t = 0
        self.precompute()
        

    def pdf(self, t):

        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        mu, sigma2 = self.prediction(t)
        predvar = sigma2 + self._noise_scale ** 2
        predprobs = gaussianpdf(self.X[t, 0], mu, predvar)
        
        if self.compute_metrics  is True :
            self.abs_error_t = np.abs(self.X[t, 0] - mu)
            self.abs_error_2_t = self.abs_error_t **2
        
        self.t = t
        
        return predprobs

    
    def logpdf(self, t):
 
        if self.eval_gradient  :
            (mu, dmu), (sigma2, dsigma2) = self.prediction(t)
            predvar = sigma2 + self._noise_scale ** 2
            dpredvar = dsigma2
            
            if self.noise_trainable :
                #adjust to return grad wrt log(noise)
                dpredvar[-1] += 2 * self._noise_scale ** 2 
            
            (logpredprobs, dlogpredprobs) = gaussianlogpdf(self.X[t, 0], np.resize(mu,1), np.resize(predvar,1), 2)
            dlogpredprobs = dmu * dlogpredprobs[:, 0] + dpredvar * dlogpredprobs[:, 1]
            return logpredprobs, dlogpredprobs * self.gradient_factor
        
        else :
            mu, sigma2 = self.prediction(t)
            predvar = sigma2 + self._noise_scale ** 2
            logpredprobs = gaussianlogpdf(self.X[t, 0], mu, predvar)
            return logpredprobs
        
        self.t = t

    def update(self, t):
        pass
  


        

class SPToeplitzTimeSerieBase(StudentProcessBase, Simple_TS_Mixin):
    
    def __init__(self, kernel, prior_parameter):
        super().__init__(kernel, prior_parameter)
        self.noise_trainable = False
        self._noise_parameter = 0.0
        self.mtype = model_class.GP
        
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
        
        
        self.t = -1
        self.SSE = 2 * self._beta0 # Initialize first SSE to contribution from gamma prior.
        
        if self.eval_gradient :
            self.dSSE = np.zeros((self.kernel.n_dims))
        
        self.precompute()
        

    def pdf(self, t):
        
        assert t == self.t + 1 

        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        df = 2 * self.alpha0 + self.MRC(t)

        self.mu, self.sigma2 = self.prediction(t)

        pred_var = self.sigma2 * self.SSE / df
        predprobs = studentpdf(self.X[t, 0], self.mu, pred_var, df, 1)
        self.t = t
        
        if self.compute_metrics  is True :
            self.abs_error_t = np.abs(self.X[t, 0] - self.mu)
            self.abs_error_2_t = self.abs_error_t **2
        
        return predprobs
    
    
    
    def logpdf(self, t):
        
        assert t == self.t + 1 

        df = 2 * self._alpha0 + self.MRC(t)
     
        if self.eval_gradient is False : 
            self.mu, self.sigma2 = self.prediction(t)
            pred_var = self.sigma2 * self.SSE / df
            logpredprobs = studentlogpdf(self.X[t, 0], self.mu, pred_var, df, 1)
            
            return logpredprobs
        
        else : 
            (self.mu, self.dmu), (self.sigma2, self.dsigma2) = self.prediction(t)
            pred_var = self.sigma2 * self.SSE / df
            (logpredprobs, dlogpredprobs) = studentlogpdf(self.X[t, 0], np.resize(self.mu,1), np.resize(pred_var,1), df, 2)
        
            #Compute Gradient
            ddf_a = 2
            dpredvar_a = - ddf_a * self.sigma2 * self.SSE / df ** 2
            dpredvar_theta = (self.dsigma2 * self.SSE + self.sigma2 * self.dSSE) / df
    
            dlogpredprobs_theta = self.dmu * dlogpredprobs[:, 0] + dpredvar_theta * dlogpredprobs[:, 1]
            dlogpredprobs_a = dpredvar_a * dlogpredprobs[:, 1] + ddf_a * dlogpredprobs[:, 2]
            
            #adjust to return grad wrt log(alpha)
            dlogpredprobs_a = self._alpha0 * dlogpredprobs_a  
            dlogpredprobs = np.concatenate((dlogpredprobs_theta, dlogpredprobs_a), axis = 0)
            self.t = t
            
            #Adjust the final grad output to the transformed space i.e. dlogpredprobs  = dlogpredprobs * self.gradient_factor
        return logpredprobs, dlogpredprobs * self.gradient_factor

    
    def update(self, t):
        
        assert t == self.t

        self.SSE = self.SSE + (self.mu - self.X[t, 0]) ** 2 / self.sigma2
  
        if self.eval_gradient is True :
            self.dSSE = self.dSSE + 2 * (self.mu - self.X[t, 0]) / self.sigma2 * self.dmu \
                -((self.mu - self.X[t, 0]) ** 2) / (self.sigma2 ** 2) * self.dsigma2
                
  



class SPTimeSerieBase(StudentProcessBase, Simple_TS_Mixin):
    
    def __init__(self, kernel, prior_parameter, noise_parameter = 0.0):
        super().__init__(kernel, prior_parameter, noise_parameter)
        self.unit_beta = True
        self.mtype = model_class.GP
        
        
    @property
    def _noise_scale(self):
        return self._noise_parameter
    
    @property
    def _beta0(self):
        if self.unit_beta is True :
            return 1.0
        else :
            return self._prior_parameter
    
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
  
        self.t = 0
        self.precompute()
        self.SSE_t = 0 
        
        if self.eval_gradient : 
            self.dSSE_t = 0
        
        

    def pdf(self, t):

        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        df = 2 * self._alpha0 + self.MRC(t)
        mu, sigma2 = self.prediction(t)

        #SSE must be updated with the prediction method above
        SSE = 2 * self._beta0 + self.SSE_t
        pred_var = sigma2 * SSE / df
        predprobs = studentpdf(self.X[t, 0], mu, pred_var, df, 1)
        self.t = t
        
        if self.compute_metrics  is True :
            self.abs_error_t = np.abs(self.X[t, 0] - mu)
            self.abs_error_2_t = self.abs_error_t **2
        
        return predprobs
    
    
    
    def logpdf(self, t):
        
        MRC = self.MRC(t)
        df = 2 * self._alpha0 + MRC

        if self.eval_gradient is False : 
            mu, sigma2 = self.prediction(t)
            
            #SSE_T must be updated with the prediction method above
            SSE = 2 * self._beta0 + self.SSE_t
            predvar = sigma2 * SSE / df
            logpredprobs = studentlogpdf(self.X[t, 0], self.mu, predvar, df, 1)

            return logpredprobs
        
        else : 
            (mu, dmu), (sigma2, dsigma2) = self.prediction(t)
 
            #SSE_t and dSSE_t must be updated within the prediction method above
            SSE = 2 * self._beta0 + self.SSE_t
            predvar = sigma2 * SSE / df
            (logpredprobs, dlogpredprobs) = studentlogpdf(self.X[t, 0], np.resize(mu,1), np.resize(predvar,1), df, 2)
        
            #Compute Gradient
            dSSE = self.dSSE_t
            
            dpredvar_a = - 2 * predvar / df
            if self.unit_beta is False :
                dpredvar_a += 2 * sigma2 / df
            
            dpredvar_theta = (dsigma2 * SSE + sigma2 * dSSE) / df
    
            dlogpredprobs_theta = dmu * dlogpredprobs[:, 0] + dpredvar_theta * dlogpredprobs[:, 1]
            dlogpredprobs_a = dpredvar_a * dlogpredprobs[:, 1] + 2 * dlogpredprobs[:, 2]
            
            #adjust to return grad wrt log(alpha)
            dlogpredprobs_a = self._alpha0 * dlogpredprobs_a 
            dlogpredprobs = np.concatenate((dlogpredprobs_theta, dlogpredprobs_a), axis = 0)
            self.t = t
            
            #Adjust the final grad output to the transformed space i.e. dlogpredprobs  = dlogpredprobs * self.gradient_factor
        return logpredprobs, dlogpredprobs * self.gradient_factor

    
    def update(self, t):
        pass

