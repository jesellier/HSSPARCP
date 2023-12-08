
import numpy as np
import scipy.optimize
import time

from bocpd.Utils.rmult import rmult
from bocpd.rt_minimize import rt_minimize
from bocpd.Utils.logsumexp import logsumexp


class BOCPD_Optimizer():
    
      def __init__(self, X, bocpd, grid = None):
            
        self.bocpd = bocpd
        self.bocpd.setData(X, grid)

        self.min_func = np.inf
        self.min_theta = None
        self.iter = 0
        self.verbose = False
        self.do_callback = True
        

      def optimize(self, optimizer = "", maxiter = 30, tol = None, bounds = None) :
          
          self.bounds = bounds
          self.optimizer  = optimizer 

          if self.optimizer  == 'scipy' :
              return self.__optimize_scipy(maxiter, tol)
          else : 
              return self.__optimize_rt_minimize(maxiter)


      def __optimize_rt_minimize(self, maxiter):
        
        initial_params = self.bocpd.trainable_parameters

        def func(theta, obj = self):
            
            t0 = time.time()
            self.bocpd.set_trainable_params(theta)
            out = self.bocpd.run_with_gradients(verbose = self.verbose)

            if out[0] < self.min_func :
                self.min_func = out[0]
                self.min_theta = theta
                
            if self.verbose :
                print("model_{}".format(self.bocpd.model.name) + "_iter_{}".format(self.iter))
                print("func:= : {}".format(out[0]))
                print("time:= : {}".format(time.time() - t0))
                print("")

            self.iter +=1
            return out
        
        
        self.result = rt_minimize(initial_params,func, -maxiter,)
        
        self.last_theta = self.bocpd.trainable_parameters
        self.bocpd.set_trainable_params(self.min_theta )
        return self.result #(theta, nlml, i)
    

      def __optimize_scipy(self, maxiter, tol):
        
            initial_params = self.bocpd.trainable_parameters
            
            if self.bounds is None :
                bounds = [(0, np.inf)] + [(np.log(0.1), np.inf)] + (len(initial_params ) - 1) * [(np.log(0.01), np.inf)]
            else : 
                bounds = [(0, np.inf)] + self.bounds
    
            self.t0 = time.time()

            def func(theta, obj = self):

                self.bocpd.set_trainable_params(theta)
                out = self.bocpd.run_with_gradients(verbose = self.verbose)
                self.fout = out[0]
    
                if self.fout < self.min_func :
                    self.min_func = self.fout
                    self.min_theta = theta
           
                return out
            
            def callback(theta, obj = self):

                if self.verbose :
                    print("model_{}".format(self.bocpd.model.name) + "_iter_{}".format(self.iter))
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
                self.result = scipy.optimize.minimize(fun = func, x0 = initial_params, tol = tol, jac=True, bounds = bounds, callback=callback, method=method, options = options)
            else :
                self.result = scipy.optimize.minimize(fun = func, x0 = initial_params, tol = tol, jac=True, bounds = bounds, method=method, options = options)
                
            self.bocpd.set_trainable_params(self.min_theta )
            return self.result





class BOCPD_with_gradients():

    def __init__(self, model, hazard_function):
            self.model = model
            self.model.computeGradient(True)
            self.hazard_function = hazard_function
            
    @property
    def trainable_parameters(self):
        return  np.append(self.hazard_function.parameters, self.model.trainable_parameters)
    
    @property
    def t(self):
        return self.model.t
    
    @property
    def X(self):
        return self.model.X
        
    @property
    def grid(self):
        return self.model.grid
    
    def set_trainable_params(self, arr):
        self.hazard_function.set_params(arr[:self.hazard_function.num_params])
        self.model.set_trainable_params(arr[self.hazard_function.num_params:])
        
    @property
    def parameters(self):
        return  np.append(self.hazard_function.parameters,  self.model.parameters)
        
    def setData(self, X, grid):
        self.model.setData(X, grid)
        
        
        
    def run_with_gradients(self, verbose = False):
        return self.run(verbose)

    def run(self, verbose = False) :
        
        assert self.model.isSet
        
        (T, D) = self.X.shape

        # Evaluate the hazard function for this interval.
        # H(r) = P(runlength_t = 0|runlength_t-1 = r-1)
        # Pre-computed the hazard in preperation for steps 4 & 5, alg 1, of [RPA]
        # logH = log(H), logmH = log(1-H)

        (logH, logmH, dlogH, dlogmH) = self.hazard_function.log_evaluate(self.grid)
        num_hazard_params = self.hazard_function.num_params
        num_model_params = self.model.num_trainable_params

        logR = np.zeros((T + 1, 1))
        dlogR_h = np.zeros((T + 1, num_hazard_params))
        dlogR_m = np.zeros((T + 1, num_model_params))
        
        # Precompute all the gpr aspects of algorithm.
        
        self.model.initialize()
        
        for t in range(T):
            
            #print("t:=" + str(t) + "\;")
 
            # Evaluate the predictive distribution for the new datum under each of
            # the parameters.  This is the standard thing from Bayesian inference.
            model = self.model
            logpredprobs, dlogpredprobs_m  = model.logpdf(t)
            model.update(t)

            #logMsg = log probability that there *was* a changepoint
            logMsg = logR[0 :t + 1, 0] + logpredprobs + logH[:t + 1, 0]  # t x 1
            dlogMsg_h = dlogR_h[:t + 1, :] + dlogH[:t + 1, :]  # t x num_hazard
            
            #Evaluate the growth log probabilities
            logR[ 1: t + 2, 0] = logR[0 :t + 1, 0] + logpredprobs + logmH[:t + 1, 0]  # t x 1. [P]
            dlogR_h[1: t + 2, :] = dlogR_h[0:t +1, :] + dlogmH[:t + 1, :]  # t x num_hazard
            dlogR_m[1: t + 2, :] = dlogR_m[0:t + 1, :] + dlogpredprobs_m  # t x num_model
            
            #i.e. logR[0,0] = log(S) + max(logMsg) where S = sum(normMsg) and normMsg = exp(logMsg - max(logMsg))
            (logR[0, 0], normMsg, S) = logsumexp(logMsg)  # 1 x 1. [P]
    
            # 1 x num_hazard
            dlogR_h[0, :] = rmult(dlogMsg_h, normMsg).sum(axis=0) / S
            dlogR_m[0, :] = rmult(dlogR_m[1:t + 2, :], normMsg).sum(axis=0) / S
            
        # Get the log marginal likelihood of the data, X(1:end), under the model
        # = P(X_1:T), integrating out all the runlengths. 1 x 1. [log P]
        nlml = -1.0 * logsumexp(logR)[0]
    
        # Do the derivatives of nlml
        normR = np.exp(logR - max(logR))  # T x 1
        dnlml_h = -rmult(dlogR_h, normR).sum(axis=0) / sum(normR)  # 1 x num_hazard
        dnlml_m = -rmult(dlogR_m, normR).sum(axis=0) / sum(normR)  # 1 x num_model

        dnlml = np.concatenate((dnlml_h, dnlml_m), axis = 0)

        return (nlml, dnlml)



    