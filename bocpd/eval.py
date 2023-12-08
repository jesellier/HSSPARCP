
import numpy as np
import pandas as pd

import copy
import scipy 

import time
from bocpd.bocpd import BOCPD
from bocpd.model_ts import TS_Optimizer
from bocpd.bocpd_learn import BOCPD_with_gradients, BOCPD_Optimizer
from bocpd.model_base import model_class
from bocpd.UPM.gs_tscp import  TSCP_GridSearch
from bocpd.model_ts import TS_GridSearch


class ParameterSampler():
    
      def __init__(self, X, shift = 1) :

          self.shift = shift
          self.R = np.quantile(X, 0.95)
          self.D = np.median(np.abs(X[1:] - X[:-1]))
          
      def sample(self, n_params, n_restart = 1 ) :
          self.mu0 = 0.5 * np.log(self.R * self.D)
          self.sigma0 =  (np.log(self.R / self.D) / 4)**2
          self.log_params = np.random.normal(self.mu0, self.sigma0, (n_params , n_restart)) +  np.log(self.shift)
          return self
          
      def get_init_params(self, n_params, n_restart = 1) :
         return  self.log_params[ :n_params, n_restart -1 ]
     
      @property
      def parameters(self):
         return np.exp(self.log_params)


class ResultStructNx():

    def __init__(self, evaltuple, n_start = 1, n_size = 1) :
        self.t1 = np.zeros(n_start)
        self.t2 = np.zeros(n_start)
        self.nlml1 = np.zeros(n_start)
        self.nlml2 = np.zeros(n_start)
        
        self.name = evaltuple.model.name
        self.p = evaltuple.model.p
        self.n = evaltuple.model.n_features
        self.optimizer = evaltuple.opt_args['optimizer']
        self.mtype = evaltuple.mtype
        self.hazard_p = evaltuple.hazard_function._hazard_params
        self.maxLen = None
        
        self.Z = np.zeros((n_start, n_size))
        self.M2 = np.zeros((n_start, n_size))
        self.M0 = np.zeros((n_start, n_size))
        
    def resize(self, n_start, n_size) :
        
        self.t1 = np.resize(self.t1, n_start)
        self.t2 = np.resize(self.t2, n_start)
        self.nlml1 = np.resize(self.nlml1, n_start)
        self.nlml2 = np.resize(self.nlml2, n_start)
        
        self.Z = np.resize(self.Z,  (n_start, n_size))
        self.M2 = np.resize(self.M2, (n_start, n_size))
        self.M0 = np.resize(self.M0, (n_start, n_size))
        



class ResultsParser():
    
     def __init__(self) :
         pass

     def _compile(self, results, n_restart = False) :
         
         valid_names = list(results.keys())
         data = {}

         for v in ['mtype', 'p', 'n', 'maxLen', 'optimizer', 'hazard_p', 'n_fail'] :
             data[v] =  [getattr(results[key], v) for key in valid_names]

         for v in ['nlml1', 't1', 'nlml2', 't2'] :
             
             data[v + ".total"] =  [np.mean(getattr(results[key], v)) for key in valid_names]
             
             if n_restart :
                 data[v + ".r_std"] =  [ np.std(getattr(results[key], v)) for key in valid_names]
               
                
         data["nlml2"] =  [ np.mean(results[key].Z) for key in valid_names]
         data["nlml2.err.95"] =  [ 1.96 * np.mean(scipy.stats.sem(results[key].Z.T)) for key in valid_names]
         
         data["mse2"] =  [ np.mean(results[key].M2) for key in valid_names]
         data["mse2.err.95"] =  [ 1.96 * np.mean(scipy.stats.sem(results[key].M2.T)) for key in valid_names]
         
         data["mad"] =  [ np.mean(results[key].M0) for key in valid_names]
         data["mad.err.95"] =  [ 1.96 * np.mean(scipy.stats.sem(results[key].M0.T)) for key in valid_names]
                 
         
         df = pd.DataFrame(data, index = valid_names)  
         
         return df 



class EvaluationHelperGPNx():
    
    def __init__(self, models, X, parameter_sampler = None) :
        self._models = models
        self._X = X

        self._X2 = None
        self.random_initialization = False
        self.results = {}
        self.verbose = True
        self._sampler = parameter_sampler 
        self._log = []
        self.shift = 1
        self.use_maxLen_args = False
        
        
        for model in self._models :

            if not (model.model.mtype == model_class.GPCP or model.model.mtype == model_class.GP)  :
                raise ValueError("EvaluationHelperGPNx only accept GP models")

    
    def initialize(self) :
        
        if self.random_initialization :

            num_trainable_params = [ model.model.num_trainable_params  for model in self._models ]
            max_trainable_params = np.max( num_trainable_params)
            
            if self._sampler is None :
                self._sampler = ParameterSampler(self._X, self.shift )
    
            self._sampler.sample( max_trainable_params, self.n_max_attempt )
            
            

    def run(self, X, n_restart = 2, n_max_attempt = None) :
        
        self.n_restart = n_restart
        self.n_max_attempt = n_max_attempt
        
        (T, D) = X.shape
        
        if self.n_max_attempt is None :
            self.n_max_attempt = 2 * self.n_restart
        
        self.results = {}

        if n_restart > 1 : self.random_initialization = True  
        self.initialize()

        for evaltuple in self._models :
            
            r = ResultStructNx(evaltuple, n_restart, T)
            r.maxLen = evaltuple.opt_args['maxLen_base']

            optimizer = evaltuple.opt_args['optimizer']
            maxiter = evaltuple.opt_args['maxiter']
            tol = evaltuple.opt_args['tol']
            bounds = evaltuple.opt_args['bounds']
            
            if self.use_maxLen_args :
                r.maxLen = evaltuple.opt_args['maxLen_train']
            
            n_succeed = 0
            n_fail = 0

            for n in range(1, self.n_max_attempt + 1) :
          
                if n_succeed == self.n_restart :
                    break
    
                try :
                    model = copy.deepcopy(evaltuple.model)
                    hazard_function = copy.deepcopy(evaltuple.hazard_function)
       
                    if self.random_initialization :
                         init_params = self._sampler.get_init_params(model.num_trainable_params, n)
                         model.set_trainable_params(init_params)

                    #optimize
                    if self.verbose : 
                        print("TRAIN." + model.name + ".n_restart." + str(n_succeed + 1))
                        print("INIT.PARAMS:= {}".format(model.parameters) + "\n" )
      
                    if self.use_maxLen_args :
                        model.maxLen = evaltuple.opt_args['maxLen_train']
          
                    if model.mtype == model_class.GPCP :
                        bocpd_opt = BOCPD_with_gradients(model, hazard_function)
                        opt = BOCPD_Optimizer(self._X, bocpd_opt)
                    elif model.mtype == model_class.GP :
                        opt = TS_Optimizer(self._X, model)
      
                    t0 = time.time()
                    opt.verbose = self.verbose 
                    opt.optimize(optimizer = optimizer, maxiter = maxiter, tol = tol, bounds = bounds)
                    t1 = time.time() - t0
      
                    #evaluate
                    if self.verbose : print("EVAL." + model.name + ".n_restart." + str(n_succeed + 1))
                    
                    if self.use_maxLen_args :
                        model.maxLen = evaltuple.opt_args['maxLen_eval']
                 
                    t0 = time.time()
                    
                    if model.mtype == model_class.GPCP :
                        bocpd_eval = BOCPD(model, bocpd_opt.hazard_function, compute_metrics = True)
                        nlml2, _, (Z, M2, M0) = bocpd_eval.run(X, verbose = self.verbose)
                    elif model.mtype == model_class.GP :
                        nlml2, (Z, M2, M0) = model.run(X, verbose = self.verbose, compute_metrics = True)

                    t2 = time.time() - t0

                    if self.verbose : 
                        print("nlml:= : {}".format(nlml2))
                        print("Finish in :=" + str(t2) + "\n")
                        
                except BaseException as err:
                    msg_id = "model#"  + model.name + " ERROR : stoped and re-attempt"
                    msg_e = f"Unexpected {err=}, {type(err)=}"
                    print(msg_id) 
                    print(msg_e)
                    print("")
                    n_fail +=1
                    self._log.append(msg_id + " : " + msg_e)
                    continue
                else:
                    r.t1[n_succeed] = t1
                    r.t2[n_succeed] = t2
                    
                    r.nlml1[n_succeed] = opt.min_func
                    r.nlml2[n_succeed] = nlml2
                    
                    r.Z[n_succeed, :] = Z[:,0]
                    r.M2[n_succeed, :] = M2[:,0]
                    r.M0[n_succeed, :] = M0[:,0]
                    n_succeed += 1
  
            r.n_fail = n_fail
            r.resize(n_succeed, T)
            self.results[model.name] = r

        n_restart_flag = (n_restart > 1)
        self.df = ResultsParser()._compile(self.results, n_restart_flag )
        
        

class EvaluationHelperTSNx():
    
    def __init__(self, models, X) :
        self._models = models
        self._X = X

        self._X2 = None
        self.results = {}
        self.verbose = True

        for model in self._models :
            if not (model.model.mtype == model_class.TS or model.model.mtype == model_class.TSCP)  :
                raise ValueError("EvaluationHelperGPNx only accept TS models")
  

    def run(self, X) :

        self.results = {}
        
        (T, D) = X.shape
        
        for evaltuple in self._models :
            
            model = copy.deepcopy(evaltuple.model)
            hazard_function = copy.deepcopy(evaltuple.hazard_function)
            r = ResultStructNx(evaltuple, 1, T)
            
            #optimize
            if evaltuple.opt_args['optimizer'] != 'none' :

                if self.verbose : print("TRAIN." + evaltuple.name + "\n")
    
                entry_range =  evaltuple.opt_args['entry_range'] 
                
    
                if model.mtype == model_class.TS :
                    gs = TS_GridSearch(self._X, model)
                elif model.mtype == model_class.TSCP :
                    bocpd_gs = BOCPD(model, hazard_function)
                    gs = TSCP_GridSearch(self._X, bocpd_gs)
      
                t0 = time.time()
                gs.verbose = self.verbose 
                # print(entry_range)
                # gs.set_param_grid(entry_range) 
                # print(len(gs.param_grid))
                gs.search(entry_range)
                t1 = time.time() - t0
                
                r.t1[0] = t1
                r.nlml1[0] = gs.min_func

            #evaluate
            if self.verbose : print("EVAL." + model.name)
            t0 = time.time()

            if model.mtype == model_class.TSCP :
                bocpd_eval = BOCPD(model, hazard_function, compute_metrics = True)
                nlml2, _, (Z, M2, M0) = bocpd_eval.run(X, verbose = self.verbose)
            elif model.mtype == model_class.TS :
                nlml2, (Z, M2, M0) = model.run(X, verbose = self.verbose, compute_metrics = True)
    
            t2 = time.time() - t0
      
            r.n_fail = 0
            r.t2[0] = t2
            r.nlml2[0] = nlml2
                    
            r.Z[0, :] = Z[:,0]
            r.M2[0, :] = M2[:,0]
            r.M0[0, :] = M0[:,0]
        
            if self.verbose : 
                print("nlml:= : {}".format(nlml2))
                print("Finish in :=" + str(t2) + "\n")

            self.results[model.name] = r
            
        self.df = ResultsParser()._compile(self.results, False)
            
            

        

    