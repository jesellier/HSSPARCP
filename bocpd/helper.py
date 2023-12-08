

import numpy as np
from enum import Enum

from bocpd.model_ts import TIM
from bocpd.model_mult import MultiConstructor

from bocpd.UPM.UPM import StudentUPM
from bocpd.GPTS.GPTSCP import GPTSCP
from bocpd.GPTS.GPTS import GPTS
from bocpd.GPAR.GPARCP import GPARCP
from bocpd.GPAR.GPAR import GPAR
from bocpd.RRGPAR.RRGPARCP import RRGPARCP, RRSPARCP
from bocpd.RRGPAR.RRGPAR import RRGPAR, RRSPAR
from bocpd.RRGPAR.gaussian_laplace_reduced_rank import  GaussianLaplaceReducedRankND

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
from bocpd.Hazards.constant import ConstantHazards




class model_type(Enum):
    TIMCP = 1
    GPARCP = 2
    GPTSCP = 3
    RRGPARCP = 4
    RRSPARCP = 5
    
    TIM = 6
    GPAR = 7
    GPTS = 8
    RRGPAR = 9
    RRSPAR = 10
    

def defaultModelNameParser(mtype) :

     if mtype == model_type.GPARCP :
        return 'GPARCP'
     elif mtype == model_type.GPTSCP :
         return 'GPTSCP'
     elif mtype == model_type.RRGPARCP :
         return 'RRGPARCP'
     elif mtype == model_type.RRSPARCP :
         return 'RRSPARCP'
     elif mtype == model_type.TIMCP :
         return 'TIMCP'
     
     elif mtype == model_type.GPAR :
        return 'GPAR'
     elif mtype == model_type.GPTS :
         return 'GPTS'
     elif mtype == model_type.RRGPAR :
         return 'RRGPAR'
     elif mtype == model_type.RRSPAR :
         return 'RRSPAR'
     elif mtype == model_type.TIM :
         return 'TIM'
     else:
        raise NotImplementedError("Unrecognized model name")
        


class EvalTuple() :
    
    def __init__(self, mtype, model, opt_args, hazard_function) :
         self.mtype = mtype
         self.model = model
         self.opt_args = opt_args
         self.hazard_function = hazard_function
         
    @property
    def name(self) :
        return self.model.name


    
class ModelHelper() :
    
      _defaultArgs = dict(
              
              name = "",
              hazard_parameter = 25,
              
              #kernel
              kernel = "RBF",
              p = 1, 
              lengthscale = 1.0,
              variance = 2.0,
              scale_mixture = 0.1,
  
              #parameters
              prior_parameter = 1.0,
              noise_parameter = 1.0,
              noise_trainable =  True,
              prior_trainable =  True,
             
              #other
              process_0_values = False,
              RYW_epsilon = 1e-08,
              alpha = .1,
              beta=1.0,
              kappa = 1.0,
              mu = 0.0,
              horizontal_update = True,

              maxLen = None,
              maxLen_train = None,
              maxLen_eval = None,
              n_features = 12,
              
              #n_dim
              n_dimension = 1,
              shared_params = False,
              
              #opt
              optimizer = 'rt_minimize',
              maxiter = 30,
              tol = None,
              entry_range = np.arange(1e-01, 2, 0.5),
              v_bound = [(np.log(0.1), np.inf)],
              l_bound = [(np.log(0.01), np.inf)],
              p_bound  = [(np.log(0.01), np.inf)]
              )
      

      def __init__(self, args = {}):
          self.resetDefaultArgs(args)
          
    
      def setDefaultArgs(self, args) :
           self._Args = {**self._Args, **args} 
           
      def resetDefaultArgs(self, args) :
           self._Args = {**self._defaultArgs, **args} 


      def get_model(self, mtype, **kwargs):

            model = None
            kwargs = {**self._Args, **kwargs} #merge kwards with default args (with priority to args) 
            
            name =  kwargs['name']
            p =  kwargs['p']
            lengthscale = np.array(p * [kwargs['lengthscale']])
            variance = kwargs['variance']
            
            prior_parameter =  kwargs['prior_parameter']
            noise_parameter =  kwargs['noise_parameter']
            maxLen =  kwargs['maxLen']
            
            noise_trainable =  kwargs['noise_trainable']
            prior_trainable =  kwargs['prior_trainable']
            n_features = kwargs['n_features']
            
            n_dimension = kwargs['n_dimension']
            shared_params = kwargs['shared_params']
 
            ### HAZARDS
            hazard_parameter =  kwargs['hazard_parameter']
            hazard_function = ConstantHazards(hazard_parameter)

            ### KERNEL
            k = None
            kernel_str = kwargs['kernel']

            if mtype == model_type.RRGPARCP or mtype == model_type.RRSPARCP or mtype == model_type.RRGPAR or mtype == model_type.RRSPAR  :
                kernel_str = "GaussianLaplaceReducedRank"

            if kernel_str == "RBF" or kernel_str is None : 
                k = ConstantKernel(variance) * RBF(length_scale=lengthscale )
            elif kernel_str == "RationalQuadratic" : 
                scale_mixture = kwargs['scale_mixture']
                lengthscale = kwargs['lengthscale']
                k = ConstantKernel(variance) * RationalQuadratic(length_scale= lengthscale, alpha= scale_mixture) 
            elif  kernel_str == "GaussianLaplaceReducedRank" :
                L = np.atleast_1d( kwargs['L'])[0]
                k = GaussianLaplaceReducedRankND(variance, lengthscale, n_dimension = p, n_features =  n_features, L = L).fit()
            else:
                raise NotImplementedError("Unrecognized kernel entry:=" + kernel_str)
                
  
            ### MODEL
            opt_args = {}
            
            if mtype == model_type.TIM :
                alpha = kwargs['alpha']
                beta = kwargs['beta']
                kappa = kwargs['kappa']
                mu = kwargs['mu']
                
                opt_args = {'optimizer' : 'grid_search', 'entry_range' : kwargs['entry_range'] }
                if kwargs['optimizer'] == 'none' :  opt_args['optimizer'] = 'none'
                
                model = TIM(alpha=alpha, beta= beta, kappa=kappa, mu=mu)
            
            elif mtype == model_type.TIMCP :
                alpha = kwargs['alpha']
                beta = kwargs['beta']
                kappa = kwargs['kappa']
                mu = kwargs['mu']
                
                opt_args = {'optimizer' : 'grid_search', 'entry_range' : kwargs['entry_range'] }
                if kwargs['optimizer'] == 'none' :  opt_args['optimizer'] = 'none'
                
                model = StudentUPM(alpha=alpha, beta=beta, mu=mu, kappa=kappa)

            elif mtype == model_type.GPARCP :
                prior_trainable = False
                opt_args = {'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = GPARCP(k, p, noise_parameter)
                
            elif mtype == model_type.GPAR :
                prior_trainable = False
                opt_args = {'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = GPAR(k, p, noise_parameter)
           
            elif mtype == model_type.GPTSCP :
                noise_trainable  = False
                prior_trainable = False
                p = 0
                RYW_epsilon = kwargs['RYW_epsilon']
                opt_args = {'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = GPTSCP(k)
                model.RYW_epsilon = RYW_epsilon
                
            elif mtype == model_type.GPTS :
                noise_trainable  = False
                prior_trainable = False
                p = 0
                RYW_epsilon = kwargs['RYW_epsilon']
                opt_args = {'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = GPTS(k)
                model.RYW_epsilon = RYW_epsilon
                
            elif mtype == model_type.RRGPARCP :
                prior_trainable = False
                opt_args = { 'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = RRGPARCP(k, p, noise_parameter)
                model.horizontal_update =  kwargs['horizontal_update']
                
            elif mtype == model_type.RRGPAR :
                prior_trainable = False
                opt_args = { 'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = RRGPAR(k, p, noise_parameter)
                model.horizontal_update =  kwargs['horizontal_update']
                
            elif mtype == model_type.RRSPARCP :
                opt_args = {'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = RRSPARCP(k, p, prior_parameter , noise_parameter)
                model.grid_scale = 0.01
                model.add_noise_scale = True
                model.horizontal_update =  kwargs['horizontal_update']
                
            elif mtype == model_type.RRSPAR :
                opt_args = {'optimizer' :  kwargs['optimizer'],  'maxiter' : kwargs['maxiter'], 'tol' : kwargs['tol'], }
                model = RRSPAR(k, p, prior_parameter , noise_parameter)
                model.grid_scale = 0.01
                model.add_noise_scale = True
                model.horizontal_update =  kwargs['horizontal_update']
        
            else :
                 raise ValueError('method type not recognized')
                 
            mtype_str = defaultModelNameParser(mtype)

            if name == "" :
                name = mtype_str + "." + str(p) + "." +  optimizerStrParser(opt_args['optimizer'])
                
            model.name = name
            
            if maxLen is not None :
                model.maxLen = maxLen
                
            if noise_trainable is True :
                model.noise_trainable  = noise_trainable
            
            if prior_trainable is True :
                model.prior_trainable  = prior_trainable 
                
            model.compute_metrics = True
            model.process_0_values = kwargs['process_0_values']

            ## MULTI-D
            if n_dimension > 1 :
   
                model = MultiConstructor()(model.mtype, shared_params)(model, n_dimension)
     
                if mtype == model_type.RRSPARCP or mtype == model_type.RRSPAR :
                    L = kwargs['L']
                    model.setL(L)
                
                model.name = name
 
    
            ## INSTANTIATE OPT OBJ
            bounds = None
            
            if opt_args['optimizer'] == "" :
                opt_args['optimizer'] = "rt_minimize"
                
            if opt_args['optimizer']  == 'scipy' :
 
                v_bound = kwargs['v_bound']
                l_bound = kwargs['l_bound'] 
                p_bound = kwargs['p_bound'] 
                
                bounds = v_bound + (model.num_kernel_params - 1) * l_bound
                
                num_p = model.num_trainable_params - model.num_kernel_params
                
                if num_p > 0 :
                    bounds +=  num_p * p_bound 

            opt_args = {**opt_args, **{'bounds' : bounds}}
            opt_args = {**opt_args, **{'maxLen_base' :  kwargs['maxLen'], 'maxLen_train' :  kwargs['maxLen_train'], 'maxLen_eval' :  kwargs['maxLen_eval']}}
            
            ## INSTANTIATE EVAL OBJ
            evlt = EvalTuple(mtype_str, model, opt_args, hazard_function)

            return evlt
        

def optimizerStrParser(optimizer):

    if optimizer == 'rt_minimize' :
        return 'm'
    elif optimizer == 'scipy' :
        return 's'
    elif optimizer ==  'grid_search':
        return 'gs'
    elif optimizer == 'none' or optimizer == 'None'  :
        return  'n'
    elif optimizer ==  '' :
        return  ''
    else:
        raise NotImplementedError("Unrecognized optimizer name")  

if __name__ == "__main__" :
    
    m1 = ModelHelper().get_model( mtype = model_type.GPTSCP)
    m2 = ModelHelper().get_model( mtype = model_type.GPTSCP, kernel = 'RationalQuadratic', variance = 2.0, alpha = 0.2)
    m4 = ModelHelper().get_model(mtype = model_type.GPAR, p = 1,  variance = 1.0, lengthscale = 0.5, kernel = "RBF", maxLen = 100)
    m5 = ModelHelper({'l_bound' : [(np.log(0.01), 2.0)]}).get_model(mtype = model_type.RRGPARCP, p = 1, L = 1.5, optimizer = 'scipy')
