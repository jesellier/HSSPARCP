import numpy as np
from bocpd.model_base import  Model, model_class
from bocpd.model_ts import Simple_TS_Mixin
import copy


def merge_tim_params_dics(param_dics) :
    
    out_dic = {}
    
    for dic, ii in zip(param_dics, range(len(param_dics))) :
            for key in dic.keys() : 
                out_dic[key + "_" + str(ii)] = dic[key]
    
    return out_dic

def separate_merged_params_dic(out_dic, n_dimension) :
    keys = list([ k[:-2] for k in out_dic.keys()])
    keys = list(set(keys))
    
    param_dics = n_dimension * [{}]
    
    for dic, i in zip(param_dics, range(len(param_dics))) :
            for key in keys : 
                param_dics[i][key] = out_dic[key + "_" + str(i)]
    
    return param_dics
    



class MultiBase(Model) :
        
    def __init__(self, model, n_dimension):
        super().__init__()

        self.n_dimension = n_dimension
        
        self.isSet = model.isSet
        self.eval_gradient = model.eval_gradient
        self.maxLen = model.maxLen
        self.mtype = model.mtype
         
        self.compute_metrics = model.compute_metrics
        self.grid_scale = model.grid_scale
        self.p = model.p
        self.process_0_values = model.process_0_values
        
        self._models = [copy.deepcopy(model) for _ in range(n_dimension)]
        self.MRC_past_0  = False
        
     
    def update(self, t):
        for m in self._models :
             m.update(t)
             
    def setL(self, L):
        for m, i in zip(self._models, range(len(L))) :
            if m.kernel.__class__.__name__ == "GaussianLaplaceReducedRankND" :
                m.kernel._L = L[i]
                m.kernel.fit()
                
    
    def pdf(self, t):
        
        assert self.eval_gradient is False

        logpredprobs = 0 #np.zeros(t + 1)
        
        for m in self._models :
            predprobs_m = m.pdf(t) 
            logpredprobs = logpredprobs + np.log(predprobs_m)
        
        predprobs = np.exp(logpredprobs)

        if self.MRC_past_0 : 
            MRC = self.MRC(t) 
            if t > MRC : predprobs[MRC+1:] = 0
           
        self.t = t

        return predprobs
 
    
                
    @property
    def grid(self):
        return self._models[0].grid

    @property
    def n_features(self):
        return self._models[0].n_features
         
    def setData(self, data, grid = None):
        self.isSet = True
        self.X = data
        for m, i in zip(self._models, range(self.n_dimension)) :
            m.setData(np.expand_dims(data[:, i], 1), grid)

    def computeGradient(self, eval_gradient = True) :
        self.eval_gradient = eval_gradient
        for m in self._models :
             m.computeGradient(eval_gradient)
             
    @property
    def abs_error_t(self):
        tmp= 0
        for m in self._models :
            tmp += m.abs_error_t

        return tmp
    
    @property
    def abs_error_2_t(self):
        tmp= 0
        for m in self._models :
            tmp += m.abs_error_2_t
        
        return tmp
    
    

        
class MultiSeparateParamsBase(MultiBase):
    
    def __init__(self, model, n_dimension):
        super().__init__(model, n_dimension)
    
    @property
    def trainable_parameters(self) :
        tmp = np.array([m.trainable_parameters for m in self._models])
        return tmp.flatten()
        
    @property
    def parameters(self) :
        tmp = np.array([m.parameters for m in self._models])
        return tmp
    
    @property
    def num_params(self) :
        tmp = 0
        for m in self._models :
          tmp += m.num_params
        return tmp
    
    @property
    def num_trainable_params(self) :
        tmp = 0
        for m in self._models :
          tmp += m.num_trainable_params
        return tmp
        

    def set_trainable_params(self, theta) :
        ii = 0
        for m in self._models :
            num_trainable_m =  m.num_trainable_params
            m.set_trainable_params(theta[ii : num_trainable_m  + ii])
            ii += num_trainable_m 

    def initialize(self):
        for m in self._models :
             m.initialize()
             
        self._static_num_trainable_params = self.num_trainable_params
   


        
class MultiSeparateParamsCPBase(MultiSeparateParamsBase):
    
    def __init__(self, model, n_dimension):
        super().__init__(model, n_dimension)

    def logpdf(self, t):
        
        if self.eval_gradient is False :
            return np.log(self.pdf(t))

        logpredprobs = 0 #np.zeros(t + 1)
        dlogpredprobs = np.zeros((t + 1, self._static_num_trainable_params))
        ii = 0
        
        for m in self._models :
            num_trainable_m =  m.num_trainable_params
            logpredprobs_m, dlogpredprobs[:, ii : num_trainable_m + ii] = m.logpdf(t) 
            logpredprobs = logpredprobs + logpredprobs_m
            ii += num_trainable_m

        self.t = t

        return logpredprobs, dlogpredprobs
    
class MultiSeparateParamsTSBase(MultiSeparateParamsBase):
    
    def __init__(self, model, n_dimension):
        super().__init__(model, n_dimension)

    def logpdf(self, t):
        
        if self.eval_gradient is False :
            return np.log(self.pdf(t))

        logpredprobs = 0 
        dlogpredprobs = np.zeros((self._static_num_trainable_params))
        ii = 0
        
        for m in self._models :
            num_trainable_m =  m.num_trainable_params
            logpredprobs_m, dlogpredprobs[ ii : num_trainable_m + ii] = m.logpdf(t) 
            logpredprobs = logpredprobs + logpredprobs_m
            ii += num_trainable_m

        self.t = t

        return logpredprobs, dlogpredprobs
    
    
        
class MultiSharedParamsBase(MultiBase):
    
    def __init__(self, model, n_dimension):
        super().__init__( model, n_dimension)
        
    @property
    def trainable_parameters(self) :
        return self._models[0].trainable_parameters
        
    @property
    def parameters(self) :
        return self._models[0].parameters
    
    @property
    def num_params(self) :
        return  self._models[0].num_params
    
    @property
    def num_trainable_params(self) :
        return self._models[0].num_trainable_params

    def set_trainable_params(self, theta) :
        for m in self._models :
            m.set_trainable_params(theta)

 
    def initialize(self):
        for m in self._models :
             m.initialize()
   
    
    def logpdf(self, t):
        
        if self.eval_gradient is False :
            return np.log(self.pdf(t))

        logpredprobs = 0 #np.zeros(t + 1)
        dlogpredprobs = 0 #np.zeros((t + 1, self.num_trainable_params))
        
        for m in self._models :
            logpredprobs_m, dlogpredprobs_m = m.logpdf(t) 
            logpredprobs = logpredprobs + logpredprobs_m
            dlogpredprobs = dlogpredprobs + dlogpredprobs_m

        self.t = t

        return logpredprobs, dlogpredprobs
    


class TIM_Params_Mixin():
    
    def set_trainable_params(self, theta) :
         params_dics = separate_merged_params_dic(theta, self.n_dimension)
         for m, dic in zip(self._models, params_dics) :
            m.set_trainable_params(dic)
       
    @property
    def trainable_parameters(self) :
        tmp = np.array([m.trainable_parameters for m in self._models]).flatten()
        return merge_tim_params_dics(tmp)




class MultiCompositeCP(MultiSeparateParamsCPBase):
    
    def __init__(self, model, n_dimension):
        super().__init__(model, n_dimension)
        
        if model.mtype == model_class.TSCP or model.mtype == model_class.TS :
            raise ValueError("MultiComposite do not accept TIM or TIMCP models")
        

class MultiTIMCP( TIM_Params_Mixin, MultiSeparateParamsCPBase) :
    
    def __init__(self, model, n_dimension):
        MultiBase.__init__(self, model, n_dimension)
        

class MultiSharedCP(MultiSharedParamsBase):
    
    def __init__(self, model, n_dimension):
        super().__init__(model, n_dimension)
        
        if model.mtype == model_class.TSCP or model.mtype == model_class.TS :
            raise ValueError("MultiComposite do not accept TIM or TIMCP models")
        
class MultiSharedTIMCP( TIM_Params_Mixin, MultiSharedParamsBase) :
    
    def __init__(self, model, n_dimension):
        MultiBase.__init__(self, model, n_dimension)
    

class MultiCompositeTS(Simple_TS_Mixin, MultiSeparateParamsTSBase):
    
    def __init__(self, model, n_dimension):
        super().__init__(model, n_dimension)
        
        if model.mtype == model_class.TSCP or model.mtype == model_class.TS :
            raise ValueError("MultiComposite do not accept TIM or TIMCP models")
        

class MultiTIMTS( Simple_TS_Mixin, TIM_Params_Mixin, MultiSeparateParamsTSBase) :
    
    def __init__(self, model, n_dimension):
        MultiBase.__init__(self, model, n_dimension)
        

class MultiSharedTS(Simple_TS_Mixin, MultiSharedParamsBase):
    
    def __init__(self, model, n_dimension):
        super().__init__(model, n_dimension)
        
        if model.mtype == model_class.TSCP or model.mtype == model_class.TS :
            raise ValueError("MultiComposite do not accept TIM or TIMCP models")
        
class MultiSharedTIMTS( Simple_TS_Mixin, TIM_Params_Mixin, MultiSharedParamsBase) :
    
    def __init__(self, model, n_dimension):
        MultiBase.__init__(self, model, n_dimension)
         


class MultiConstructor():

    def __init__(self):
        pass
    
    def __call__(self, mtype, shared_params = False) :
        
        if type(mtype) != model_class :
            raise ValueError("mtype must be a model_class enum")
    
        if not shared_params :

            if mtype == model_class.TS :
                return MultiTIMTS
            elif mtype == model_class.TSCP :
                return MultiTIMCP
            elif mtype == model_class.GPCP :
                return  MultiCompositeCP
            elif mtype == model_class.GP:
                return MultiCompositeTS
            raise NotImplementedError("Unrecognized MultiConstructor")
        
        else :
            if mtype == model_class.TS :
                return MultiSharedTIMTS
            elif mtype == model_class.TSCP :
                return MultiSharedTIMCP
            elif model_class.GPCP :
                return  MultiSharedCP
            elif mtype == model_class.GP :
                return  MultiSharedTS
            raise NotImplementedError("Unrecognized MultiConstructor")
    
     
        
   


# if __name__ == "__main__" :

#     import time

#     import bocpd.generate_data as gd
#     from bocpd.helper import ModelHelper, model_type

#     data, _, _ = gd.import_bee_data()
#     grid = np.array(range(len(data)))

#     X = data
#     X = (data - np.mean(data)) / np.std(data)
    
#     eval_cutoff = 1000
#     X_train = X[: eval_cutoff ]
#     grid_train = grid[: eval_cutoff ]

#     lengthscale = 1.0
#     variance = 2.0

#     maxLen = 300
#     noise_parameter = 0.01
#     prior_parameter = 2.0
    
#     L = 4/3*np.amax(abs(X), 0)
#     n_features = 5
    
#     optimizer = 'rt_minimize'
    
#     model = ModelHelper({'optimizer' : optimizer, 'maxiter' : 30, 'noise_trainable' : False}).get_model(
#         model_type.RRSPARCP, p = 1, variance = variance, lengthscale = lengthscale, noise_parameter =  noise_parameter,  prior_parameter = prior_parameter,
#         maxLen = maxLen, n_features  = n_features, L = max(L), horizontal_update = True)
    
#     m = MultiConstructor()(model.mtype, False)(model, 3)
#     #m=  MultiSharedCP(model.model, 3)
    
#     m.setL(L)
#     m.setData(X)
#     m.computeGradient(True)
#     m.initialize()

#     n = 4
    
#     t0 = time.time()
#     for i in range(n) :
#           out = m.logpdf(i)
#     out = m.logpdf(n)
#     print(out[1])
#     print("")

#     m0 = m._models[0]
#     m0.initialize()
    
#     m1 = m._models[1]
#     m1.initialize()

#     m2 = m._models[2]
#     m2.initialize()

#     for i in range(n) :
#         for mii in m._models :
#             mii.logpdf(i)

#     out0 = m0.logpdf(n)
#     out1 = m1.logpdf(n)
#     out2 = m2.logpdf(n)
    
#     print(out0[1])
#     print(out1[1])
#     print(out2[1])
#     print("")
    
#     outii = out0[1] + out1[1] + out2[1]
#     print(outii)
    
#     # ###########################

#     # mii = m1
#     # mii.computeGradient(False)
#     # mii.initialize()
    
#     # for i in range(n) :
#     #     mii.pdf(i)
#     # outii = mii.pdf(n)
    
#     # print(outii)
#     # print(np.log(outii))
#     # print("")

        
    
    
    
    
