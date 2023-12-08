from abc import ABC, abstractmethod
import numpy as np

from enum import Enum


class model_class(Enum):
    TS = 1,
    TSCP = 2,
    GP = 3,
    GPCP = 4,
    UNDEFINED = 5


class Model(ABC):
    """
    This is an abstract class to serve as a template for future users to mimick
    if they want to add new models for online bayesian changepoint detection.

    Make sure to override the abstract methods to do which is desired.
    Otherwise you will get an error.

    Update theta has **kwargs to pass in the timestep iteration (t) if desired.
    To use the time step add this into your update theta function:
        timestep = kwargs['t']
    """
    
    def __init__(self):
         self.name = ""
         self.isSet = False
         self.eval_gradient = False
         self.maxLen = None
         self.mtype = model_class.UNDEFINED
         
         self.compute_metrics = False
         self.grid_scale = 1.0
         self.p = int(0)
         self.process_0_values = False
         
    @property
    def n_features(self):
        return None
         
    def setData(self, data, grid = None):
        self.X = data
        self.grid = grid
        self.isSet = True
        
        if grid is None :
            self.grid = self.grid_scale * np.expand_dims(np.array(range(data.shape[0])),1)


    def computeGradient(self, eval_gradient = True) :
        self.eval_gradient = eval_gradient
    
        
    @property
    @abstractmethod
    def parameters(self) :
        raise NotImplementedError(
            "Parameters accessor. Please define in separate class to override this function."
        )
     
    @property
    @abstractmethod
    def num_params(self) :
            raise NotImplementedError(
            "Parameters accessor. Please define in separate class to override this function."
        )
    
    @property
    @abstractmethod
    def num_trainable_params(self) :
          raise NotImplementedError(
            "Parameters accessor. Please define in separate class to override this function."
        )
        
    @abstractmethod
    def set_trainable_params(self, theta) :
        raise NotImplementedError(
            "Parameters setter. Please define in separate class to override this function."
        )

    @abstractmethod
    def initialize(self):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def pdf(self, data: np.array):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )
        
    @abstractmethod
    def logpdf(self, data: np.array):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def update(self, data: np.array, **kwargs):
        raise NotImplementedError(
            "Update theta is not defined. Please define in separate class to override this function."
        )
        
    
    def MRC(self, t):
        if self.maxLen is not None :
            return min(t, self.maxLen)
        else :
            return t
        
        



class LogExpSpaceTransformer():

    def transform(self, a) :
        return np.log( np.exp( a) - 1)
    
    def inverse_transform(self, a ) :
        return np.log(np.exp( a) + 1)
    
    def gradient_factor(self, a) :
        return ( np.exp(a) - 1) / (a * np.exp(a))
    
    
class LogSpaceTransformer():

    def transform(self, a) :
        return np.log( a)
    
    def inverse_transform(self, a ) :
        return np.exp( a)
    
    def gradient_factor(self, a) :
        return np.ones_like(a)
    
    


class StudentProcessBase(Model):
    
    def __init__(self, kernel, prior_parameter, noise_parameter = 0.0):
        
        assert len(np.array(prior_parameter).shape) < 2
        super().__init__()

        self.kernel = kernel
        self.fast_computation = True
        self.transformer = LogSpaceTransformer()
        
        # Load scalePrior parameters
        self._prior_parameter = prior_parameter
 
        #noise scale 
        self._noise_parameter = noise_parameter
        self.noise_trainable = False #not trainable by default
        self.prior_trainable = True #trainable by default


    @property
    def is_noise(self):
        return self._noise_parameter != 0.0
    
    @property
    def num_kernel_params(self):
        return self.kernel.n_dims
    
    @property
    def num_trainable_params(self) :
        return self.num_kernel_params + int(self.noise_trainable) + int(self.prior_trainable)
    
    @property
    def num_params(self) :
        return self.num_kernel_params + int(self.is_noise) + 1 #prior parameter
    
    @property
    def _noise_param_idx(self):
        return self.num_kernel_params
    
    @property
    def _prior_param_idx(self):
        return -1

    @property
    def parameters(self):
        params = np.zeros(self.num_params)
        params[:self.num_kernel_params] = np.exp(self.kernel.theta)
        params[self._prior_param_idx] = self._prior_parameter
        if self.is_noise :
            params[self._noise_param_idx] = self._noise_parameter
        return  params

    @property
    def trainable_parameters(self):
        params = np.zeros(self.num_trainable_params)
        params[:self.num_kernel_params] = np.exp(self.kernel.theta)
        if self.noise_trainable  :  params[self._noise_param_idx] = self._noise_parameter
        if self.prior_trainable : params[self._prior_param_idx] = self._prior_parameter
        return  self.transformer.transform(params)


    def set_trainable_params(self, params):
        params = self.transformer.inverse_transform(params)
        self.kernel.theta = np.log(params[:self.num_kernel_params])
    
        if self.noise_trainable :
            self._noise_parameter = params[self._noise_param_idx]
            
        if self.prior_trainable :
            self._prior_parameter = params[self._prior_param_idx]
        
        
    @property
    def gradient_factor(self):
        params = self.parameters
        if self.is_noise and not self.noise_trainable :
            params = np.append(np.exp(params[:self.num_kernel_params]), params[self._prior_param_idx])
            return self.transformer.gradient_factor(params)
        else :
            return self.transformer.gradient_factor(params)
            
        
       

class GaussianProcessBase(Model):
    
    def __init__(self, kernel, noise_parameter = 0.1):
        
        super().__init__()
        self.kernel = kernel
        self.fast_computation = True
        self.transformer = LogSpaceTransformer()
        
        # Load scalePrior parameters
        self.noise_trainable = False
        self._noise_parameter = noise_parameter
        
    @property
    def is_noise(self):
        return self._noise_parameter != 0.0

    @property
    def num_kernel_params(self):
        return self.kernel.n_dims
    
    @property
    def num_trainable_params(self) :
        return self.num_kernel_params + int(self.noise_trainable)
    
    @property
    def num_params(self) :
        return self.num_kernel_params + int(self.is_noise) 
    
    @property
    def _noise_param_idx(self):
        return -1

    @property
    def parameters(self):
        params = np.zeros(self.num_params)
        params[:self.num_kernel_params] = np.exp(self.kernel.theta)
        if self.is_noise :
            params[self._noise_param_idx] = self._noise_parameter
        return  params

    @property
    def trainable_parameters(self):
        params = np.zeros(self.num_trainable_params)
        params[:self.num_kernel_params] = np.exp(self.kernel.theta)
        if self.noise_trainable  :
             params[self._noise_param_idx] =  self._noise_parameter
        return  self.transformer.transform(params)
            

    def set_trainable_params(self, params):
        
        params = self.transformer.inverse_transform(params)
        self.kernel.theta = np.log(params[:self.num_kernel_params])
        
        if self.noise_trainable :
            self._noise_parameter =  params[self._noise_param_idx] 
        
    @property
    def gradient_factor(self):
        if self.is_noise and not self.noise_trainable :
            return self.transformer.gradient_factor(self.parameters[:-1])
        else :
            return self.transformer.gradient_factor(self.parameters)
            
        
    

