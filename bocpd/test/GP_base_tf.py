
import numpy as np
from bocpd.model_base import Model
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions




def logexp_space_transformation( a ) :
    return np.log( np.exp( a) - 1)

def logexp_space_inverse_transformation( a ) :
    return tf.math.log( tf.math.exp( a) + 1)

def logexp_space_gradient_factor(a):
    #transform gradient from  np.log( np.exp( a) - 1) space to log( a ) space
    return  a * np.exp(a) / ( np.exp(a) - 1)



class TensorMisc():
    
    @staticmethod
    def pack_tensors(tensors):
        flats = [tf.reshape(tensor, (-1,)) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector
    
    
    @staticmethod
    def pack_tensors_to_zeros(tensors) :
        flats = [tf.zeros(shape = tf.reshape(tensor, (-1,)).shape, dtype = tensor.dtype) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector


    @staticmethod
    def unpack_tensors(to_tensors, from_vector) :
        s = 0
        values = []
        for target_tensor in to_tensors:
            shape = tf.shape(target_tensor)
            dtype = target_tensor.dtype
            tensor_size = tf.reduce_prod(shape)
            tensor_vector = from_vector[s : s + tensor_size]
            tensor = tf.reshape(tf.cast(tensor_vector, dtype), shape)
            values.append(tensor)
            s += tensor_size
        return values
    

    @staticmethod
    def assign_tensors(to_tensors, values) -> None:
        if len(to_tensors) != len(values):
            raise ValueError("to_tensors and values should have same length")
        for target, value in zip(to_tensors, values):
            target.assign(value)
    
   



class STPBase_TF(Model):
    
    def __init__(self, kernel, prior_parameter, noise_parameter = 0.0):
        
        super().__init__()
        self.kernel = kernel
        self.fast_computation = True
        
        self.noise_trainable = False 
        self._logexp_noise_parameter = None
        
        self.unit_beta = True
        
        if noise_parameter != 0.0 :
            self._logexp_noise_parameter = tf.Variable(logexp_space_transformation(noise_parameter))

        # Load scalePrior parameters
        self._logexp_prior_parameter = tf.Variable(logexp_space_transformation(prior_parameter))
        
    def num_params(self):
        return len(self.parameters)
    
    def num_trainable_params(self):
        return len(self.trainable_parameters)
        
    def addTrainableNoise(self, noise_parameter) :
         self._logexp_noise_parameter = tf.Variable(logexp_space_transformation(noise_parameter ))
         self.noise_trainable = True 

    def computeGradient(self, eval_gradient = True) :
        if eval_gradient  :
            raise ValueError("cannot turn gradient computation on class 'GPTSCP_TF' ")
        pass
    
    @property
    def is_noise(self):
        if tf.is_tensor(self._logexp_noise_parameter) :
            return (self._logexp_noise_parameter ).numpy() is not None
        else :
            return self._logexp_noise_parameter is not None
        
    
    @property
    def _noise_parameter(self) :
        if self.is_noise :
            return logexp_space_inverse_transformation(self._logexp_noise_parameter )
        else :
            return 0.0
        
    @property
    def _prior_parameter(self) :
        return logexp_space_inverse_transformation(self._logexp_prior_parameter )
        
    
    @property
    def _noise_scale(self) :
        return self._noise_parameter
        
    @property
    def _alpha0(self) :
        return self._prior_parameter
    
    @property
    def _beta0(self):
        if self.unit_beta:
            return 1.0
        else :
            return self._prior_parameter
    
    @property
    def grad_adjustment_factor(self):
        params = self.parameters
        if self.is_noise and not self.noise_trainable :
            params = np.append(params[:-2], params[-1])
        return logexp_space_gradient_factor(params)


    @property
    def parameters(self):
        kernel_parameters = TensorMisc().pack_tensors(self.kernel.trainable_variables)
        out = np.array(logexp_space_inverse_transformation(kernel_parameters))
        if self.is_noise :
            out = np.append(out, self._noise_parameter)
        return  np.append(out, self._prior_parameter)

    @property
    def trainable_parameters(self):
        out = self.kernel.trainable_variables
        if self.noise_trainable and self.is_noise :
            out += (self._logexp_noise_parameter,)
        return  out + (self._logexp_prior_parameter,)


    def set_trainable_params(self, params):
        
         if self.noise_trainable :
            
             for v, p in zip(self.kernel.trainable_variables, params[:-2]):
                v.assign(p)

             self._logexp_noise_parameter = tf.Variable(params[-2]) 
            
         else :
            for v, p in zip(self.kernel.trainable_variables, params[:-1]):
                v.assign(p)
                
         self._logexp_prior_parameter = tf.Variable(params[-1]) 
        
    
    def logpdf(self, t):
        return tf.math.log(self.pdf(t))
    
    
    


class GPBase_TF(Model):
    
    def __init__(self, kernel, noise_parameter):
        
        super().__init__()
        self.kernel = kernel
        self.fast_computation = True

        self.noise_trainable = False 
        self._logexp_noise_parameter = None
        
        if noise_parameter != 0 :
            self._logexp_noise_parameter = tf.Variable(logexp_space_transformation(noise_parameter))
            
    @property
    def is_noise(self):
        if tf.is_tensor(self._logexp_noise_parameter) :
            return (self._logexp_noise_parameter).numpy() is not None
        else :
            return self._logexp_noise_parameter is not None
        
    def num_params(self):
        return len(self.parameters)
    
    def num_trainable_params(self):
        return len(self.trainable_parameters)
        


    def computeGradient(self, eval_gradient = True) :
        if eval_gradient  :
            raise ValueError("cannot turn gradient computation on class 'GPTSCP_TF' ")
        pass

        
    @property
    def _noise_parameter(self) :
        if self.is_noise :
            return logexp_space_inverse_transformation(self._logexp_noise_parameter)
        else :
            return 0.0

    @property
    def _noise_scale(self) :
        return self._noise_parameter

    @property
    def grad_adjustment_factor(self):
        params = self.parameters
        if not self.noise_trainable :
            params = params[:-1]
        return logexp_space_gradient_factor(self.parameters)

    @property
    def parameters(self):
        trainable_variables = TensorMisc().pack_tensors(self.kernel.trainable_variables)
        if self.is_noise :
            return  np.append(np.array(logexp_space_inverse_transformation(trainable_variables)), self._noise_parameter)
        else :
            return np.array(logexp_space_inverse_transformation(trainable_variables))

    @property
    def trainable_parameters(self):
        if self.noise_trainable :
            return  self.kernel.trainable_variables + (self._logexp_noise_parameter,)
        else :
            
            return self.kernel.trainable_variables


    def set_trainable_params(self, params):
        
        if self.noise_trainable :
            for v, p in zip(self.kernel.trainable_variables, params[:-1]):
                v.assign(p)
    
            self._logexp_noise_parameter = tf.Variable(params[-1]) 
            
        else :
            for v, p in zip(self.kernel.trainable_variables, params):
                v.assign(p)
            

    def logpdf(self, t):
        return tf.math.log(self.pdf(t))


