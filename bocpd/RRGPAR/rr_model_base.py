import numpy as np


class RR_SPBase_Mixin():
    
    
    @property
    def n_features(self):
        self.kernel.n_features
    
    @property
    def num_kernel_params(self):
        return self.kernel.num_params

    @property
    def parameters(self):
        params = np.zeros(self.num_params)
        params[:self.num_kernel_params] = self.kernel.theta
        params[self._prior_param_idx] = self._prior_parameter
        if self.is_noise :
            params[self._noise_param_idx] = self._noise_parameter
        return  params

    @property
    def trainable_parameters(self):
        params = np.zeros(self.num_trainable_params)
        params[:self.num_kernel_params] = self.kernel.theta
        if self.noise_trainable  :  params[self._noise_param_idx] = self._noise_parameter
        if self.prior_trainable : params[self._prior_param_idx] = self._prior_parameter
        return  self.transformer.transform(params)


    def set_trainable_params(self, params):
        params = self.transformer.inverse_transform(params)
        self.kernel.set_theta(params[:self.num_kernel_params])
    
        if self.noise_trainable  :
            self._noise_parameter = params[self._noise_param_idx]
            
        if self.prior_trainable :
            self._prior_parameter = params[self._prior_param_idx]
            
        
    @property
    def gradient_factor(self):
        #return self.transformer.gradient_factor(self.transformer.inverse_transform(self.trainable_parameters))
        return 1.0
        

class RR_GPBase_Mixin():
    
    @property
    def n_features(self):
        self.kernel.n_features

    @property
    def num_kernel_params(self):
        return self.kernel.num_params
    
    @property
    def parameters(self):
        params = np.zeros(self.num_params)
        params[:self.num_kernel_params] = self.kernel.theta
        if self.is_noise :
            params[self._noise_param_idx] = self._noise_parameter
        return  params

    @property
    def trainable_parameters(self):
        params = np.zeros(self.num_trainable_params)
        params[:self.num_kernel_params] = self.kernel.theta
        if self.noise_trainable  :  params[self._noise_param_idx] = self._noise_parameter
        return  self.transformer.transform(params)


    def set_trainable_params(self, params):
        params = self.transformer.inverse_transform(params)
        self.kernel.set_theta(params[:self.num_kernel_params])
    
        if self.noise_trainable : 
            self._noise_parameter = params[self._noise_param_idx]

        
    @property
    def gradient_factor(self):
        #return self.transformer.gradient_factor(self.transformer.inverse_transform(self.trainable_parameters))
        return 1.0
