
import numpy as np
import scipy.stats as ss

from bocpd.model_base import Model, model_class


class UPM(Model):
    
     def __init__(self) :
         super().__init__()
         self.mtype = model_class.TSCP
         pass
     
     def logpdf(self, t):
         return np.log(self.pdf(t))
     
     @property
     def num_trainable_params(self) :
        return len(self.trainable_parameters)
    
     @property
     def num_params(self) :
        return len(self.parameters)


class StudentUPM(UPM):
    
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):
        """
        StudentT distribution except normal distribution is replaced with the student T distribution
        https://en.wikipedia.org/wiki/Normal-gamma_distribution

        Parameters:
            alpha - alpha in gamma distribution prior
            beta - beta inn gamma distribution prior
            mu - mean from normal distribution
            kappa - variance from normal distribution
        """
        super().__init__()
        self.alpha0 = np.resize(alpha,1)
        self.beta0 = np.resize(beta,1)
        self.kappa0 = np.resize(kappa,1)
        self.mu0 = np.resize(mu,1)
        self.trainable_entries = ['alpha0', 'beta0']
        
    @property
    def parameters(self):
        return  np.array([self.alpha0[0], self.beta0[0], self.mu0[0], self.kappa0[0]])
    
    @property
    def trainable_parameters(self ) :
        tmp = {}
        for e in  self.trainable_entries :
            tmp[e] = getattr(self, e)
        return tmp

    def set_trainable_params(self, entry_dict):
        for key in entry_dict :
            setattr(self, key,  np.atleast_1d(entry_dict[key])) 
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

    def update(self, t):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        mu_new  = np.concatenate(
            (self.mu0, (self.kappa * self.mu + self.X[t]) / (self.kappa + 1))
        )
        kappa_new = np.concatenate((self.kappa0, self.kappa + 1.0))
        alpha_new = np.concatenate((self.alpha0, self.alpha + 0.5))
        beta_new = np.concatenate((
                self.beta0,
                self.beta + (self.kappa * (self.X[t] - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))
            )
        )

        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new
        
        
class GaussianUPM(UPM):
    
    def __init__(
        self, mu0: float = 0, kappa0 : float = 1, lamb: float = 1
    ):
        """
        Parameters:
            mu0 - mean from normal distribution
        """
        super().__init__()
        self.mu0 = np.resize(mu0,1)
        self.kappa0 = kappa0
        self.λ = lamb
        
        self.running_sum = np.array([0])
        self.count =  np.array([0])
        
        self.trainable_entries = ['kappa0', 'λ']
        
    
    def trainable_parameters(self) :
        tmp = {}
        for e in self.trainable_entries :
            tmp[e] = getattr(self, e)
        return tmp

    def set_trainable_params(self, entry_dict):
        for key in entry_dict :
            setattr(self, key,  np.atleast_1d(entry_dict[key])) 
        pass

        
    def initialize(self):
        self.mu = self.mu0 
        self.kappa =self.kappa0 
        
        self.running_sum = np.array([0])
        self.count =  np.array([0])

    
    @property
    def scale2(self):
        return 1 / self.kappa + 1 / self.λ
        

    def pdf(self, t):
        """
        Return the pdf function of the t distribution

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        
        if self.compute_metrics  is True :
            self.abs_error_t = np.abs(self.X[t, 0] - self.mu)
            self.abs_error_2_t = self.abs_error_t **2
        
        return ss.norm.pdf(
            x=self.X[t],
            loc=self.mu,
            scale=np.sqrt(self.scale2)
        )

    def update(self, t):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        self.running_sum  = np.concatenate((np.array([0]), (self.running_sum + self.X[t] )))
        self.count = np.concatenate((np.array([0]), (self.count + 1 )))
        
        self.kappa = self.count * self.λ + self.kappa0 
        self.mu = (self.mu0 * self.λ0 + self.running_sum * self.λ) / self.kappa
 
        
    



