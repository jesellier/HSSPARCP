
import numpy as np

from bocpd.GPAR.arsplit import ARsplit
from bocpd.model_ts import SPTimeSerieBase, GPTimeSerieBase
from bocpd.RRGPAR.gaussian_laplace_reduced_rank import  GaussianLaplaceReducedRank, GaussianLaplaceReducedRankND
from bocpd.RRGPAR.rr_model_base import RR_GPBase_Mixin, RR_SPBase_Mixin
import scipy

def fastest_update(Q,u):
    # Warning: `overwrite_a=True` silently fails when B is not an order=F array!
    #use DGER, which perform the ran1 iteration : Q:= Q+uu⊤
    Qu = Q @ u
    alpha = -1 / (1 + u.T @ Qu)
    return scipy.linalg.blas.dger(alpha, Qu, u.T @ Q, a=Q, overwrite_a=1) 

def fastest_downdate(Q,u):
    # Warning: `overwrite_a=True` silently fails when B is not an order=F array!
    #use DGER, which perform the ran1 iteration : Q:= Q+uu⊤
    Qu = Q @ u
    alpha = 1 / (1 - u.T @ Qu)
    return scipy.linalg.blas.dger(alpha, Qu, u.T @ Q, a=Q, overwrite_a=1) 




class RR_Prediction_Mixin():

    @property
    def v_inv_lamb(self):
        return  self.total_noise_variance *  np.diag(self.invΛ)

    
    def prior(self):
        mu = 0.0
        sigma2 = self.kernel(self.lagMatrix[0].T) 
        return mu, sigma2[0][0]
    
    def prior_with_grads(self):
        
        mu = 0.0
        dmu = np.zeros(len(self.kernel.theta))
        dsigma2 = np.zeros(len(self.kernel.theta))

        #(sigma2, dsigma2) = self.kernel.eval_with_grads(self.lagMatrix[0].T)
        #dsigma2 = dsigma2[0][0]
        sigma2 = dsigma2[0] = self.kernel.variance
        
        if self.noise_trainable :
            dsigma2 = np.append(dsigma2, 0.0)
            dmu = np.append(dmu, 0.0)
            
        return (mu, dmu), (sigma2, dsigma2)
    
    
    def prediction(self, t):
         
        if self.eval_gradient is False  :
            return self.__prediction(t)
        else :
            return self.__prediction_with_grads(t)


    def __prediction(self, t):
        
        if t ==0 : 
            return self.prior()
        
        MRC = self.MRC(t)

        #xt = self.lagMatrix[t]
        yp = self.X[t- MRC  : t ]
        #xp = self.lagMatrix[:t,:,0]

        pxt = self.Phi[t,:].T

        if self.fast_computation :
            if t > 1 :
                #perform an update of the additional observation
                phi = np.expand_dims(self.Phi[t-1, :], 1)
                self.u += yp[-1]  * phi
                self.Q = fastest_update(self.Q, phi)
                
                if t > MRC :
                    #perform a downdate of the value last value dropped
                    phi = np.expand_dims(self.Phi[t- MRC - 1, :], 1)
                    self.u -= self.X[t- MRC - 1]  * phi
                    self.Q = fastest_downdate(self.Q, phi)
            else :
                phi = np.expand_dims(self.Phi[t-1, :], 1)
                self.u = yp[-1]  * phi
                self.Q= np.linalg.inv(phi @ phi.T +  self.v_inv_lamb)
        else :
            self.u = self.Phi[t- MRC :t,:].T @ yp
            self.Q = np.linalg.inv(self.Phi[t- MRC  :t,:].T @ self.Phi[t- MRC :t,:] + self.v_inv_lamb) 
            
        Q_u = self.Q @ self.u[:,0]
        Q_pxt = self.Q @ pxt
        
        mu  = pxt.T @ Q_u
        sigma2  = self.total_noise_variance * pxt.T @ Q_pxt

        if self.compute_SSE :
            self.SSE_t = yp.T @ yp - self.u.T @ Q_u

        return (mu, sigma2)
    
    
    def __prediction_with_grads(self, t):

        if t == 0 : 
            return self.prior_with_grads()
        
        MRC = self.MRC(t)

        #xt = self.lagMatrix[t]
        yp = self.X[t- MRC  : t ]
        #xp = self.lagMatrix[:t,:,0]

        pxt = self.Phi[t,:].T

        if self.fast_computation :
            if t > 1 :
                #perform an update of the additional observation
                phi = np.expand_dims(self.Phi[t-1, :], 1)
                self.u += yp[-1]  * phi
                self.Q = fastest_update(self.Q, phi)
                
                if t > MRC :
                    #perform a downdate of the value last value dropped
                    phi = np.expand_dims(self.Phi[t- MRC - 1, :], 1)
                    self.u -= self.X[t- MRC - 1]  * phi
                    self.Q = fastest_downdate(self.Q, phi)
            else :
                phi = np.expand_dims(self.Phi[t-1, :], 1)
                self.u = yp[-1]  * phi
                self.Q= np.linalg.inv(phi @ phi.T +  self.v_inv_lamb)
        else :
            self.u = self.Phi[t- MRC :t,:].T @ yp
            self.Q = np.linalg.inv(self.Phi[t- MRC  :t,:].T @ self.Phi[t- MRC :t,:] + self.v_inv_lamb) 
        
        Q_u = self.Q @ self.u[:,0]
        Q_pxt = self.Q @ pxt
        mu  = pxt.T @ Q_u
        sigma2  = self.total_noise_variance * pxt.T @ Q_pxt
        
        #compute the derivatives
        #dQ = Q  @ np.diag(self.total_noise_variance * dinvΛ) 
        dQ = self.total_noise_variance * (np.expand_dims(self.dinvΛ ,1) * np.expand_dims(self.Q,0)) 
        dQQu = np.expand_dims(dQ @ Q_u, 2)
        
        #dmu
        dmu = (pxt.T @ dQQu)[:,0] 
        
        #dsigma2 
        dsigma2 =  self.total_noise_variance * pxt.T @ dQ @ Q_pxt #pxt
        
        if self.compute_SSE :
            self.SSE_t = (yp.T @ yp - self.u.T @ Q_u)[:,0] / self.total_noise_variance
            self.dSSE_t =  - (self.u[:,0].T @ dQQu)[:,0] / self.total_noise_variance
        
        if self.noise_trainable :
            noise_variance = self._noise_parameter**2
            tmp = self.invΛ * Q_pxt
            dmu = np.append(dmu, - 2 * noise_variance * sum(tmp * Q_u))
            #dmu = np.append(dmu, 0)
            dsigma2 = np.append(dsigma2, - 2*self.total_noise_variance* noise_variance * sum(tmp * Q_pxt) + 2 * sigma2)
            
            if self.compute_SSE :
                tmp = 2 * noise_variance / self.total_noise_variance  * (sum(self.invΛ* Q_u**2)  - self.SSE_t )
                self.dSSE_t = np.append(self.dSSE_t, tmp)
                
            
        return (mu, dmu), (sigma2, dsigma2)
    
    
      

        
        
class RRGPAR( RR_Prediction_Mixin, RR_GPBase_Mixin, GPTimeSerieBase ):
    
    def __init__(self, kernel, p = 1, noise_parameter = 0.1):
        super().__init__(kernel, noise_parameter)
        assert p > 0
        self.p = int(p)
        
        if self.is_noise :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10

        self.compute_SSE = False
        
        
    @property
    def total_noise_variance(self):
        return   self._noise_scale**2 + self._jitter
    
    @property
    def n_features(self):
        return  self.kernel.n_features

    def precompute(self):
         self.kernel.fit()
         self.lagMatrix = ARsplit(self.X, self.p)
         self.Phi = self.kernel.feature(self.lagMatrix[:,:,0]) 
         
         if self.eval_gradient is True :
             self.invΛ, self.dinvΛ = self.kernel.invΛ_with_grad()
             #self.dinvΛ = np.array(self.dinvΛ)
         else :
             self.invΛ = self.kernel.invΛ
             
             

class RRSPAR( RR_Prediction_Mixin, RR_SPBase_Mixin, SPTimeSerieBase):
    
    
    def __init__(self, kernel, p = 1, prior_parameter = 0.1, noise_parameter = 0.0):
        super().__init__(kernel, prior_parameter, noise_parameter)
        assert p > 0
        self.p = int(p)
        
        if self.is_noise :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10

        self.compute_SSE = True
        self.unit_beta = False
 
        
    @property
    def total_noise_variance(self):
        return   self._noise_scale**2 + self._jitter
    
    @property
    def n_features(self):
        return  self.kernel.n_features

    def precompute(self):
         self.kernel.fit()
         self.lagMatrix = ARsplit(self.X, self.p)
         self.Phi = self.kernel.feature(self.lagMatrix[:,:,0]) 
         
         if self.eval_gradient is True :
             self.invΛ, self.dinvΛ = self.kernel.invΛ_with_grad()
             #self.dinvΛ = np.array(self.dinvΛ)
         else :
             self.invΛ = self.kernel.invΛ


    


    
    