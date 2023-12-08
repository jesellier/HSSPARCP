
import numpy as np
import copy

from bocpd.GPAR.arsplit import ARsplit
from bocpd.GPAR.GPARCP import ARPredictionCPMixin
from bocpd.model_cp import GPCPBase, SPCPBase
from bocpd.RRGPAR.rr_model_base import RR_GPBase_Mixin, RR_SPBase_Mixin
from bocpd.RRGPAR.RRGPAR import  fastest_update, fastest_downdate



class RRCP_Prediction_Mixin():
            
    @property
    def v_inv_lamb(self):
        return  self.total_noise_variance *  np.diag(self.invΛ)
    
    @property
    def total_noise_variance(self):
        return   self._noise_parameter**2 + self._jitter

    @property
    def n_features(self):
        return  self.kernel.n_features
    
    @property
    def n_pred_params(self):
        tmp = self.kernel.num_params
        if self.noise_trainable is True :
            tmp += 1
        return tmp
            

    def precompute(self):
        
        self.kernel.fit()
        self.lagMatrix = ARsplit(self.X, self.p)
        self.Phi = self.kernel.feature(self.lagMatrix[:,:,0]) 
        self._n_pred_params = self.n_pred_params
       
        if self.eval_gradient is True :
             self.invΛ, self.dinvΛ = self.kernel.invΛ_with_grad()
             #self.dinvΛ = np.array(self.dinvΛ)
        else :
             self.invΛ = self.kernel.invΛ

        self.SSE_t = np.atleast_2d(0)
        self.dSSE_t = np.zeros((1, self._n_pred_params))
        self.cum_norm = 0
  
    
    def prior(self):
        mu = 0.0
        sigma2 = self.kernel.variance  * np.ones((1))
        
        return mu, sigma2
    
    
    def prior_with_grads(self):
            
        mu = np.zeros((1))
        dmu = np.zeros(len(self.kernel.theta))
        
        sigma2 = self.kernel.variance * np.ones((1))
        dsigma2 = np.zeros(len(self.kernel.theta))
        dsigma2[0] = sigma2

        if self.noise_trainable :
            dsigma2 = np.append(dsigma2, 0.0)
            dmu = np.append(dmu, 0.0)
            
        return (mu, np.atleast_2d(dmu)), (sigma2, np.atleast_2d(dsigma2))
    
    
    def prediction(self, t):
        
        if self.horizontal_update :
            if self.eval_gradient is False  :
                return self.__prediction_hu(t)
            else :
                return self.__prediction_hu_with_grads(t)
        else : 
            if self.eval_gradient is False  :
                return self.__prediction(t)
            else :
                return self.__prediction_with_grads(t)


    def __prediction(self, t):
        if t ==0 : 
            return self.prior()
        
        MRC = self.MRC(t)

        yp = self.X[t- MRC  : t ]
        pxt = self.Phi[t,:].T
        
        mu = np.zeros((t+1,1))
        sigma2 = np.zeros((t+1, 1))

        ########## 
        phi = np.expand_dims(self.Phi[t-1, :], 1)
        cum_norm = yp[-1]**2
        u = yp[-1]  * phi
        Q = np.linalg.inv(phi @ phi.T +  self.v_inv_lamb)
        
        mu[1] =  pxt.T  @ Q @ u
        sigma2[1] =  self.total_noise_variance * pxt.T @ Q @ pxt
        
        if self.compute_SSE :
            self.SSE_t = np.zeros((t+1))
            self.SSE_t[1] = cum_norm - u.T @ Q @ u

        for ii in range(2, MRC +1) :
            phi = np.expand_dims(self.Phi[t-ii, :], 1)
            u += yp[-ii] * phi  #u = P[-ii:].T @ yp1[-ii:]
            
            Q = fastest_update(Q, phi)  #Q = np.linalg.inv(P[-ii:].T @ P[-ii:] + variance_lamb_inv)
            mu[ii]  = pxt.T @ Q @ u
            sigma2[ii]  = self.total_noise_variance * pxt.T @ Q @ pxt
            
            if self.compute_SSE :
                cum_norm +=  yp[-ii]**2
                self.SSE_t[ii] = cum_norm - u.T @ Q @ u
              
        # adjust for change point i.e add prior distribution on top of the vector
        #mu[0] = self.SSE_t[0] = 0.0
        sigma2[0] = self.kernel.variance
        
        if self.compute_SSE :
            self.SSE_t = self.SSE_t / self.total_noise_variance
            
        #Extend the mu prediction for the older (> MRC) run length hypothesis
        if MRC < t:  
            mu[MRC+1:] = mu[MRC] * np.ones((t - MRC,1))  # t - MRC x 1. [x]
            sigma2[MRC+1:] =  sigma2[MRC] * np.ones((t - MRC,1))
            
            if self.compute_SSE :
                self.SSE_t[MRC+1:] = self.SSE_t[MRC] * np.ones((t - MRC))

        return (mu[:,0], sigma2[:,0])
    
    
    def __prediction_with_grads(self, t):
        
        if t ==0 : 
            return self.prior_with_grads()
        
        MRC = self.MRC(t)

        yp = self.X[t- MRC  : t ]
        pxt = self.Phi[t,:].T

        noise_variance = self._noise_parameter **2
        adj_invΛ = noise_variance * self.invΛ / self.total_noise_variance

        mu = np.zeros((t+1))
        sigma2 = np.zeros((t+1))

        dmu = np.zeros((t+1, self._n_pred_params))
        dsigma2 = np.zeros((t+1, self._n_pred_params))
        
        ########## initialize first run lenght
        phi = np.expand_dims(self.Phi[t-1, :], 1)
        cum_norm = 0 
        u = yp[-1]  * phi
        Q = np.linalg.inv(phi @ phi.T +  self.v_inv_lamb)
        
        #mu[1] =  pxt.T  @ Q @ u
        #sigma2[1] =  pxt.T @ Q @ pxt
        
        if self.compute_SSE :
            self.SSE_t = np.zeros((t+1))
            self.dSSE_t = np.zeros((t+1, self._n_pred_params))
            #self.SSE_t[1] = (cum_norm - u.T @ Q @ u)

        ########## loop others
        for ii in range(1, MRC +1) :
            
            if ii > 1 :
                #perform fast update
                phi = np.expand_dims(self.Phi[t-ii, :], 1)
                u += yp[-ii] * phi  #u = P[-ii:].T @ yp[-ii:]
                Q = fastest_update(Q, phi)  #Q = np.linalg.inv(P[-ii:].T @ P[-ii:] + variance_lamb_inv)
            
            Q_u = Q @ u[:,0]
            Q_pxt = Q @ pxt
            
            mu[ii]  =  pxt.T @ Q_u
            sigma2[ii]  = pxt.T @ Q_pxt

            dQ =  (np.expand_dims(self.dinvΛ ,1) * np.expand_dims(Q,0)) 
            dQQu = np.expand_dims(dQ @ Q_u, 2)
            
            if self.noise_trainable is False :
                dmu[ii,:] = (pxt.T @ dQQu)[:,0]
                dsigma2[ii,:] =   pxt.T @ dQ @ Q_pxt
            
                if self.compute_SSE :
                    cum_norm +=  yp[-ii]**2
                    self.SSE_t[ii] = cum_norm  - u.T @ Q_u
                    self.dSSE_t[ii,:] =  - (u[:,0].T @ dQQu)[:,0] 
    
            else :
                tmp = - 2 * adj_invΛ * Q_pxt
                dmu[ii,:] = np.append( (pxt.T @ dQQu)[:,0], sum(tmp * Q_u))
                dsigma2[ii,:] = np.append(pxt.T @ dQ @ Q_pxt, sum(tmp * Q_pxt))
        
                if self.compute_SSE :
                    cum_norm +=  yp[-ii]**2
                    self.SSE_t[ii] = cum_norm - u.T @ Q_u 
                    self.dSSE_t[ii,:] =  np.append( - (u[:,0].T @ dQQu)[:,0],  2* sum(adj_invΛ* Q_u**2)) 

        #adjustment and rescaling
        dmu *= self.total_noise_variance
        sigma2 *= self.total_noise_variance
        dsigma2 *= self.total_noise_variance **2
        
        if self.noise_trainable is True :
            dsigma2[:,-1] +=  2 * sigma2
        
        if self.compute_SSE :
            self.SSE_t /= self.total_noise_variance
            
            if self.noise_trainable is True :
                self.dSSE_t[:,-1] = self.dSSE_t[:,-1] -  2 * noise_variance * self.SSE_t / self.total_noise_variance 
        
        # adjust for change point i.e add prior distribution on top of the vector
        #mu[0] = self.SSE_t[0] = 0.0
        sigma2[0] = self.kernel.variance

        #Extend the mu prediction for the older (> MRC) run length hypothesis
        if MRC < t:  
            mu[MRC+1:] = mu[MRC] * np.ones((t - MRC))  # t - MRC x 1. [x]
            sigma2[MRC+1:] =  sigma2[MRC] * np.ones((t - MRC))
            
            if self.compute_SSE :
                 self.SSE_t[MRC+1:] = self.SSE_t[MRC] * np.ones((t - MRC))
   
        return (mu, dmu), (sigma2, dsigma2)
    
    

    def __prediction_hu(self, t):
        
        if t ==0 : 
            return self.prior()
        
        MRC = self.MRC(t)
        yp = self.X[t- MRC  : t ]

        if t > 1 :
            #perform an update of the additional observation
            phi = np.expand_dims(self.Phi[t-1, :], 1)
            self.u = self.u + yp[-1]  * phi
            self.Q = fastest_update(self.Q, phi)
            self.cum_norm += yp[-1]**2
                        
            if t > MRC :
                #perform a downdate of the value last value dropped
                self.cum_norm -= self.X[t- MRC - 1]**2
                phi = np.expand_dims(self.Phi[t- MRC - 1, :], 1)
                self.u = self.u -  self.X[t- MRC - 1]  * phi
                self.Q = fastest_downdate(self.Q, phi)
        else :
            self.u = self.Phi[t- MRC :t,:].T @ yp
            self.Q = np.linalg.inv(self.Phi[t- MRC  :t,:].T @ self.Phi[t- MRC :t,:] + self.v_inv_lamb) 
            self.cum_norm += yp[-1]**2
    

        pxt = self.Phi[t,:].T
        mu = np.zeros((t+1,1))
        sigma2 = np.zeros((t+1, 1))
        
        if self.compute_SSE :
            self.SSE_t = np.zeros((t+1))

        ########## 
        cum_norm_t = copy.copy(self.cum_norm)
        Q = copy.copy(self.Q)
        u = copy.copy(self.u)
    
        for ii in range(MRC, 0, -1) :
            
            if ii < MRC :  #do update
                phi = np.expand_dims(self.Phi[t-ii-1, :], 1)
                u -= yp[-ii-1] * phi
                cum_norm_t -=  yp[-ii-1]**2
                Q = fastest_downdate(Q, phi) #np.linalg.inv(Phi[-ii:].T @ Phi[-ii:] + m.v_inv_lamb)
            
            mu[ii]  = pxt.T @ Q @ u
            sigma2[ii]  = self.total_noise_variance * pxt.T @ Q @ pxt
            if self.compute_SSE :
                self.SSE_t[ii] = (cum_norm_t - u.T @ Q @ u) 
                  
        # # adjust for change point i.e add prior distribution on top of the vector
        #mu[0] = self.SSE_t[0] = 0.0
        sigma2[0] = self.kernel.variance
        if self.compute_SSE :
            self.SSE_t = self.SSE_t / self.total_noise_variance
                
        #Extend the mu prediction for the older (> MRC) run length hypothesis
        if MRC < t:  
            mu[MRC+1:] = mu[MRC] * np.ones((t - MRC,1))  # t - MRC x 1. [x]
            sigma2[MRC+1:] =  sigma2[MRC] * np.ones((t - MRC,1))
            if self.compute_SSE :
                self.SSE_t[MRC+1:] = self.SSE_t[MRC] * np.ones((t - MRC))
    
        return (mu[:,0], sigma2[:,0])
    
    
    def __prediction_hu_with_grads(self, t):
   
        if t ==0 : 
            return self.prior_with_grads()
        
        MRC = self.MRC(t)
        yp = self.X[t- MRC  : t ]
        
        noise_variance = self._noise_parameter **2
        adj_invΛ = noise_variance * self.invΛ / self.total_noise_variance

        if t > 1 :
            #perform an update of the additional observation
            phi = np.expand_dims(self.Phi[t-1, :], 1)
            self.u = self.u + yp[-1]  * phi
            self.Q = fastest_update(self.Q, phi)
            self.cum_norm += yp[-1]**2
                        
            if t > MRC :
                #perform a downdate of the value last value dropped
                self.cum_norm -= self.X[t- MRC - 1]**2
                phi = np.expand_dims(self.Phi[t- MRC - 1, :], 1)
                self.u = self.u -  self.X[t- MRC - 1]  * phi
                self.Q = fastest_downdate(self.Q, phi)
        else :
            self.u = self.Phi[t- MRC :t,:].T @ yp
            self.Q = np.linalg.inv(self.Phi[t- MRC  :t,:].T @ self.Phi[t- MRC :t,:] + self.v_inv_lamb) 
            self.cum_norm += yp[-1]**2
 
        pxt = self.Phi[t,:].T
        mu = np.zeros((t+1))
        sigma2 = np.zeros((t+1))

        dmu = np.zeros((t+1, self._n_pred_params))
        dsigma2 = np.zeros((t+1, self._n_pred_params))
  
        if self.compute_SSE :
            self.SSE_t = np.zeros((t+1))
            self.dSSE_t = np.zeros((t+1, self._n_pred_params))
      
        ########## 
        cum_norm_t = copy.copy(self.cum_norm)
        Q = copy.copy(self.Q)
        u = copy.copy(self.u)
    
        for ii in range(MRC, 0, -1) :
            
            if ii < MRC :  #do update
                phi = np.expand_dims(self.Phi[t-ii-1, :], 1)
                u -= yp[-ii-1] * phi
                cum_norm_t -=  yp[-ii-1]**2
                Q = fastest_downdate(Q, phi) #np.linalg.inv(Phi[-ii:].T @ Phi[-ii:] + m.v_inv_lamb)
                
            Q_u = Q @ u[:,0]
            Q_pxt = Q @ pxt
            
            mu[ii]  =  pxt.T @ Q_u
            sigma2[ii]  = pxt.T @ Q_pxt
     
            dQ =  (np.expand_dims(self.dinvΛ ,1) * np.expand_dims(Q,0)) 
            dQQu = np.expand_dims(dQ @ Q_u, 2)
   
            if self.noise_trainable is False :
                dmu[ii,:] = (pxt.T @ dQQu)[:,0]
                dsigma2[ii,:] =   pxt.T @ dQ @ Q_pxt
            
                if self.compute_SSE :
                    self.SSE_t[ii] = (cum_norm_t - u.T @ Q @ u) 
                    self.dSSE_t[ii,:] =  - (u[:,0].T @ dQQu)[:,0] 
            else :
                tmp = - 2 * adj_invΛ * Q_pxt
                dmu[ii,:] = np.append( (pxt.T @ dQQu)[:,0], sum(tmp * Q_u))
                dsigma2[ii,:] = np.append(pxt.T @ dQ @ Q_pxt, sum(tmp * Q_pxt))
        
                if self.compute_SSE :
                    self.SSE_t[ii] = (cum_norm_t - u.T @ Q @ u) 
                    self.dSSE_t[ii,:] =  np.append( - (u[:,0].T @ dQQu)[:,0],  2* sum(adj_invΛ* Q_u**2)) 
        
        #adjustment and rescaling
        dmu *= self.total_noise_variance
        sigma2 *= self.total_noise_variance
        dsigma2 *= self.total_noise_variance **2

        if self.noise_trainable is True :
            dsigma2[:,-1] +=  2 * sigma2
        
        if self.compute_SSE :
            self.SSE_t /= self.total_noise_variance
            
            if self.noise_trainable is True :
                self.dSSE_t[:,-1] = self.dSSE_t[:,-1] -  2 * noise_variance * self.SSE_t / self.total_noise_variance 
                
        # # adjust for change point i.e add prior distribution on top of the vector
        #mu[0] = self.SSE_t[0] = 0.0
        sigma2[0] = self.kernel.variance
        
                
        #Extend the mu prediction for the older (> MRC) run length hypothesis
        if MRC < t:  
            mu[MRC+1:] = mu[MRC] * np.ones((t - MRC))  # t - MRC x 1. [x]
            sigma2[MRC+1:] =  sigma2[MRC] * np.ones((t - MRC))
            
            if self.compute_SSE :
                self.SSE_t[MRC+1:] = self.SSE_t[MRC] * np.ones((t - MRC))
    
        return  (mu, dmu), (sigma2, dsigma2)
    



class RRSPARCP(RRCP_Prediction_Mixin, RR_SPBase_Mixin, SPCPBase):
    
    def __init__(self, kernel, p = 1, prior_parameter = 0.1, noise_parameter = 0.0):
        super().__init__(kernel, prior_parameter, noise_parameter)
        assert p > 0
        
        self.p = int(p)
        self.compute_SSE = True
        self.horizontal_update = True
        self.unit_beta = False
        
        if self.is_noise :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10
            

class RRSPARCP_no_noise(ARPredictionCPMixin, RR_SPBase_Mixin, SPCPBase):
    
    def __init__(self, kernel, p=1, prior_parameter=0.1):
        noise_parameter  = 0.0
        super().__init__(kernel, prior_parameter,  noise_parameter)
        assert p > 0
        
        self.p = int(p)
        self.unit_beta = False
        self._jitter = 1e-10
        
        self.compute_SSE = True
        self.SSE_t = 0
        self.dSSE_t = 0
        

    @property
    def total_noise_variance(self):
        return  self._jitter
        
    def precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)


class RRGPARCP(RRCP_Prediction_Mixin, RR_GPBase_Mixin, GPCPBase):
    
    def __init__(self, kernel, p = 1, noise_parameter = 0.0):
        super().__init__(kernel, noise_parameter)
        assert p > 0
        
        self.p = int(p)
        self.compute_SSE = False
        self.horizontal_update = True
        
        if self.is_noise :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10
            
            




    
    
    
    
    
    
    
    
    
    
    
    
    
    

