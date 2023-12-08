import numpy as np

from scipy.linalg import cholesky, solve_triangular
from bocpd.GPAR.arsplit import ARsplit
from bocpd.GPAR.cholesky import CholeskyMatrixDowndate, CholeskyMatrixUpdate
from bocpd.GPAR.arsplit import exchangeMatrix, differenceMatrixInverse
from bocpd.model_ts import SPTimeSerieBase, GPTimeSerieBase



class ARPredictionMixin():

    def prior(self):
        mu = 0.0
        sigma2 = self.kernel(self.lagMatrix[0].T) 
        return mu, sigma2[0][0]
    
    def prior_with_grads(self):
        
        mu = 0.0
        dmu = np.zeros(len(self.kernel.theta))

        (sigma2, dsigma2) = self.kernel(self.lagMatrix[0].T, eval_gradient = True)
        
        sigma2 = sigma2[0][0]
        dsigma2 = dsigma2[0][0]
        
        if self.noise_trainable :
            dsigma2 = np.append(dsigma2, 0.0)
            dmu = np.append(dmu, 0.0)
            
        return (mu, dmu), (sigma2, dsigma2)
    
    
    def prediction(self, t):
        
        if self.horizontal_update is True :
            
            if self.eval_gradient is False  :
                return self.__prediction_cholUpdate(t)
            else :
                return self.__prediction_cholUpdate_with_grads(t)
            
        else : 
            if self.eval_gradient is False  :
                return self.__prediction(t)
            else :
                return self.__prediction_with_grads(t)
            
            
    def __prediction(self, t):
        
         if t == 0 :
             
            return self.prior()
        
         MRC = self.MRC(t) #:= min(t, MaxLen)
            
         E = exchangeMatrix(MRC)
         xt = self.lagMatrix[t]
         yp = E @ self.X[t- MRC : t ]
         xp = np.atleast_2d(E @ self.lagMatrix[t- MRC:t,:,0])
         
         kt = self.kernel(xt.T)
         Kss =  self.kernel(xp)
         Ks = self.kernel( xp, xt.T)
         Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
         
         L =  cholesky(Kss, lower=True, check_finite=False) 
         alpha = solve_triangular(L, yp, lower=True, check_finite=False)
         v = solve_triangular(L, Ks, lower=True, check_finite=False)
         
         mu = (v.T @ alpha)
         sigma2 = (kt[0][0] - v.T @ v)

         if self.compute_SSE :
            self.SSE_t = np.atleast_3d(alpha.T @ alpha)[0][0]

         return np.atleast_2d(mu)[0][0], np.atleast_2d(sigma2)[0][0]

     
 
    def __prediction_with_grads(self, t):
        
        if t == 0 : 
            return self.prior_with_grads()
        
        MRC = self.MRC(t) #:= min(t, MaxLen)

        yp = exchangeMatrix(MRC) @ self.X[t- MRC : t ]
        xt = self.lagMatrix[t]
        
        (K, dK) = self.kernel(exchangeMatrix(MRC + 1) @ self.lagMatrix[t - MRC :t+1, :, 0], eval_gradient = True)
        kt = self.kernel(xt.T)
        Kss = K[1:,1:]
        Ks = np.expand_dims(K[0, 1:],1)
        Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
                
        dKss =  dK[1:, 1:, :]
        dKs = dK[0, 1:, :]
        #dkt = dK[-1,-1]
        dkt = dK[0,0]
        
        if self.noise_trainable :
            dKss = np.append(dKss, 2 * self._noise_scale**2 * np.atleast_3d(np.eye(MRC)), axis=2) #adjust to return grad wrt log(noise)
            dKs = np.hstack((dKs, np.zeros((MRC,1))))   
            dkt = np.append(dkt,0.0)
        
        L =  cholesky(Kss, lower=True, check_finite=False)   
        alpha = solve_triangular(L, yp, lower=True, check_finite=False)
        v = solve_triangular(L, Ks, lower=True, check_finite=False)
        
        # adjust for change point i.e add prior distribution on top of the vector
        mu = v.T @ alpha
        sigma2 = kt[0][0] - v.T @ v

        #compte dL following Sarkka (2013) i.e dL = L Φ(L^{-1} dK L^{-1}.T )
        Dinv = differenceMatrixInverse(MRC)
        F = Dinv - 0.5 * np.eye(MRC)
        Linv = solve_triangular(L, np.identity(MRC), lower=True, check_finite=False)

        dL = np.einsum('ijk,jl-> ilk', dKss, Linv.T)
        dL = Linv @ np.moveaxis(dL, -1, 0) 
        dL = np.expand_dims(F,0) * dL
        dL = L @ dL

        #compute dalpha
        tmp = np.atleast_3d(dL @ alpha)[:,:,0].T
        dalpha = - solve_triangular(L, tmp, lower=True, check_finite=False)
        
        #compute dv
        Linv_dKs = solve_triangular(L, dKs, lower=True, check_finite=False)
        tmp = np.moveaxis((dL @ v)[:,:,0], -1, 0) 
        dLinv_Ks = - solve_triangular(L, tmp, lower=True, check_finite=False)
        dv = dLinv_Ks  + Linv_dKs
        
        dmu = alpha.T @ dv + v.T @ dalpha
        dsigma2 = dkt - 2 *  v.T @ dv
        
        self.tmp = mu

        #compute_SSE with prior adjustment
        if self.compute_SSE :
            self.SSE_t =  np.atleast_2d(alpha.T @ alpha)[0][0] 
            self.dSSE_t = 2 * (alpha.T @ dalpha)[0]
        
        return (np.atleast_2d(mu)[0][0], dmu[0]), (np.atleast_2d(sigma2)[0][0], dsigma2[0])





    def __prediction_cholUpdate(self, t):
        
        if t ==0 : 
            return self.prior()

        xt = self.lagMatrix[t]
        
        yp = self.X[ : t ,0 ]
        xp = self.lagMatrix[:t,:,0]
        
        kt = self.kernel(xt.T)
        Kss = self.kernel(xp)
        Ks = self.kernel( xp, xt.T)

        Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
        
        if self.fast_computation :
            if t > 1 :
                new_col = Kss[:, -1]
                L = self.chol.update(t, new_col)
            else :
                L = np.sqrt(Kss)
                self.chol = CholeskyMatrixUpdate(L, len(self.X))
        else :
            L =  cholesky(Kss, lower=True, check_finite=False)

        alpha = solve_triangular(L, yp, lower=True, check_finite=False)
        v = solve_triangular(L, Ks, lower=True, check_finite=False)
        
        mu = (v.T @ alpha)
        sigma2 = (kt[0][0] - v.T @ v)
        
        if self.compute_SSE :
            self.SSE_t = alpha.T @ alpha
        
        return (mu, sigma2)
    
    
    def __prediction_cholUpdate_with_grads(self, t):

        if t == 0 : 
            return self.prior_with_grads()
        
        #n_kernel_params = len(self.kernel.theta)
        #xt = np.expand_dims(self.X[t-1],1)
        xt = self.lagMatrix[t]
        yp = self.X[ : t ,0]

        (K, dK) = self.kernel(self.lagMatrix[:t+1, :, 0], eval_gradient = True)
        kt = self.kernel(xt.T)
        Kss = K[:t, : t]
        Ks = K[-1, :-1]
        Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
        
        #dKss => (t) x (t) x kernel.n_params
        dKss =  dK[:t, :t, :]
        dKs = dK[-1, :-1, :]
        dkt = dK[-1,-1]

        if self.noise_trainable :
            dKss = np.append(dKss, 2 * self._noise_scale**2 * np.atleast_3d(np.eye(t)), axis=2) #adjust to return grad wrt log(noise)
            dKs = np.hstack((dKs, np.zeros((t,1))))   
            dkt = np.append(dkt,0.0)
        
        if self.fast_computation :
            if t > 1 :
                new_col = Kss[:, -1]
                L = self.chol.update(t, new_col)
            else :
                L = np.sqrt(Kss)
                self.chol = CholeskyMatrixUpdate(L, len(self.X))
        else :
            L =  cholesky(Kss, lower=True, check_finite=False)
            
        alpha = solve_triangular(L, yp, lower=True, check_finite=False)
        v = solve_triangular(L, Ks, lower=True, check_finite=False)
        
        mu = v.T @ alpha
        sigma2 = kt[0][0] - v.T @ v
        

        #compte dL following Sarkka (2013) i.e dL = L Φ(L^{-1} dK L^{-1}.T )
        Linv = solve_triangular(L, np.identity(t), lower=True, check_finite=False)
        dL = np.einsum('ijk,jl-> ilk', dKss, Linv.T)
        dL = Linv @ np.moveaxis(dL, -1, 0) 
        dL = np.expand_dims(np.tri(t)- 0.5 * np.eye(t),0) * dL
        dL = L @ dL
        
        #compute dalpha
        tmp = np.moveaxis((dL @ alpha), -1, 0) 
        dalpha = - solve_triangular(L, tmp, lower=True, check_finite=False)
        
        #compute dv
        Linv_dKs = solve_triangular(L, dKs, lower=True, check_finite=False)
        tmp = np.moveaxis((dL @ v), -1, 0) 
        dLinv_Ks = - solve_triangular(L, tmp, lower=True, check_finite=False)
        dv = dLinv_Ks  + Linv_dKs
        
        dmu = alpha.T @ dv + v.T @ dalpha
        dsigma2 = dkt - 2 *  v.T @ dv
        
        if self.compute_SSE :
            self.SSE_t = alpha.T @ alpha
            self.dSSE_t = 2 *  alpha.T @ dalpha
            
        return (mu, dmu), (sigma2, dsigma2)
        
        
    # def __prediction_cholUpdate_with_grads(self, t):

    #     if t == 0 : 
    #         return self.prior_with_grads()
        
    #     n_kernel_params = len(self.kernel.theta)
        
    #     #xt = np.expand_dims(self.X[t-1],1)
    #     xt = self.lagMatrix[t]
    #     yp = self.X[ : t ]

    #     (K, dK) = self.kernel(self.lagMatrix[:t+1, :, 0], eval_gradient = True)
    #     kt = self.kernel(xt.T)
    #     Kss = K[:t, : t]
    #     Ks = K[-1, :-1]
    #     Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
        
    #     #dKss => (t) x (t) x kernel.n_params
    #     dKss =  dK[:t, :t, :]
    #     dKs = dK[-1, :-1, :]
    #     dkt = dK[-1,-1]

    #     if self.noise_trainable :
    #         dKss = np.append(dKss, 2 * self._noise_scale**2 * np.atleast_3d(np.eye(t)), axis=2) #adjust to return grad wrt log(noise)
    #         dKs = np.hstack((dKs, np.zeros((t,1))))   
    #         dkt = np.append(dkt,0.0)
        
    #     if self.fast_computation :
    #         if t > 1 :
    #             new_col = Kss[:, -1]
    #             L = self.chol.update(t, new_col)
    #         else :
    #             L = np.sqrt(Kss)
    #             self.chol = CholeskyMatrixUpdate(L, len(self.X))
    #     else :
    #         L =  cholesky(Kss, lower=True, check_finite=False)

    #     #mu
    #     alpha = cho_solve((L, True), yp, check_finite=False,) # Rasmussen Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
    #     mu = (Ks.T @ alpha)[0]

    #     dmu = np.zeros((n_kernel_params ,1))
    #     for ii in range(n_kernel_params ):
    #           tmp = dKss[:,:, ii] @ alpha
    #           tmp = cho_solve((L, True), tmp, check_finite=False,)
    #           dmu[ii] = - Ks.T @ tmp
        
    #     dmu += dKs.T @ alpha
    #     dmu = dmu[:,0]
        
    #     if self.noise_trainable :
    #         tmp =  2 * self._noise_scale * alpha
    #         tmp = cho_solve((L, True), tmp, check_finite=False,)
    #         dmu_n = - Ks.T @ tmp
    #         #adjust to return grad wrt log(noise)
    #         dmu = np.append(dmu, self._noise_scale * dmu_n )
        
    #     #sigma
    #     v = solve_triangular(L, Ks, lower=True, check_finite=False)
    #     sigma2 = (kt - v.T @ v)[0][0]
        
    #     dsigma2 = np.zeros((n_kernel_params  ,1))
    #     u = cho_solve((L, True), Ks, check_finite=False,)
    #     for ii in range(n_kernel_params  ):
    #         dsigma2[ii] = u.T @ dKss[:,:, ii] @ u
        
    #     dsigma2 -= 2 * np.expand_dims(dKs.T @ u,1)
    #     dsigma2 +=  np.expand_dims(dkt,1)
    #     dsigma2= dsigma2[:,0]

    #     if self.noise_trainable :
    #         dsigma2_n = 2 * self._noise_scale * u.T @ u
    #         #adjust to return grad wrt log(noise)
    #         dsigma2 = np.append(dsigma2, self._noise_scale * dsigma2_n)

    #     return (mu, dmu), (sigma2, dsigma2)

    
    # def __prediction_cholDownpdate(self, t):
    #     #TO DO check the instability of the Cholesky downdate 
    #     #Not updated for LagMatrix
        
    #     if t < 2 : 
    #         return self.prior()
        
    #     MRC = self.MRC(t)
    #     xt = np.expand_dims(self.X[t-1],1)

    #     yp = self.X[t - MRC + 1 : t ][::-1]
    #     k0 = self.kernel(xt.T)
    #     Kss = self.kernel(self.X[t - MRC : t - 1][::-1])
    #     Ks = self.kernel( self.X[t - MRC : t - 1], xt)
    #     Kss[np.diag_indices_from(Kss)] += self.total_noise_variance
        
    #     if MRC < t :
    #         self._cholDowndate.prune(MRC - 2)
            
    #     if self.fast_computation :
    #         if t > 2 :
    #             new_col = Kss[:, 0]
    #             if MRC < t :
    #                 self._cholDowndate.prune(MRC - 2)
    #             L = self._cholDowndate.update(t, new_col)
    #         else :
    #             L = np.sqrt(Kss)
    #             self._cholDowndate = CholeskyMatrixDowndate(L, len(self.X))
    #     else :
    #         L =  cholesky(Kss, lower=True, check_finite=False)
        
    #     alpha = cho_solve((L, True), yp, check_finite=False,) # Rasmussen Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
    #     mu = (Ks.T @ alpha[::-1])[0]

    #     v = solve_triangular(L, Ks[::-1], lower=True, check_finite=False)
    #     sigma2 = (k0 - v.T @ v)[0][0]
        
    #     return mu, sigma2
    
    
   
        

class GPAR( ARPredictionMixin, GPTimeSerieBase):
    
    def __init__(self, kernel, p = 1, noise_parameter = 0.1):
        super().__init__(kernel, noise_parameter)
        assert p > 0
        self.p = int(p)
        self._jitter = 0.0
        self.compute_SSE = False
        self.horizontal_update = False

    @property
    def total_noise_variance(self):
        return   self._noise_scale**2 + self._jitter
        
    def precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)
        

class SPAR( ARPredictionMixin, SPTimeSerieBase):
    
    def __init__(self, kernel, p = 1, prior_parameter = 0.1, noise_parameter = 0.0):
        super().__init__(kernel, prior_parameter, noise_parameter)
        assert p > 0
        self.p = int(p)
        self.horizontal_update = False
        
        if self.is_noise :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10

        self.compute_SSE = True
        
    @property
    def total_noise_variance(self):
        return self._noise_scale**2 + self._jitter
        
        
    def precompute(self):
         self.lagMatrix = ARsplit(self.X, self.p)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    