
import numpy as np

from bocpd.test.GP_base_tf import STPBase_TF, GPBase_TF, TensorMisc
from bocpd.test.GP_base_tf import logexp_space_transformation, logexp_space_inverse_transformation
from bocpd.test.GPAR_tf import  AR_CP_Prediction_Mixin

from bocpd.GPAR.arsplit import ARsplit

import tensorflow as tf
from gpflow.config import default_float

import tensorflow_probability as tfp
tfd = tfp.distributions




class GaussianLaplaceReducedRank_TF():
    
    def __init__(self, variance, lengthscales , n_features = 10, n_dimension = 1, L = 5):
        self.n_features = n_features
        self.n_dimension = n_dimension
        
        self._L = L
        self._logexp_lengthscales = tf.Variable(logexp_space_transformation(lengthscales), dtype =default_float())
        self._logexp_variance = tf.Variable(logexp_space_transformation(variance), dtype =default_float())
        self._sqrt_eigenval = self.j *np.pi/2/ self._L
        

    @property
    def _variance(self) :
        return logexp_space_inverse_transformation(self._logexp_variance )
 
    @property
    def _lengthscales (self) :
      return logexp_space_inverse_transformation(self._logexp_lengthscales)


    @property
    def j(self) :
        j = np.expand_dims(np.array(range(1, self.n_features + 1)),1)[:,0]

        jnd = np.zeros((self.n_features**self.n_dimension, self.n_dimension))
        for ii in range(self.n_dimension):
            jnd[:,ii] = np.tile(np.repeat(j, self.n_features**(self.n_dimension-1-ii)), self.n_features**(ii)) 
        
        return jnd


    def feature(self, X):
        """ Transforms the data X (n_samples, n_dimension) 
        to feature map space Z(X) (n_samples, n_features)"""
        tmp = np.expand_dims(X + self._L, 1) * np.expand_dims(self._sqrt_eigenval , 0)
        features = self._L**(-self.n_dimension/2) * np.prod(np.sin(tmp),2)
        return features


    def __call__(self, X, X2 = None):
        if X2 is None :
            Z = self.feature(X)
            return Z @ tf.linalg.diag(self.Λ())  @ tf.transpose(Z)
        return  self.feature(X)  @ tf.linalg.diag(self.Λ())  @  tf.transpose(self.feature(X2))
    
    def S(self, v) :
        tmp = np.sqrt(2 * np.pi) *  self._lengthscales * tf.math.exp( -0.5 * (v * self._lengthscales)**2)
        return self._variance  * tf.math.reduce_prod(tmp,1)
    
    def Λ(self):
        return self.S(self._sqrt_eigenval) #[:,0]

    
    def invΛ(self):
        return 1 / self.Λ()

    
    @property
    def parameters(self):
        out = np.array( self._variance)
        out = np.append(out, self._lengthscales)
        return  out

    @property
    def trainable_parameters(self):
            out = (self._logexp_variance,) + (self._logexp_lengthscales,)
            return out
        
    @property
    def trainable_variables(self):
        return self.trainable_parameters
    
        

class ReducedRankMixin():
    
    @property
    def v_inv_lamb(self):
        return  self.total_noise_variance *  tf.linalg.diag(self.kernel.invΛ())
    
          
    def initialize(self):
        self.t = 0
        self.SSE_t = 0
        self._precompute()
        
    def _precompute(self):
        self.lagMatrix = ARsplit(self.X, self.p)
        self.Phi = self.kernel.feature(self.lagMatrix[:,:,0]) 
        
    @property
    def total_noise_variance(self):
        return  self._noise_scale**2 + self._jitter
    
    def prior(self):
        mu = 0.0
        sigma2 = self.kernel._variance
        return mu, sigma2
    
    def run(self):

        self.computeGradient(False)
        self.initialize()
        
        (T,D) = self.X.shape
        Z = tf.zeros([1, 1], dtype= default_float())
        
        sigma = mu = np.zeros((1, 1))

        for t in range(T) :
            mu_tf, sigma_tf = self.prediction(t)
            Z = tf.experimental.numpy.vstack([Z, self.logpdf(t)])
            mu = np.append(mu, mu_tf)
            sigma = np.append(sigma, sigma_tf)
            self.update(t)

        nlml = - tf.math.reduce_sum(Z, 0)
        return nlml, Z[1:].numpy(), (mu, sigma)
    
    
    
         
class RRGPAR_TF(ReducedRankMixin, GPBase_TF):
    
    def __init__(self, kernel, p, noise_parameter = 0.1):
        super().__init__(kernel, noise_parameter )
        self._jitter = 0.0
        self.p = int(p)
        self.noise_trainable = True
        


    def prediction(self, t):

        if t ==0 : 
            return self.prior()
        
        MRC = self.MRC(t)

        #xt = self.lagMatrix[t]
        yp = self.X[t- MRC  : t ]
        #xp = self.lagMatrix[:t,:,0]

        pxt = tf.expand_dims(self.Phi[t,:],1)
        u = tf.transpose(self.Phi[t- MRC :t,:]) @ yp
        Q = tf.linalg.inv(tf.transpose(self.Phi[t- MRC  :t,:]) @ self.Phi[t- MRC :t,:] + self.v_inv_lamb) 
        
        self.Q = Q
        mu  = tf.transpose(pxt) @ Q @ u
        sigma2  = self.total_noise_variance * tf.transpose(pxt)  @ Q @ pxt
 
        return (mu[0], sigma2)

    
    def pdf(self, t):
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        self.t = t

        mu, sigma2 = self.prediction(t)
        pred_scale = tf.math.sqrt(sigma2 + self._noise_scale**2)
        predprobs = tfp.distributions.Normal(mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
    
    def update(self, t):
        pass
    



class RRSPAR_TF(ReducedRankMixin, STPBase_TF):

    def __init__(self, kernel, p, prior_parameter = 0.1, noise_parameter = 0.1):
        super().__init__(kernel, prior_parameter, noise_parameter)
        self._jitter = 0.0
        self.p = int(p)
        self.noise_trainable = True


    def prediction(self, t):

        if t ==0 : 
            return self.prior()
        
        MRC = self.MRC(t)

        #xt = self.lagMatrix[t]
        yp = self.X[t- MRC  : t ]
        #xp = self.lagMatrix[:t,:,0]

        pxt = tf.expand_dims(self.Phi[t,:],1)
        u = tf.transpose(self.Phi[t- MRC :t,:]) @ yp
        Q = tf.linalg.inv(tf.transpose(self.Phi[t- MRC  :t,:]) @ self.Phi[t- MRC :t,:] + self.v_inv_lamb) 
        
        self.Q = Q
        mu  = tf.transpose(pxt) @ Q @ u
        sigma2  = self.total_noise_variance * tf.transpose(pxt)  @ Q @ pxt
        
        self.SSE_t = ( tf.transpose(yp) @ yp - tf.transpose(u) @ Q @ u) / self.total_noise_variance
 
        return (mu[0], sigma2)

    
    def pdf(self, t):
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        self.t = t
        
        df = 2 * self._alpha0 + self.MRC(t)
        mu, sigma2 = self.prediction(t)

        SSE = 2 * self._beta0 + self.SSE_t
        pred_var = sigma2 * SSE / df
        pred_scale = tf.math.sqrt(pred_var)
        predprobs = tfp.distributions.StudentT(df, mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
    
    
    def update(self, t):
        pass
    


class RR_CP_Prediction_Mixin():

     def prediction(self, t):
            if t ==0 : 
                return self.prior()
            
            MRC = self.MRC(t)
    
            yp = self.X[t- MRC  : t ]
            pxt = tf.expand_dims(self.Phi[t,:],1)
    
            ########## 
            Phi = self.Phi[:t,:]
            phi = tf.expand_dims(self.Phi[t-1, :], 1)
            cum_norm = yp[-1]**2
            u = yp[-1]  * phi
            Q = tf.linalg.inv(phi @ tf.transpose(phi) +  self.v_inv_lamb)
            
            mu =  tf.transpose(pxt)  @ Q @ u
            sigma2 =  self.total_noise_variance * tf.transpose(pxt) @ Q @ pxt
            self.SSE_t = (cum_norm - tf.transpose(u) @ Q @ u)
    
            for ii in range(2, MRC +1) :
                phi = np.expand_dims(self.Phi[t-ii, :], 1)
                u = tf.transpose(Phi[-ii:]) @ yp[-ii:]
                cum_norm +=  yp[-ii]**2
                Q = tf.linalg.inv(tf.transpose(Phi[-ii:]) @ Phi[-ii:] + self.v_inv_lamb)

                mu  = tf.concat((mu, tf.transpose(pxt) @ Q @ u), axis=0) 
                sigma2  = tf.concat((sigma2, self.total_noise_variance * tf.transpose(pxt) @ Q @ pxt), axis=0)  
                self.SSE_t  = tf.concat((self.SSE_t, cum_norm - tf.transpose(u) @ Q @ u ), axis=0)   
                  
            # adjust for change point i.e add prior distribution on top of the vector
            #mu[0] = self.SSE_t[0] = 0.0
            zero_tf = tf.zeros((1,1),  dtype =default_float())
            mu  = tf.concat(( zero_tf , mu), axis=0) 
            sigma2  = tf.concat((self.kernel._variance * tf.ones((1,1),  dtype =default_float()), sigma2), axis=0) 
            self.SSE_t   = tf.concat(( zero_tf , self.SSE_t ), axis=0) 
            self.SSE_t = self.SSE_t / self.total_noise_variance
                
            if MRC < t :
                mu = tf.experimental.numpy.vstack([mu, mu[-1] * tf.ones((t - MRC,1), dtype = default_float() )])
                sigma2 = tf.experimental.numpy.vstack([sigma2, sigma2[-1] * tf.ones((t - MRC,1), dtype = default_float() )])
                self.SSE_t = tf.experimental.numpy.vstack([self.SSE_t, self.SSE_t[-1] * tf.ones((t - MRC,1), dtype = default_float() )])
    
            return (mu, sigma2)
        
    
        


class RRGPARCP_TF(ReducedRankMixin,  RR_CP_Prediction_Mixin, GPBase_TF):
    
    def __init__(self, kernel, p, noise_parameter ):
        super().__init__(kernel, noise_parameter )
        self._jitter = 0.0
        self.p = int(p)

    def pdf(self, t):
        
        self.t = t
        
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        mu, sigma2 = self.prediction(t)
        pred_scale = tf.math.sqrt(sigma2 + self._noise_scale**2)
        predprobs = tfp.distributions.Normal(mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
    
    def update(self, t):
        pass
    


class RRSPARCP_TF(ReducedRankMixin,  RR_CP_Prediction_Mixin, STPBase_TF):
    
    def __init__(self, kernel, p, prior_parameter , noise_parameter  = 0.0):
        super().__init__(kernel, prior_parameter , noise_parameter )
        self.p = int(p)
        
        if noise_parameter  != 0.0 :
            self._jitter = 0.0
        else :
            self._jitter = 1e-10


    def pdf(self, t):
        
        self.t = t
        
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        t_alpha = tf.Variable(self.grid[ : t + 1], dtype = default_float() )
        df = 2 * self._alpha0 + t_alpha
        mu, sigma2 = self.prediction(t)
        SSE = 2 * self._beta0 + self.SSE_t
        pred_var = sigma2 * SSE / df
        pred_scale = tf.math.sqrt(pred_var)
        predprobs = tfp.distributions.StudentT(df, mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
 
    
    def update(self, t):
        assert self.t == t
        pass
    
    
class RRSPARCP_NO_NOISE_TF(ReducedRankMixin, AR_CP_Prediction_Mixin , STPBase_TF):
    
    def __init__(self, kernel, p, prior_parameter):
        super().__init__(kernel, prior_parameter , 0.0 )
        self.p = int(p)
        self._jitter = 1e-10
        self.noise_trainable = False 
        
    @property
    def total_noise_variance(self):
        return  self._jitter


    def pdf(self, t):
        
        self.t = t
        
        # compute the predictive probabilities at step t i.e P(X_t | X_{s:t-1})
        t_alpha = tf.Variable(self.grid[ : t + 1], dtype = default_float() )
        df = 2 * self._alpha0 + t_alpha
        mu, sigma2 = self.prediction(t)
        SSE = 2 * self._beta0 + self.SSE_t
        pred_var = sigma2 * SSE / df
        pred_scale = tf.math.sqrt(pred_var)
        
        self.sigma2 = sigma2
        self.SSE = SSE
        self.df = df
        
        self.pred_var =  pred_var
        self.pred_scale =  pred_scale
        self.mu = mu
        self.point = self.X[t, :]
        predprobs = tfp.distributions.StudentT(df, mu, pred_scale, validate_args=False, allow_nan_stats=True).prob(self.X[t, :])

        return predprobs
 
    
    def update(self, t):
        assert self.t == t
        pass






if __name__ == '__main__':
    
    from bocpd.RRGPAR.RRGPAR import RRSPAR, RRGPAR
    from bocpd.RRGPAR.RRGPARCP import RRSPARCP_no_noise
    from bocpd.GPAR.arsplit import ARsplit, exchangeMatrix, differenceMatrixInverse
    from scipy.linalg import cholesky, solve_triangular
    
    import gpflow.kernels as gfk
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    import bocpd.generate_data as gd
    
    from sklearn import preprocessing
    
    from bocpd.RRGPAR.gaussian_laplace_reduced_rank import GaussianLaplaceReducedRank, GaussianLaplaceReducedRankND
    
    from bocpd.Utils.studentpdf import  studentpdf, studentlogpdf
    from bocpd.Utils.gaussianpdf import gaussianpdf, gaussianlogpdf
    from bocpd.GPAR.arsplit import ARsplit, exchangeMatrix
    
    from bocpd.GPAR.GPARCP import GPARCP
    from bocpd.test.GPAR_tf import GPARCP_TF

    partition, data = gd.generate_normal_time_series(7, 50, 200)
    data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    data = data[ : 100,]
    
    n_dimension = 1
    variance = 2
    lengthscales = np.array([0.5])
    prior_parameter = 0.1
    X = data
    
    variance = 2
    lengthscales = 0.5
    k_tf = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
    jitter = 0.0
    scale_noise = 0.0
  
    #MODEL1 = GPARCP_TF
    m_tf = GPARCP_TF(k_tf, 1, scale_noise)
    m_tf.setData(data)
    m_tf._jitter = jitter

    #MODEL2 =  GPARCP
    k = ConstantKernel(variance) * RBF(length_scale=lengthscales )
    m = GPARCP(k, 1, scale_noise)
    m.setData(data)
    m._jitter = jitter
    m.computeGradient()
    
    lrgp_tf = GaussianLaplaceReducedRank_TF(variance, lengthscales , n_features = 5, n_dimension = 1, L = 3)
    lrgp = GaussianLaplaceReducedRank(variance, lengthscales , n_features = 5, L = 3).fit()

    m_tf = RRSPARCP_NO_NOISE_TF(lrgp_tf,  n_dimension, prior_parameter)
    m_tf.setData(X) 
    m_tf.unit_beta = False
    m_tf.initialize()
    
    # m = RRGPAR(lrgp, n_dimension, noise_parameter)
    m = RRSPARCP_no_noise(lrgp,  n_dimension, prior_parameter)
    
    m.setData(X) 
    m.computeGradient(True)
    m.fast_computation = False
    m.unit_beta = False
    m.initialize()

    t = MRC = 10
    ii = 5

    #prediction
    out = m.prediction(t)
    print(out[1][0][ii])
    print(out[1][1][ii,:])
    print("")

    with tf.GradientTape() as tape:
        m_tf.initialize()
        mu_tf, sigma_tf  = m_tf.prediction(t)
        out_tf = sigma_tf[ii]
    dout_tf = tape.gradient(out_tf, m_tf.trainable_parameters)
    dout_tf = m_tf.grad_adjustment_factor[:-1] *  TensorMisc().pack_tensors( dout_tf[:-1])
    #dout_tf = m_tf.grad_adjustment_factor *  TensorMisc().pack_tensors( dout_tf)
    print(out_tf)
    print(dout_tf)
    print("")
    
    #logpdf
    logpdf = m.logpdf(t)
    print(logpdf[0][ii])
    print(logpdf[1][ii,:])
    print("")

    with tf.GradientTape() as tape:
        m_tf.initialize()
        logpdf_tf = m_tf.logpdf(t)[ii]
    dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
    dlogpdf_tf = m_tf.grad_adjustment_factor *  TensorMisc().pack_tensors( dlogpdf_tf)
    print(logpdf_tf)
    print(dlogpdf_tf)
    
    
    ######################################################
    yp = exchangeMatrix(MRC) @ m.X[t- MRC : t ]
    xt = m.lagMatrix[t]
        
    (K, dK) = m.kernel(exchangeMatrix(MRC + 1) @ m.lagMatrix[t - MRC :t+1, :, 0], eval_gradient = True)
    kt = m.kernel(xt.T)
    Kss = K[1:,1:]
    Ks = np.expand_dims(K[0, 1:],1)
    Kss[np.diag_indices_from(Kss)] += m.total_noise_variance
    
    dKss =  dK[1:, 1:, :]
    dKs = dK[0, 1:, :]
    dkt = dK[0,0]
    
    #dkt = np.array([ 0.37634487, -0.02190749]) 

    L =  cholesky(Kss, lower=True, check_finite=False)    
    alpha = solve_triangular(L, yp, lower=True, check_finite=False)
    v = solve_triangular(L, Ks, lower=True, check_finite=False)
        
    Dinv = differenceMatrixInverse(MRC)
    mu = Dinv @ (alpha * v)
    sigma2 = kt - Dinv @ (v * v)
        
    #compte dL following Sarkka (2013) i.e dL = L Φ(L^{-1} dK L^{-1}.T )
    F = Dinv - 0.5 * np.eye(MRC)
    Linv = solve_triangular(L, np.identity(MRC), lower=True, check_finite=False)

    dL = np.einsum('ijk,jl-> ilk', dKss, Linv.T)
    dL = Linv @ np.moveaxis(dL, -1, 0) 
    dL = np.expand_dims(F,0) * dL
    dL = L @ dL

    #compute dalpha
    #tmp = np.moveaxis((dL @ alpha)[:,:,0], -1, 0) 
    tmp = np.atleast_3d(dL @ alpha)[:,:,0].T
    tmp =  solve_triangular(L, tmp, lower=True, check_finite=False)
    dalpha = - tmp
        
    #compute dv
    Linv_dKs = solve_triangular(L, dKs, lower=True, check_finite=False)
    tmp = np.moveaxis((dL @ v)[:,:,0], -1, 0) 
    dLinv_Ks = - solve_triangular(L, tmp, lower=True, check_finite=False)
    dv = dLinv_Ks  + Linv_dKs
        
    dmu = Dinv @ (alpha * dv + v * dalpha)
    dsigma2 = dkt - 2 * Dinv @ (v * dv)
    
    # adjust for change point i.e add prior distribution on top of the vector
    mu = np.insert(mu, 0, 0)
    sigma2 = np.insert(sigma2, 0, kt)
        
    zero_np = np.zeros((dmu.shape[1]))
    dmu = np.insert(dmu, 0, zero_np, axis = 0)
    dsigma2 = np.insert(dsigma2, 0, dkt, axis = 0)
    
    
    with tf.GradientTape() as tape:
        
        m_tf.initialize()
        
        Dinv = tf.experimental.numpy.tri(MRC, dtype = default_float())
        E = tf.convert_to_tensor(np.atleast_2d(exchangeMatrix(MRC)), dtype = default_float())

        xt_tf = m_tf.lagMatrix[t]
        yp_tf = E @  m_tf.X[t- MRC : t ]
        xp_tf = E @  m_tf.lagMatrix[t- MRC:t,:,0]
 
        ks_tf =  m_tf.kernel(xp_tf, tf.transpose(xt_tf))
        kss_tf =  m_tf.kernel(xp_tf)
        kss_tf += ( m_tf._jitter +  m_tf._noise_scale**2) * tf.eye(kss_tf.shape[0], dtype = default_float() )
        
        L_tf = tf.linalg.cholesky(kss_tf)
        alpha_tf = tf.linalg.triangular_solve(L_tf, yp_tf, lower=True)
        v_tf = tf.linalg.triangular_solve(L_tf, ks_tf, lower=True)

        k0_tf = m_tf.kernel(tf.transpose(xt_tf))
        mu_tf = Dinv @ (v_tf * alpha_tf)
        sigma2_tf = k0_tf - Dinv @ (v_tf * v_tf)
        
        zeros_tf = tf.zeros([1,1], dtype = default_float())
        mu_tf = tf.experimental.numpy.vstack([zeros_tf, mu_tf ])
        sigma2_tf  = tf.experimental.numpy.vstack([k0_tf , sigma2_tf ])
        
        out_tf = sigma2_tf[ii]
        out_tf = v_tf[0]
        out_tf = k0_tf
        #out_tf = alpha_tf[0]
        #out_tf = mu_tf[1]

    dout_tf = tape.gradient(out_tf, m_tf.trainable_parameters)
    dout_tf = m_tf.grad_adjustment_factor[:-1] *  TensorMisc().pack_tensors( dout_tf[:-1])
    #dout_tf = m_tf.grad_adjustment_factor*  TensorMisc().pack_tensors( dout_tf)


    


    
    
    
 














   