
import numpy as np

from bocpd.model_base import LogExpSpaceTransformer
from bocpd.GPTS.GPTS import GPTS
from bocpd.test.GPTS_tf import GPTS_TF
import bocpd.generate_data as gd

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor

import tensorflow as tf
import gpflow.kernels as gfk

import tensorflow_probability as tfp
tfd = tfp.distributions

import unittest




def switch(a):
     return a[::-1]


class Test_GPTS(unittest.TestCase):

    def setUp(self) :
     
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-500 :,]
        
        grid = (np.atleast_2d(range(len(data))) * 0.5).T

        variance = 2
        lengthscales = 0.5
        k1 = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
        
        #MODEL1 = GPTS_TF
        m_tf = GPTS_TF(k1)
        m_tf.setData(data, grid)

        #MODEL2 = GPTS
        k2 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m2 = GPTS(k2)
        m2.setData(data, grid)
        m2.computeGradient()
        
        #MODEL3 = GPTS with LogExp transforn gradient
        k3 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m3 = GPTS(k3)
        m3.setData(data, grid)
        m3.computeGradient()
        m3.transformer = LogExpSpaceTransformer()
        
        self.m_tf = m_tf
        self.m2 = m2
        self.m3 = m3

   
    ###################################################
    #@unittest.SkipTest
    def test_global_logspace(self):
        
        m_tf = self.m_tf
        m2 = self.m2

        factor = m_tf.grad_adjustment_factor
        with tf.GradientTape() as tape:
            nlml_tf, Z = m_tf.run()
        dnlml_tf = tape.gradient(nlml_tf, m_tf.trainable_parameters)
        dnlml_tf = np.array(dnlml_tf) * factor
        dnlml_tf = switch(dnlml_tf)

        nlml, dnlml, _, _ = m2.run_with_gradients()
        
        print("")
        print("GLOBAL_TF_VS_NP_logGrad")
        print("tf_nlml:= : {}".format(np.array(nlml_tf)))
        print("np_nlml:= : {}".format(nlml))
        print("")
        print("tf_grad:= : {}".format(np.array(dnlml_tf)))
        print("np_grad:= : {}".format(dnlml))
        
    ###################################################
    #@unittest.SkipTest
    def test_global_logexpspace(self):
        
        m_tf = self.m_tf
        m3 = self.m3

        with tf.GradientTape() as tape:
            nlml_tf, Z = m_tf.run()
        dnlml_tf = tape.gradient(nlml_tf, m_tf.trainable_parameters)
        dnlml_tf= switch(dnlml_tf)

        nlml, dnlml, _, _ = m3.run_with_gradients()
        
        print("")
        print("GLOBAL_TF_VS_NP_logExpGrad")
        print("tf_nlml:= : {}".format(np.array(nlml_tf)))
        print("np_nlml:= : {}".format(nlml))
        print("")
        print("tf_grad:= : {}".format(np.array(dnlml_tf)))
        print("np_grad:= : {}".format(dnlml))


    ###################################################
    @unittest.SkipTest
    def test_dA(self):
        
        m_tf = self.m_tf
        m2 = self.m2
        factor = m_tf.grad_adjustment_factor
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            out = tf.reduce_sum(tf.reduce_sum(m_tf.A,0),0)
        g1 = tape.gradient(out, m_tf.trainable_parameters)
        g1 = np.array(g1) * factor
        g1 = switch(g1)
        
        m2.initialize()
        g2 =  sum(sum(m2.dA))
        
        print("")
        print("TEST_dA")
        print("tf_grad:= : {}".format(np.array(g1)))
        print("np_grad:= : {}".format(g2))
    
    # ################################################
    @unittest.SkipTest
    def test_dSig2(self):
        
        m_tf = self.m_tf
        m2 = self.m2
        factor = m_tf.grad_adjustment_factor
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            out = tf.reduce_sum(tf.reduce_sum(m_tf.Sig2,0),0)
        g = tape.gradient(out, m_tf.trainable_parameters)
        g1 = np.array(g) * factor
        g1 = switch(g1)
        
        m2.initialize()
        g2 =  sum(m2.dSig2)
        
        print("")
        print("TEST_dSig2")
        print("tf_grad:= : {}".format(np.array(g1)))
        print("np_grad:= : {}".format(g2))
        
        
  
    ###################################################
    @unittest.SkipTest
    def test_logpdf_0(self):
    # # TEST logpdf(0)
        m_tf = self.m_tf
        m2 = self.m2
        factor = m_tf.grad_adjustment_factor
    
        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(0)
        dlogpdf_tf  = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf ) * factor
        dlogpdf_tf  = switch(dlogpdf_tf)
        
        #(K, dK) = m2.kernel(grid, eval_gradient = True)
        m2.initialize()
        logpdf, dlogpdf = m2.logpdf(0)
        
        print("")
        print("TEST_logpdf(0)")
        print("tf_logpdf(0):= : {}".format(np.array(logpdf_tf[0] )))
        print("np_logpdf(0):= : {}".format(np.array(logpdf[0] )))
        
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf )))
        print("np_grad:= : {}".format(dlogpdf))

    
    ###################################################
    #@unittest.SkipTest
    # TEST n-logpdf UPDATE
    def test_logpdf_n(self):
        
        m_tf = self.m_tf
        m2 = self.m2
        factor = m_tf.grad_adjustment_factor
        
        n = 20
        with tf.GradientTape() as tape:
            m_tf.initialize()
            for i in range(n):
                logpdf_tf = m_tf.logpdf(i)
                m_tf.update(i)
                
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf= np.array(dlogpdf_tf) * factor
        dlogpdf_tf = switch(dlogpdf_tf)
    
        m2.initialize()
        for i in range(n):
            logpdf, dlogpdf = m2.logpdf(i)
            m2.update(i)
            
        print("")
        print("TEST_logpdf(n)")
        print("tf_logpdf(n):= : {}".format(np.array(logpdf_tf[0][0])))
        print("np_logpdf(n):= : {}".format(np.array(logpdf[0])))
        
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        
    
    ###################################################
    #@unittest.SkipTest
    # TEST n-logpdf UPDATE
    def test_logpdf_n_logexp(self):
        
        m_tf = self.m_tf
        m3 = self.m3
        
        n = 20
        with tf.GradientTape() as tape:
            m_tf.initialize()
            for i in range(n):
                logpdf_tf = m_tf.logpdf(i)
                m_tf.update(i)
                
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf = switch(dlogpdf_tf)
    
        m3.initialize()
        for i in range(n):
            logpdf, dlogpdf = m3.logpdf(i)
            m3.update(i)
            
        print("")
        print("TEST_logpdf(n)_logexp")
        print("tf_logpdf(n):= : {}".format(np.array(logpdf_tf[0][0])))
        print("np_logpdf(n):= : {}".format(np.array(logpdf[0])))
        print("")
        
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        
        
    
    ###################################################
    # TEST n-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m2 = self.m2

        factor = m_tf.grad_adjustment_factor
        m_tf.fast_computation = False

        n = 25
        
        #TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, _ = m_tf.tf_prediction(n)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        dmu_tf = dmu_tf * factor
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.tf_prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf * factor
        dsig_tf = switch(dsig_tf)
        
        #Fast calculation
        m2.initialize()
        for i in range(n + 1):
            (mu1, dmu1), (sig1, dsig1) = m2.prediction(i)

        #Scipy recompute
        xt = m2.grid[n]
        yp = m2.X[ : n ]
        xp = m2.grid[: n]
        gpr = GaussianProcessRegressor(kernel= m2.kernel, optimizer = None, alpha= 0.0, copy_X_train = False).fit(xp, yp)
        scp_mu, scp_sig = gpr.predict(xt.reshape(1, -1), return_std=True)
        scp_sig = scp_sig**2

        print("")
        print("TEST_prediction("+ str(n) + ")")
        
        print("scp_mu(n):= : {}".format(scp_mu[0][0]))
        print("tf_mu(n):= : {}".format(mu_tf[0][0]))
        print("np_mu(n):= : {}".format(mu1))
        print("")
        
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu1))
        print("")
        
        print("scp_mu(n):= : {}".format(scp_sig[0]))
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig1))
        print("")
        
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig1))


if __name__ == '__main__':
    unittest.main()



