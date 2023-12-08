
import numpy as np
import tensorflow as tf
import gpflow.kernels as gfk

import tensorflow_probability as tfp
tfd = tfp.distributions

from bocpd.model_base import LogExpSpaceTransformer
from bocpd.GPAR.GPAR import GPAR
from bocpd.test.GPAR_tf import GPAR_TF
import bocpd.generate_data as gd

from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import unittest
import copy



def switch(a):
    if len(a) == 2 :
        return a[::-1]
    else :
        return np.append(a[:-1][::-1], a[-1])





class Test_GPAR1(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-1500 :,]
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-500 :,]
        
        data, _ = gd.import_snowfall_data()
        X = (data - np.mean(data)) / np.std(data)
        
        eval_cutoff = 1000
        data = X[eval_cutoff : eval_cutoff + 50]
        
        variance = 2
        lengthscales = 0.5
        k_tf = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
        noiseScale = 0.1
        jitter = 1e-10
        
        #MODEL1 = GPAR_TF
        m_tf = GPAR_TF(k_tf, 1, noiseScale)
        m_tf.setData(data)
        m_tf._jitter = jitter

        #MODEL2 = fast GPAR
        k = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m = GPAR(k, 1, noiseScale)
        m._jitter = jitter
        m.horizontal_update = True
        m.setData(data)
        m.computeGradient()

        self.m_tf = m_tf
        self.m = m




    ###################################################
    @unittest.SkipTest
    def test_global_grad(self):
        
        m_tf = self.m_tf
        m = self.m
        factor = m_tf.grad_adjustment_factor
        m_tf.noise_trainable = True
        m.noise_trainable = True
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            nlml_tf, Z = m_tf.run()
        g_tf = tape.gradient(nlml_tf, m_tf.trainable_parameters)
        g_tf  = g_tf * factor
        g_tf = switch(g_tf)
        
        m.fast_computation = True
        nlml, dnlml, Z, dZ = m.run_with_gradients()
        g = dnlml
  
        print("")
        print("GLOBAL_TF_VS_NP_logExpGrad")
        print("tf_nlml:= : {}".format(nlml_tf))
        print("np_nlml:= : {}".format(nlml))
        print("")
        print("tf_grad:= : {}".format(np.array(g_tf)))
        print("np_grad:= : {}".format(g))
   

    
    ###################################################
    @unittest.SkipTest
    # TEST n-logpdf UPDATE
    def test_logpdf_n(self):

        m_tf = self.m_tf
        m = self.m

        n = 1000
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        m.noise_trainable = True
        
        tmp = copy.deepcopy(m)
        tmp.fast_computation = False

        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(n)
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf  = dlogpdf_tf * factor
        dlogpdf_tf = switch(dlogpdf_tf)
        
        m.initialize()
        for i in range(n + 1):
            logpdf, dlogpdf = m.logpdf(i)
            m.update(i)
            
        tmp.initialize()
        logpdf_d, dlogpdf_d = tmp.logpdf(n)

        print("")
        print("TEST_logpdf("+ str(n) +")")
        print("tf_pdf(n):= : {}".format(logpdf_tf[0]))
        print("np_pdf(n):= : {}".format(logpdf))
        print("np_direct_pdf(n):= : {}".format(logpdf_d))
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        print("np_direct_grad:= : {}".format(dlogpdf_d))


    ###################################################
    # TEST n-prediction UPDATE
    #@unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m = self.m

        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        tmp = copy.deepcopy(m)
        tmp.fast_computation = False

        n = 46
        
        #TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, sig_tf = m_tf.prediction(n)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        dmu_tf = dmu_tf * factor
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf * factor
        dsig_tf = switch(dsig_tf)
        
        #Fast calculation
        m.initialize()
        for i in range(n + 1):
            (mu1, dmu1), (sig1, dsig1) = m.prediction(i)
            
        #Direct calculation
        tmp.initialize()
        (mu2, dmu2), (sig2, dsig2) = tmp.prediction(n)
        
        #Scipy recompute
        xt = m.lagMatrix[n]
        yp = m.X[ : n ]
        xp = m.lagMatrix[:n,:,0]
        gpr = GaussianProcessRegressor(kernel= m.kernel, optimizer = None, alpha= m.total_noise_variance, copy_X_train = False).fit(xp, yp)
        scp_mu, scp_sig = gpr.predict(xt, return_std=True)
        scp_sig = scp_sig**2

        print("")
        print("TEST_prediction("+ str(n) + ")")
        
        print("scp_mu(n):= : {}".format(scp_mu[0][0]))
        print("tf_mu(n):= : {}".format(mu_tf[0][0]))
        print("np_fast_mu(n):= : {}".format(mu1))
        print("np_direct_mu(n):= : {}".format(mu2))
        print("")

        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_fast_grad:= : {}".format(dmu1))
        print("np_mu_direct_grad:= : {}".format(dmu2))
        print("")
        
        print("scp_sig(n):= : {}".format(scp_sig[0]))
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig1))
        print("np_direct_sig(n):= : {}".format(sig2))
        print("")
        
        print("tf_sig_fast_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_fast_grad:= : {}".format(dsig1))
        print("np_sig_direct_grad:= : {}".format(dsig2))
        
    
     ###################################################
    # TEST n-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_n_with_MRC(self):
 
        m_tf = self.m_tf
        m = self.m
        
        n = 25

        m.maxLen = 10
        m.horizontal_update = False
        m_tf.maxLen = 10

        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor

        #TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, sig_tf = m_tf.prediction(n)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        dmu_tf = dmu_tf * factor
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf * factor
        dsig_tf = switch(dsig_tf)
        
        m.initialize()
        (mu1, dmu1), (sig1, dsig1) = m.prediction(n)

        print("")
        print("TEST_GPAR_prediction("+ str(n) + "_with_MRC)")

        print("tf_mu(n):= : {}".format(mu_tf[0][0]))
        print("np_mu(n):= : {}".format(mu1))
        print("")
        
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu1))
        print("")
        
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig1))
        print("")
        
        print("tf_sig_fast_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_fast_grad:= : {}".format(dsig1))

    
    ###################################################
    # TEST n-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_0(self):
        
        m_tf = self.m_tf
        m = self.m

        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        tmp = copy.deepcopy(m)
        tmp.fast_computation = False

        n = 0

        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[:-1] * factor[:-1]
        dsig_tf = switch(dsig_tf)
        
        
        m.initialize()
        for i in range(n + 1):
            (mu1, dmu1), (sig1, dsig1) = m.prediction(i)
            
        tmp.initialize()
        (mu2, dmu2), (sig2, dsig2) = tmp.prediction(n)

        print("")
        print("TEST_prediction("+ str(n) + ")")
        
        print("tf_mu(0):= : {}".format(mu_tf[0][0]))
        print("np_fast_mu(n):= : {}".format(mu1))
        print("np_direct_mu(0):= : {}".format(mu2))
        print("")
        
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig1))
        print("np_direct_sig(0):= : {}".format(sig2))
        print("")
        
        print("tf_sig_fast_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_fast_grad:= : {}".format(dsig1))
        print("np_sig_direct_grad:= : {}".format(dsig2))
        

        
    ###################################################


if __name__ == '__main__':
    unittest.main()



