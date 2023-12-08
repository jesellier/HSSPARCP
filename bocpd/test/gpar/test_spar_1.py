

import numpy as np

from bocpd.model_base import LogExpSpaceTransformer
from bocpd.GPAR.GPAR import SPAR
from bocpd.test.GPAR_tf import SPAR_TF
import bocpd.generate_data as gd

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor

import tensorflow as tf
import gpflow.kernels as gfk

import tensorflow_probability as tfp
tfd = tfp.distributions

import unittest
import copy


def switch(a):
    if len(a) == 2 :
        return a[::-1]
    else :
        return np.append(a[:-1][::-1], a[-1])


class Test_SPAR(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-200 :,]
        
        variance = 0.2
        lengthscales = 0.5
        k_tf = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
        priorScale = 0.1
        jitter = 0.1 #1e-05
        
        #MODEL1 = SPAR_TF
        m_tf = SPAR_TF(k_tf, 1, priorScale)
        m_tf.setData(data)
        m_tf._jitter = jitter
        

        #MODEL2 = fast SPAR
        k = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m = SPAR(k, 1, priorScale)
        m._jitter = jitter
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
        
        with tf.GradientTape() as tape:
            nlml_tf, Z = m_tf.run()
        g_tf = tape.gradient(nlml_tf, m_tf.trainable_parameters)
        g_tf= g_tf* factor
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
    #@unittest.SkipTest
    # TEST n-logpdf UPDATE
    def test_logpdf_n(self):

        m_tf = self.m_tf
        m = self.m

        n = 10
        factor = m_tf.grad_adjustment_factor
        m_fast = copy.deepcopy(m)

        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(n)
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf  = dlogpdf_tf  * factor
        dlogpdf_tf = switch(dlogpdf_tf)
        
        ############## NP
        m.fast_computation = False
        m.horizontal_update = True
        m.initialize()
        logpdf, dlogpdf = m.logpdf(n)

        m_fast.fast_computation = True
        m_fast.initialize()
        
        for i in range(n):
              _ = m_fast.logpdf(i)
              m_fast.update(i)
        logpdf_fast, dlogpdf_fast = m_fast.logpdf(n)

        print("")
        print("TEST_logpdf("+ str(n) +")")
        print("tf_pdf(n):= : {}".format(logpdf_tf[0]))
        print("np_pdf(n):= : {}".format(logpdf))
        print("np_fast_pdf(n):= : {}".format(logpdf_fast))
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        print("np_fast_grad:= : {}".format(dlogpdf_fast))

        
    ###################################################
    # TEST n-prediction UPDATE
    #@unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m = self.m

        factor = m_tf.grad_adjustment_factor
        n = 10
        
        ##################### TF
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, sig_tf = m_tf.prediction(n)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        dmu_tf= dmu_tf[:-1]  * factor[:-1] 
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[:-1]  * factor[:-1] 
        dsig_tf = switch(dsig_tf)
        

        ############## NP
        m.horizontal_update = True
        m.fast_computation = False
        m.initialize()
        (mu, dmu), (sig, dsig) = m.prediction(n)
        
        m_fast = copy.deepcopy(m)
        m_fast.fast_computation = True
        m_fast.initialize()
        for i in range(n):
              logpdf, dlogpdf = m_fast.logpdf(i)
              m_fast.update(i)
        (mu_fast, dmu_fast), (sig_fast, dsig_fast) = m_fast.prediction(n)

        
        #####################Sciy
        xt = m.lagMatrix[n]
        yp = m.X[ : n ]
        xp = m.lagMatrix[:n,:,0]
        gpr = GaussianProcessRegressor(kernel= m.kernel, optimizer = None, alpha=  m.total_noise_variance, copy_X_train = False).fit(xp, yp)
        mu_scp, sig_scp = gpr.predict(xt.reshape(1, -1), return_std=True)
        sig_scp = sig_scp**2

        print("")
        print("TEST_prediction("+ str(n) + ")")
        
        print("scp_mu(n):= : {}".format(mu_scp[0][0]))
        
        print("tf_mu(n):= : {}".format(mu_tf[0][0]))
        print("np_mu(n):= : {}".format(mu))
        print("np_mu_fast(n):= : {}".format(mu_fast))
        print("")
        
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu))
        print("np_mu_fast_grad:= : {}".format(dmu_fast))
        print("")
        
        print("scp_sig(n):= : {}".format(sig_scp[0]))
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig))
        print("np_sig_fast(n):= : {}".format(sig_fast))
        print("")
        
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig))
        print("np_sig_fast_grad:= : {}".format(dsig_fast))
        
        
        
    ###################################################
    # TEST n-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_n_with_MRC(self):
        
        m_tf = self.m_tf
        m = self.m
        
        m.maxLen = 10
        m.horizontal_update = False
        m_tf.maxLen = 10

        factor = m_tf.grad_adjustment_factor
        n = 20
        
        ##################### TF
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, _ = m_tf.prediction(n)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        dmu_tf= dmu_tf[:-1]  * factor[:-1] 
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[:-1]  * factor[:-1] 
        dsig_tf = switch(dsig_tf)
        

        ############## NP
        m.initialize()
        (mu, dmu), (sig, dsig) = m.prediction(n)


        print("")
        print("TEST_prediction("+ str(n) + "_with_MRC)")

        print("tf_mu(n):= : {}".format(mu_tf[0][0]))
        print("np_mu(n):= : {}".format(mu))
        print("")
        
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu))
        print("")
        
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig))
        print("")
        
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig))

    
    ###################################################
    # TEST n-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_0(self):
        
        m_tf = self.m_tf
        m = self.m
        
        factor = m_tf.grad_adjustment_factor

        with tf.GradientTape() as tape:
            mu_tf, sig_tf = m_tf.prediction(0)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[:-1] * factor[:-1]
        dsig_tf = switch(dsig_tf)
        
        m.initialize()
        (mu, dmu), (sig, dsig) = m.prediction(0)
        
        
        m_fast = copy.deepcopy(m)
        m_fast.fast_computation = True
        m_fast.initialize()
        (mu_fast, dmu_fast), (sig_fast, dsig_fast) = m_fast.prediction(0)

        print("")
        print("TEST_prediction(0)")

        print("np_mu(n):= : {}".format(mu))
        print("np_mu_fast(n):= : {}".format(mu_fast))
        print("")

        print("np_mu_grad:= : {}".format(dmu))
        print("np_mu_fast_grad:= : {}".format(dmu_fast))
        print("")
        
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig))
        print("np_sig_fast(n):= : {}".format(sig_fast))
        print("")
        
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig))
        print("np_sig_fast_grad:= : {}".format(dsig_fast))

  ###################################################
    #@unittest.SkipTest
    # TEST SSE UPDATE
    def test_SSE_update(self):

        m_tf = self.m_tf
        m = self.m

        n = 10
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        ############ TF
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _ = m_tf.logpdf(n)
            SSE_tf = m_tf.SSE_t[0][0]
        dSSE_tf = tape.gradient(SSE_tf, m_tf.trainable_parameters)
        dSSE_tf = switch(dSSE_tf)
        dSSE_tf= dSSE_tf[:-1] * factor[:-1]

        ############## NP
        m.fast_computation = True
        m.initialize()
        
        #Fast calculation
        for i in range(n + 1):
            _ = m.prediction(i)
        SSE = m.SSE_t
        dSSE = m.dSSE_t
        
        ############## NP2
        m.horizontal_update = False
        m.initialize()
 
        _ = m.prediction(n)
        SSE = m.SSE_t
        dSSE = m.dSSE_t
        
        #RECALC
        #xt = m.lagMatrix[n]
        yp = m.X[ : n ]
        xp = m.lagMatrix[:n,:,0]
        
       #kt = m.kernel(xt.T)
        Kss = m.kernel(xp)
        #Ks = m.kernel( xp, xt.T)
        Kss[np.diag_indices_from(Kss)] += m.total_noise_variance
        
        Kinv = np.linalg.inv(Kss)
        SSE_recalc = (yp.T @ Kinv @ yp)[0][0]

        print("")
        print("TEST_SSE("+ str(n) +")")
        print("tf_SSE(n):= : {}".format(SSE_tf))
        print("np_SSE(n):= : {}".format(SSE))
        print("recalc_SSE(n):= : {}".format(SSE_recalc))
        print("") 
        print("tf_SSE_grad(n):= : {}".format(dSSE_tf))
        print("np_SSE_grad(n):= : {}".format(dSSE))

if __name__ == '__main__':
    unittest.main()



