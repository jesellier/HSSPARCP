
import numpy as np

from bocpd.model_base import LogExpSpaceTransformer
from bocpd.GPAR.GPAR import GPAR

from bocpd.test.GP_base_tf import TensorMisc
from bocpd.test.GPAR_tf import GPAR_TF
import bocpd.generate_data as gd

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing

import tensorflow as tf
import gpflow.kernels as gfk

import tensorflow_probability as tfp
tfd = tfp.distributions

import unittest
import copy


def switch(a, p):
    out = a[p]
    out = np.append(out, a[:p])
    
    if len(a) > p+1 :
        out = np.append(out, a[-1])

    return out





class Test_GPAR2(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-200 :,]
        
        p = 2
        variance = 0.2
        lengthscales = [0.5,0.5] #,0.1]
        k1 = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
        noiseScale = 0.1
        jitter = 0.0
        
        #MODEL1 = GPAR_TF
        m_tf = GPAR_TF(k1, p, noiseScale)
        m_tf.setData(data)
        m_tf.initialize()
        m_tf._jitter = jitter
        

        #MODEL2 = GPAR
        k2 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m2 = GPAR(k2, p, noiseScale)
        m2._jitter = jitter
        m2.setData(data)
        m2.initialize()
        m2.computeGradient()
        
        #MODEL3 = with LogExp transform gradient
        k3 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m3 = GPAR(k3, p, noiseScale)
        m3._jitter = jitter
        m3.setData(data)
        m3.initialize()
        m3.computeGradient()
        m3.transformer = LogExpSpaceTransformer()

        self.m_tf = m_tf
        self.m2 = m2
        self.m3 = m3



    ###################################################
    #@unittest.SkipTest
    def test_global_grad(self):
        
        m_tf = self.m_tf
        m3 = self.m3
        
        m_tf.noise_trainable = True
        m3.noise_trainable = True
        
        with tf.GradientTape() as tape:
            nlml_tf, Z = m_tf.run()
        dnlml_tf = tape.gradient(nlml_tf, m_tf.trainable_parameters)
        dnlml_tf = TensorMisc().pack_tensors(dnlml_tf)
        dnlml_tf = switch(dnlml_tf, m_tf.p)
        
        m3.fast_computation = True
        nlml3, dnlml3, Z3, dZ3 = m3.run_with_gradients()
  
        print("")
        print("GLOBAL_TF_VS_NP_logExpGrad")
        print("tf_nlml:= : {}".format(nlml_tf))
        print("np_nlml:= : {}".format(nlml3))
        print("")
        print("tf_grad:= : {}".format(np.array(dnlml_tf)))
        print("np_grad:= : {}".format(dnlml3))
   

    
    ###################################################
    #@unittest.SkipTest
    # TEST n-logpdf UPDATE
    def test_logpdf_n(self):

        m_tf = self.m_tf
        m3 = self.m3
        
        m_tf.noise_trainable = True
        m3.noise_trainable = True

        tmp = copy.deepcopy(m3)
        tmp.fast_computation = False
 
        n = 25

        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(n)
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf = TensorMisc().pack_tensors(dlogpdf_tf)
        dlogpdf_tf = switch(dlogpdf_tf, m_tf.p)

        for i in range(n + 1):
            logpdf, dlogpdf = m3.logpdf(i)
            m3.update(i)
            
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
        m2 = self.m2
        
        n = 25
        m2.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        tmp = copy.deepcopy(m2)
        tmp.fast_computation = False

        
        with tf.GradientTape() as tape:
            mu_tf, sig_tf = m_tf.prediction(n)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        dmu_tf = TensorMisc().pack_tensors(dmu_tf) * factor
        dmu_tf = switch(dmu_tf, m_tf.p)
        
        with tf.GradientTape() as tape:
            mu_tf, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  TensorMisc().pack_tensors(dsig_tf) * factor
        dsig_tf = switch(dsig_tf,  m_tf.p)
 

        ###############################################################
        
        for i in range(n + 1):
            (mu1, dmu1), (sig1, dsig1) = m2.prediction(i)
        
        
        (mu2, dmu2), (sig2, dsig2) = tmp.prediction(n)

        print("")
        print("TEST_prediction("+ str(n) + ")")
        
        print("tf_mu(n):= : {}".format(mu_tf[0][0]))
        print("np_fast_mu(n):= : {}".format(mu1))
        print("np_direct_mu(n):= : {}".format(mu2))
        print("")
        
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_fast_grad:= : {}".format(dmu1))
        print("np_mu_direct_grad:= : {}".format(dmu2))
        print("")
        
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig1))
        print("np_direct_sig(n):= : {}".format(sig2))
        print("")
        
        print("tf_sig_fast_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_fast_grad:= : {}".format(dsig1))
        print("np_sig_direct_grad:= : {}".format(dsig2))

    
    ###################################################
    # TEST 0-prediction UPDATE
    #@unittest.SkipTest
    def test_prediction_0(self):
        
        m_tf = self.m_tf
        m2 = self.m2
        
        n = 0

        m2.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor

        with tf.GradientTape() as tape:
            mu_tf, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf  = TensorMisc().pack_tensors(dsig_tf[:-1]) * factor[:-1]
        dsig_tf = switch(dsig_tf, m_tf.p)
   
        (mu, dmu), (sig, dsig) = m2.prediction(n)

        print("")
        print("TEST_prediction("+ str(n) + ")")
        
        print("tf_mu(0):= : {}".format(mu_tf[0][0]))
        print("np_fast_mu(n):= : {}".format(mu))
        print("")
        
        print("tf_sig(n):= : {}".format(sig_tf[0][0]))
        print("np_sig(n):= : {}".format(sig))
        print("")
        
        print("tf_sig_grad:= : {}".format(dsig_tf))
        print("np_sig_grad:= : {}".format(dsig))
        

        
    ###################################################


if __name__ == '__main__':
    unittest.main()



