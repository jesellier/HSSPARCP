
import numpy as np

from bocpd.GPAR.GPARCP import GPARCP
from bocpd.test.GPAR_tf import GPARCP_TF
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
    if len(a) == 2 :
        return a[::-1]
    else :
        return np.append(a[:-1][::-1], a[-1])


class Test_GPARCP(unittest.TestCase):

    def setUp(self) :
        
        partition, data = gd.generate_normal_time_series(7, 50, 200)
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        #data = data[-200 :,]
   
        variance = 2
        lengthscales = 0.5
        k_tf = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
        jitter = 0.0
        scale_noise = 0.1
  
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

        self.m_tf = m_tf
        self.m = m
        
    


    ###################################################
    # TEST n-logpdf UPDATE
    @unittest.SkipTest
    def test_logpdf_n(self):
        
        m_tf = self.m_tf
        m = self.m

        index = 5
        n = 10
        m.noise_trainable = True
        m_tf.noise_trainable = True
        
        ###################### TF
        
        factor = m_tf.grad_adjustment_factor
        
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(n)   
            logpdf_tf_i = logpdf_tf[index,:]   
        dlogpdf_tf = tape.gradient(logpdf_tf_i, m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf) * factor
        dlogpdf_tf = switch(dlogpdf_tf)
        
        ###################### NP
        m.computeGradient(True)
        m.initialize()
        logpdf, dlogpdf = m.logpdf(n)

        print("")
        print("TEST_logpdf(n)")
        print("logpdf(n) - compare")
        for i in range(len(logpdf)):
             print("[tf, np]:= :[ {}".format(logpdf_tf[i][0]) + ",{}".format(logpdf[i]) + "]")
             self.assertAlmostEqual(logpdf[i], logpdf_tf[i][0].numpy(), places=7)

        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf[index,:]))
        
        
    ###################################################
    # TEST n-pdf UPDATE
    @unittest.SkipTest
    def test_pdf_n(self):
        
        m_tf = self.m_tf
        m = self.m

        n = 10
        ###################### TF
        m_tf.initialize()
        pdf_tf = m_tf.pdf(n)   

        ###################### NP
        m.computeGradient(False)
        m.initialize()
        pdf = m.pdf(n)

        print("")
        print("TEST_pdf(n)")
        print("pdf(n) - compare")
        for i in range(len(pdf)):
             print("[tf, np]:= :[ {}".format(pdf_tf[i][0]) + ",{}".format(pdf[i]) + "]")
             self.assertAlmostEqual(pdf[i], pdf_tf[i][0].numpy(), places=7)

        
        
    ###################################################
    # TEST n-logpdf UPDATE
    @unittest.SkipTest
    def test_logpdf_0(self):
        
        m_tf = self.m_tf
        m = self.m
        
        m.noise_trainable = True
        m_tf.noise_trainable = True

        ###################### TF
        factor = m_tf.grad_adjustment_factor
        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(0)
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf) * factor
        dlogpdf_tf = switch(dlogpdf_tf)
        
        #################### NP
        m.computeGradient(True)
        m.initialize()
        logpdf, dlogpdf = m.logpdf(0)
        dlogpdf =  dlogpdf[0]

        print("")
        print("TEST_logpdf(0)")
        print("tf_pdf:= : {}".format(np.array(logpdf_tf[0][0])))
        print("np_pdf:= : {}".format(logpdf[0]))
        self.assertAlmostEqual(logpdf[0], logpdf_tf[0][0].numpy(), places=7)
        
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        
        
    ###################################################
    # TEST n-prediction UPDATE
    #@unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m = self.m

        n = 800
        index = 5
        
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor

        ####################### TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, _ = m_tf.prediction(n)
            mu_tf_i = mu_tf[index]
        dmu_tf = tape.gradient(mu_tf_i, m_tf.trainable_parameters)
        dmu_tf = dmu_tf * factor
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(n)
            sig_tf_i = sig_tf[index]
        dsig_tf = tape.gradient(sig_tf_i, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf * factor
        dsig_tf = switch(dsig_tf)
        
        #################### NP
        m.initialize()
        (mu, dmu), (sig, dsig) = m.prediction(n)

        #Scipy recompute
        mu_scp = np.zeros((n + 1,1))
        sig_scp = np.zeros((n + 1,1))
        
        xt = m.lagMatrix[n]
        for i in range(n) :
            yp = m.X[i : n ]
            xp = m.lagMatrix[i:n,:,0]
            gpr = GaussianProcessRegressor(kernel= m.kernel, optimizer = None, alpha=  m.total_noise_variance, copy_X_train = False).fit(xp, yp)
            mu_scp[i], sig_scp[i] = gpr.predict(xt.reshape(1, -1), return_std=True)
            sig_scp[i] = sig_scp[i]**2
        
        mu_scp[-1] =  0
        sig_scp[-1] =  m.kernel(xt)
        mu_scp  = mu_scp [::-1]
        sig_scp = sig_scp [::-1]

        print("")
        print("TEST_prediction("+ str(n) + ")")
    
        print("COMPARE_mu(n)")
        for i in range(n+1):
              print("[scp, tf, np]:= :[ {}".format(mu_scp[i][0]) + ",{}".format(mu_tf[i][0]) + ",{}".format(mu[i]) + "]")
              self.assertAlmostEqual(mu_scp[i][0], mu_tf[i][0].numpy(), places=7)
              self.assertAlmostEqual(mu_scp[i][0], mu[i], places=7)
        
        print("")
        print("COMPARE_sig(n)")
        for i in range(n+1):
              print("[scp, tf, np]:= :[ {}".format(sig_scp[i][0]) + ",{}".format(sig_tf[i][0]) + ",{}".format(sig[i]) + "]")
              self.assertAlmostEqual(sig_scp[i][0], sig_tf[i][0].numpy(), places=7)
              self.assertAlmostEqual(sig_scp[i][0], sig[i], places=7)

        print("")
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu[index]))

        print("")
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig[index]))
        

    ###################################################
    # TEST 0-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_0(self):

        m_tf = self.m_tf
        m = self.m
        
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor

        ####################### TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, _ = m_tf.prediction(0)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(0)
           
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[:-1] * factor[:-1]
        dsig_tf = switch(dsig_tf)
        
        #################### NP
        m.initialize()
        (mu, dmu), (sig, dsig) = m.prediction(0)
  
        print("")
        print("TEST_prediction(0)")

        print("mu[tf, np]:= :[ {}".format(mu_tf[0]) + ",{}".format(mu) + "]")
        print("sig[tf, np]:= :[ {}".format(sig_tf[0]) + ",{}".format(sig) + "]")
      
        print("")
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu))

        print("")
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig))
        
        
    ###################################################
    # TEST n-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_n_with_MRC(self):
        
        m_tf = self.m_tf
        m = self.m

        n = 20
        index = 5
        
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        m.maxLen = 10
        m_tf.maxLen = 10

        ####################### TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, _ = m_tf.prediction(n)
            mu_tf_i = mu_tf[index]
        dmu_tf = tape.gradient(mu_tf_i, m_tf.trainable_parameters)
        dmu_tf = dmu_tf * factor
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(n)
            sig_tf_i = sig_tf[index]
        dsig_tf = tape.gradient(sig_tf_i, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf * factor
        dsig_tf = switch(dsig_tf)
        
        #################### NP
        m.initialize()
        (mu, dmu), (sig, dsig) = m.prediction(n)

     
        print("")
        print("TEST_prediction_with_MRC("+ str(n) + ")")
    
        print("COMPARE_mu(n)")
        for i in range(n+1):
              print("[tf, np]:= :[{}".format(mu_tf[i][0]) + ",{}".format(mu[i]) + "]")
              self.assertAlmostEqual(mu[i], mu_tf[i][0].numpy(), places=7)
        
        print("")
        print("COMPARE_sig(n)")
        for i in range(n+1):
              print("[tf, np]:= :[ {}".format(sig_tf[i][0]) + ",{}".format(sig[i]) + "]")
              self.assertAlmostEqual(sig[i], sig_tf[i][0].numpy(), places=7)


        print("")
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu[index]))

        print("")
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig[index]))






if __name__ == '__main__':
    unittest.main()



