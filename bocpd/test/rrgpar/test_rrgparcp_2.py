
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions

from bocpd.test.GP_base_tf  import TensorMisc
from bocpd.RRGPAR.RRGPARCP import RRGPARCP
from bocpd.test.RRGPAR_tf import RRGPARCP_TF, GaussianLaplaceReducedRank_TF
from bocpd.RRGPAR.gaussian_laplace_reduced_rank import  GaussianLaplaceReducedRankND
import bocpd.generate_data as gd

from sklearn import preprocessing

import unittest



class Test_RRGPARSCP(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-100 :,]
        
        
        variance = 0.2
        lengthscales = np.array([0.5,0.5])
        noise_parameter  = 0.5
        jitter = 0.0
        n_dimension = 2
        
        #MODEL1 = GPAR_TF
        lrgp_tf = GaussianLaplaceReducedRank_TF(variance, lengthscales ,  n_dimension =  n_dimension, n_features = 15, L = 3)
        m_tf = RRGPARCP_TF( lrgp_tf, n_dimension, noise_parameter)
        m_tf.setData(data)
        m_tf.initialize()
        m_tf._jitter = jitter

        #MODEL2 = fast GPAR
        k = GaussianLaplaceReducedRankND(variance, lengthscales ,  n_dimension =  n_dimension, n_features = 15, L = 3)
        m = RRGPARCP(k, n_dimension, noise_parameter )
        m._jitter = jitter
        m.setData(data)
        m.computeGradient()
        m.initialize()

        self.m_tf = m_tf
        self.m = m
        
      
        
    


    ###################################################
    # TEST n-logpdf UPDATE
    #@unittest.SkipTest
    def test_logpdf_n(self):
        
        m_tf = self.m_tf
        m = self.m

        index = 15
        n = 20
        m.noise_trainable = True
        m_tf.noise_trainable = True
        
        ###################### TF
        
        factor = m_tf.grad_adjustment_factor
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(n)   
            logpdf_tf_i = logpdf_tf[index,:]   
        dlogpdf_tf = tape.gradient(logpdf_tf_i, m_tf.trainable_parameters)
        dlogpdf_tf = TensorMisc().pack_tensors( dlogpdf_tf ) * factor
  
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
    # TEST n-prediction UPDATE
    #@unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m = self.m

        n = 20
        index = 18
        
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor

        ####################### TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, _ = m_tf.prediction(n)
            mu_tf_i = mu_tf[index]
        dmu_tf = tape.gradient(mu_tf_i, m_tf.trainable_parameters)
        dmu_tf = factor * TensorMisc().pack_tensors( dmu_tf)

        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(n)
            sig_tf_i = sig_tf[index]
        dsig_tf = tape.gradient(sig_tf_i, m_tf.trainable_parameters)
        dsig_tf = factor * TensorMisc().pack_tensors(dsig_tf)

        #################### NP
        m.initialize()
        (mu, dmu), (sig, dsig) = m.prediction(n)

        #recalc
        mu_rec = np.zeros((n + 1,1))
        sig_rec = np.zeros((n + 1,1))
        SSE_rec = np.zeros((n + 1,1))
        
        xt = m.lagMatrix[n]
        yp = m.X[ : n ]
        xp = m.lagMatrix[:n,:,0]

        for ii in range(1, n + 1) : 
            Kss2 = m.kernel(xp[-ii:])
            Ks2 = m.kernel(  xp[-ii:], xt.T)
            Kss2[np.diag_indices_from(Kss2)] += m.total_noise_variance
            Kinv = np.linalg.inv(Kss2)
            mu_rec[ii]  = Ks2.T @ Kinv @ yp[-ii:]
            sig_rec[ii]  = m.kernel(xt.T) - Ks2.T @ np.linalg.inv(Kss2) @  Ks2
            SSE_rec[ii]  =  yp[-ii:].T @ Kinv @ yp[-ii:]
            
        mu_rec[0] = SSE_rec[0] = 0.0
        sig_rec[0] =  m.kernel.variance

        print("")
        print("TEST_prediction("+ str(n) + ")")
    
        print("COMPARE_mu(n)")
        for i in range(n+1):
              print("[scp, tf, np]:= :[ {}".format(mu_rec[i][0]) + ",{}".format(mu_tf[i][0]) + ",{}".format(mu[i]) + "]")
              self.assertAlmostEqual(mu_rec[i][0], mu_tf[i][0].numpy(), places=7)
              self.assertAlmostEqual(mu_rec[i][0], mu[i], places=7)
        
        print("")
        print("COMPARE_sig(n)")
        for i in range(n+1):
              print("[scp, tf, np]:= :[ {}".format(sig_rec[i][0]) + ",{}".format(sig_tf[i][0]) + ",{}".format(sig[i]) + "]")
              self.assertAlmostEqual(sig_rec[i][0], sig_tf[i][0].numpy(), places=7)
              self.assertAlmostEqual(sig_rec[i][0], sig[i], places=7)

        print("")
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu[index]))

        print("")
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig[index]))

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
 
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(n)
            sig_tf_i = sig_tf[index]
        dsig_tf = tape.gradient(sig_tf_i, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf * factor
        
      
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
        
    
    ###################################################
    # TEST 0-prediction 
    #@unittest.SkipTest
    def test_prediction_0(self):
        
        m_tf = self.m_tf
        m = self.m

        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor

        with tf.GradientTape() as tape:
            mu_tf, sig_tf = m_tf.prediction(0)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[0] * factor[0]

        (mu, dmu), (sig, dsig) = m.prediction(0)

        print("")
        print("TEST_prediction("+ str(0) + ")")
        
        print("tf_mu(0):= : {}".format(mu_tf))
        print("np_direct_mu(0):= : {}".format(mu))
        print("")
        
        print("tf_sig(0):= : {}".format(sig_tf))
        print("np_direct_sig(0):= : {}".format(sig))
        print("")
        
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig))
        
        
     ###################################################
    #@unittest.SkipTest
    # TEST 0-logpdf UPDATE
    def test_logpdf_0(self):

        m_tf = self.m_tf
        m = self.m

        m_tf.noise_trainable = True
        m.noise_trainable = True

        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(0)
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf = (dlogpdf_tf[0], 0, 0, dlogpdf_tf[2])
        dlogpdf_tf = m_tf.grad_adjustment_factor * dlogpdf_tf

        logpdf, dlogpdf = m.logpdf(0)
   

        print("")
        print("TEST_logpdf("+ str(0) +")")
        print("tf_pdf(0):= : {}".format(logpdf_tf[0]))
        print("np_pdf(0):= : {}".format(logpdf))
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))

        







if __name__ == '__main__':
    unittest.main()



