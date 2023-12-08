
import numpy as np
import tensorflow as tf
import copy

import tensorflow_probability as tfp
tfd = tfp.distributions

from bocpd.RRGPAR.RRGPARCP import RRSPARCP
from bocpd.test.RRGPAR_tf import RRSPARCP_TF, GaussianLaplaceReducedRank_TF
from bocpd.RRGPAR.gaussian_laplace_reduced_rank import  GaussianLaplaceReducedRank
import bocpd.generate_data as gd

from sklearn import preprocessing

import unittest


class Test_RRSPARCP(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-500 :,]
        
        variance = 0.2
        lengthscales = 0.5
        noise_parameter  = 0.5
        prior_parameter  = 0.1
        jitter = 0.0
        
        #MODEL1 = RRSPAR_TF
        lrgp_tf = GaussianLaplaceReducedRank_TF(variance, lengthscales , n_features = 15, L = 3)
        m_tf = RRSPARCP_TF( lrgp_tf, 1,  prior_parameter, noise_parameter)
        m_tf.setData(data)
        m_tf.initialize()
        m_tf._jitter = jitter

        #MODEL2 = fast RRSPAR
        k = GaussianLaplaceReducedRank(variance, lengthscales , n_features = 15, L = 3).fit()
        m = RRSPARCP(k, 1,  prior_parameter, noise_parameter )
        m._jitter = jitter
        m.setData(data)
        m.horizontal_update = True
        m.computeGradient()
        m.initialize()

        self.m_tf = m_tf
        self.m = m
        
    


    ###################################################
    # TEST n-logpdf UPDATE
    @unittest.SkipTest
    def test_logpdf_n(self):
        
        m_tf = self.m_tf
        m = self.m

        index = 15
        n = 20

        m.noise_trainable = True
        m_tf.noise_trainable = True
        
        m.unit_beta =  False
        m_tf.unit_beta =  False
        
        ###################### TF
        
        factor = m_tf.grad_adjustment_factor
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(n)   
            logpdf_tf_i = logpdf_tf[index,:]   
        dlogpdf_tf = tape.gradient(logpdf_tf_i, m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf) * factor
  
        ###################### NP2
        m.computeGradient(True)
        m.initialize()
        
        for i in range(n+1):
            logpdf, dlogpdf = m.logpdf(i)
            
        ###################### NP2
        m2 = copy.deepcopy(m)
        m2.computeGradient(False)
        m2.initialize()
        
        for i in range(n+1):
            logpdf2 = np.log( m2.pdf(i))

        print("")
        print("TEST_logpdf(n)")
        print("logpdf(n) - compare")
        for i in range(len(logpdf)):
             print("[tf, np1, np2]:= :[ {}".format(logpdf_tf[i][0]) + ",{}".format(logpdf[i]) + ",{}".format(logpdf2[i]) + "]")
             #self.assertAlmostEqual(logpdf[i], logpdf_tf[i][0].numpy(), places=7)

        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf[index,:]))
        
        

        
    ###################################################
    # TEST n-prediction UPDATE
    @unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m = self.m

        n = 300
        index = 1
        
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor

        ####################### TF calculation
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf, _ = m_tf.prediction(n)
            mu_tf_i = mu_tf[index]
        dmu_tf = tape.gradient(mu_tf_i, m_tf.trainable_parameters)
        dmu_tf = dmu_tf[:-1] * factor[:-1]

        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(n)
            sig_tf_i = sig_tf[index]
        dsig_tf = tape.gradient(sig_tf_i, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[:-1] * factor[:-1]

        #################### NP
        m.initialize()
        for i in range(n+1) :
            (mu, dmu), (sig, dsig) = m.prediction(i)

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
            sig_rec[ii]  = m.kernel(xt) - Ks2.T @ np.linalg.inv(Kss2) @  Ks2
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

        n = 300
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
        dmu_tf = dmu_tf[:-1] * factor[:-1]
 
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _, sig_tf = m_tf.prediction(n)
            sig_tf_i = sig_tf[index]
        dsig_tf = tape.gradient(sig_tf_i, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf[:-1] * factor[:-1]
      
        #################### NP
        m.initialize()
        for i in range(n+1):
            (mu, dmu), (sig, dsig) = m.prediction(i)
            
               
        ###################### NP2
        m2 = copy.deepcopy(m)
        m2.computeGradient(False)
        m2.initialize()
        
        for i in range(n+1):
            (mu2, sig2) = m2.prediction(i)

        print("")
        print("TEST_prediction_with_MRC("+ str(n) + ")")
    
        print("COMPARE_mu(n)")
        for i in range(n+1):
            print("[tf, np1, np2]:= :[ {}".format(mu_tf[i][0]) + ",{}".format(mu[i]) + ",{}".format(mu2[i]) + "]")
            self.assertAlmostEqual(mu[i], mu_tf[i][0].numpy(), places=7)
        
        print("")
        print("COMPARE_sig(n)")
        for i in range(n+1):
              print("[tf, np1, np2]:= :[ {}".format(sig_tf[i][0]) + ",{}".format(sig[i]) + ",{}".format(sig2[i]) + "]")
              self.assertAlmostEqual(sig[i], sig_tf[i][0].numpy(), places=7)


        print("")
        print("tf_mu_grad:= : {}".format(np.array(dmu_tf)))
        print("np_mu_grad:= : {}".format(dmu[index]))

        print("")
        print("tf_sig_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_grad:= : {}".format(dsig[index]))
        
    
    ###################################################
    # TEST 0-prediction 
    @unittest.SkipTest
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
        
        m.initialize()
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
    @unittest.SkipTest
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
        dlogpdf_tf = (dlogpdf_tf[0], 0, 0, dlogpdf_tf[3])
        dlogpdf_tf *= m_tf.grad_adjustment_factor
        
        m.initialize()
        logpdf, dlogpdf = m.logpdf(0)
        
        print("")
        print("TEST_logpdf("+ str(0) +")")
        print("tf_pdf(0):= : {}".format(logpdf_tf[0]))
        print("np_pdf(0):= : {}".format(logpdf))
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
       
        
    #########################################################
    #@unittest.SkipTest
    # TEST SSE UPDATE
    def test_SSE_update(self):

        m_tf = self.m_tf
        m = self.m

        t = n = 30
        index = 10
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        ############ TF
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _ = m_tf.logpdf(n)
            SSE_tf = m_tf.SSE_t
            out = SSE_tf[index][0]
        dSSE_tf = tape.gradient(out, m_tf.trainable_parameters)
        dSSE_tf= dSSE_tf[:-1] * factor[:-1]

        ############## NP
        m.fast_computation = True
        m.initialize()
        
        #Fast calculation
        m.initialize()
        for i in range(n + 1):
            _ = m.prediction(i)
        
        SSE_t = m.SSE_t
        dSSE_t = m.dSSE_t
        
        #RECALC1
        SSE_rec = np.zeros((n + 1,1))
        SSE_rec_2 = np.zeros((n + 1,1))
        yp = m.X[ : n ]
        xp = m.lagMatrix[:n,:,0]

        for ii in range(1, n + 1) : 
            Kss2 = m.kernel(xp[-ii:])
            Kss2[np.diag_indices_from(Kss2)] += m.total_noise_variance
            Kinv = np.linalg.inv(Kss2)
            SSE_rec[ii]  =  yp[-ii:].T @ Kinv @ yp[-ii:]

            
        #RECLAC2
        for ii in range(1, n + 1) : 
            u = m.Phi[n-ii:n,:].T @ yp[-ii:]
            Q = np.linalg.inv(m.Phi[n-ii:n,:].T @ m.Phi[n- ii : n,:] + m.v_inv_lamb) 
            SSE_rec_2[ii] = yp[-ii:].T @ yp[-ii:] - u.T @ Q @ u

        SSE_rec_2= SSE_rec_2 / m.total_noise_variance
        

        print("TEST_SSE("+ str(n) +")")
        for i in range(n+1):
            print("[rec, rec2, tf, np]:= :[{}".format(SSE_rec[i][0]) + ",{}".format(SSE_rec_2[i][0]) + ",{}".format(SSE_tf[i][0]) + ",{}".format(SSE_t[i]) + "]")
            self.assertAlmostEqual(SSE_rec[i][0], SSE_t[i], places=7)
            
        print("")
        print("tf_SSE_grad:= : {}".format(np.array(dSSE_tf)))
        print("np_SSE_grad:= : {}".format(dSSE_t[index]))
        





if __name__ == '__main__':
    unittest.main()



