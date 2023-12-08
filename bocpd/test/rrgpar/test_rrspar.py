
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions

from bocpd.RRGPAR.RRGPAR import RRSPAR
from bocpd.test.RRGPAR_tf import RRSPAR_TF, GaussianLaplaceReducedRank_TF
from bocpd.RRGPAR.gaussian_laplace_reduced_rank import  GaussianLaplaceReducedRank
import bocpd.generate_data as gd

from sklearn import preprocessing

import unittest
import copy



class Test_RRSPAR(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-200 :,]
        
        unit_beta = False
        variance = 0.2
        lengthscales = 0.5
        noise_parameter  = 0.5
        prior_parameter  = 0.2
        jitter = 0.0
        
        #MODEL1 = GPAR_TF
        lrgp_tf = GaussianLaplaceReducedRank_TF(variance, lengthscales, n_features = 15, L = 3)
        m_tf = RRSPAR_TF( lrgp_tf, 1, prior_parameter, noise_parameter)
        m_tf.setData(data)
        m_tf.unit_beta = unit_beta
        m_tf._jitter = jitter
        m_tf.initialize()

        #MODEL2 = fast GPAR
        k = GaussianLaplaceReducedRank(variance, lengthscales, n_features = 15, L = 3)
        m = RRSPAR(k, 1, prior_parameter, noise_parameter )
        m._jitter = jitter
        m.setData(data)
        m.unit_beta = unit_beta
        m.computeGradient(True)
        m.initialize()

        self.m_tf = m_tf
        self.m = m


    ###################################################
    @unittest.SkipTest
    def test_global_grad(self):
        
        m_tf = self.m_tf
        m = self.m
        
        m_tf.noise_trainable = True
        m.noise_trainable = True

        with tf.GradientTape() as tape:
            nlml_tf, Z_tf, (mu_tf, sigma_tf) = m_tf.run()  
        dnlml_tf = tape.gradient(nlml_tf, m_tf.trainable_parameters)
        dnlml_tf *= m_tf.grad_adjustment_factor

        m.fast_computation = True
        nlml, dnlml, Z, dZ = m.run_with_gradients()

        (T,D) = m.X.shape
        sigma = mu = np.zeros((1, 1))
        for t in range(T) :
            out = m.prediction(t)
            mu = np.append(mu, out[0][0])
            sigma = np.append(sigma, out[1][0])
            m.update(t)

        print("")
        print("GLOBAL_TF_VS_NP_logExpGrad")
        print("tf_nlml:= : {}".format(nlml_tf))
        print("np_nlml:= : {}".format(nlml))
        print("")
        print("tf_grad:= : {}".format(np.array(dnlml_tf)))
        print("np_grad:= : {}".format(dnlml))

    ###################################################
    #@unittest.SkipTest
    # TEST n-logpdf UPDATE
    def test_logpdf_n(self):

        m_tf = self.m_tf
        m = self.m

        n = 1
        m_tf.noise_trainable = True
        m.noise_trainable = True
        
        tmp = copy.deepcopy(m)
        tmp.fast_computation = False

        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(n)
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf = dlogpdf_tf * m_tf.grad_adjustment_factor

        for i in range(n + 1):
            logpdf, dlogpdf = m.logpdf(i)
            m.update(i)
            
       
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
    @unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m = self.m

        m.noise_trainable = False
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        tmp = copy.deepcopy(m)
        tmp.fast_computation = False

        n = 1
        
        #TF calculation
        with tf.GradientTape() as tape:
            mu_tf, sig_tf = m_tf.prediction(n)
        dmu_tf = tape.gradient(mu_tf, m_tf.trainable_parameters)
        dmu_tf = dmu_tf[:-1] * factor[:-1]
        
        with tf.GradientTape() as tape:
            mu_tf, sig_tf = m_tf.prediction(n)
        dsig_tf = tape.gradient(sig_tf, m_tf.trainable_parameters)
        dsig_tf = dsig_tf[:-1]  * factor[:-1]

        #Fast calculation
        for i in range(n + 1):
            (mu1, dmu1), (sig1, dsig1) = m.prediction(i)
            
        #Direct calculation
        (mu2, dmu2), (sig2, dsig2) = tmp.prediction(n)

        print("")
        print("TEST_prediction("+ str(n) + ")")
        
        print("tf_mu(n):= : {}".format(mu_tf[0]))
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

        (mu, dmu), (sig, dsig) = m.prediction(0)

        print("")
        print("TEST_prediction("+ str(0) + ")")
        
        print("tf_mu(0):= : {}".format(mu_tf))
        print("np_direct_mu(0):= : {}".format(mu))
        print("")
        
        print("tf_sig(n):= : {}".format(sig_tf))
        print("np_direct_sig(0):= : {}".format(sig))
        print("")
        
        print("tf_sig_fast_grad:= : {}".format(np.array(dsig_tf)))
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

        logpdf, dlogpdf = m.logpdf(0)
   

        print("")
        print("TEST_logpdf("+ str(0) +")")
        print("tf_pdf(0):= : {}".format(logpdf_tf[0]))
        print("np_pdf(0):= : {}".format(logpdf))
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        

    ###################################################
    #@unittest.SkipTest
    # TEST SSE UPDATE
    def test_SSE_update(self):

        m_tf = self.m_tf
        m = self.m

        n = 25
        m.noise_trainable = True
        m_tf.noise_trainable = True
        factor = m_tf.grad_adjustment_factor
        
        ############ TF
        with tf.GradientTape() as tape:
            m_tf.initialize()
            _ = m_tf.logpdf(n)
            SSE_tf = m_tf.SSE_t[0][0]
        dSSE_tf = tape.gradient(SSE_tf, m_tf.trainable_parameters)
        dSSE_tf= dSSE_tf[:-1] * factor[:-1]

        ############## NP
        m.fast_computation = True
        m.initialize()
        
        #Fast calculation
        for i in range(n + 1):
            _ = m.prediction(i)
        SSE = m.SSE_t[0]
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



