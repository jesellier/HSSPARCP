
import numpy as np

from bocpd.GPTS.GPTSCP import GPTSCP
from bocpd.test.GPTS_tf import GPTSCP_TF
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
        return np.append(a[:2][::-1], a[2:])




class Test_GPTSCP(unittest.TestCase):

    def setUp(self) :
        
        partition, data = gd.generate_normal_time_series(7, 50, 200)
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-200 :,]
        
        grid = (np.atleast_2d(range(len(data))) * 0.5).T

        variance = 2
        lengthscales = 0.5
        k1 = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
        
        #MODEL1 = fast GPTS_TF
        m_tf = GPTSCP_TF(k1)
        m_tf.setData(data, grid)

        #MODEL2 = GPTS
        k2 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m2 = GPTSCP(k2)
        m2.setData(data, grid)
        m2.computeGradient()

        self.m_tf = m_tf
        self.m2 = m2
        

    
    ###################################################
    # TEST n-logpdf UPDATE
    #@unittest.SkipTest
    def test_logpdf_n(self):
        
        m_tf = self.m_tf
        m2 = self.m2

        index = 18
        n = 20
        
        factor = m_tf.grad_adjustment_factor
        
        ##### TF
        with tf.GradientTape() as tape:
            m_tf.initialize()
            for i in range(n):
                m_tf.logpdf(i)
                m_tf.update(i)
            logpdf_tf = m_tf.logpdf(n) 
            logpdf_tf_i = logpdf_tf[index,:]   
        dlogpdf_tf = tape.gradient(logpdf_tf_i, m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf) * factor
        dlogpdf_tf = switch(dlogpdf_tf)
        
        ##### NP
        m2.computeGradient(True)
        m2.initialize()
        for i in range(n):
            m2.logpdf(i)
            m2.update(i)
        logpdf, dlogpdf = m2.logpdf(n) 
        dlogpdf = dlogpdf[index]

        print("TEST_logpdf(n)")
        print("logpdf(n) - compare")
        for i in range(len(logpdf)):
             print("[tf, np]:= :[ {}".format(logpdf_tf[i][0]) + ",{}".format(logpdf[i]) + "]")
             self.assertAlmostEqual(logpdf_tf[i], logpdf_tf[i][0].numpy(), places=7)

        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        
        
    ###################################################
    # TEST n-logpdf UPDATE
    #@unittest.SkipTest
    def test_logpdf_0(self):
        
        m_tf = self.m_tf
        m2 = self.m2

        ####### TF
        factor = m_tf.grad_adjustment_factor
        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(0)
        dlogpdf_tf = tape.gradient(logpdf_tf, m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf) * factor
        dlogpdf_tf = switch(dlogpdf_tf)
        
        ##### NP
        m2.computeGradient(True)
        m2.initialize()
        logpdf, dlogpdf = m2.logpdf(0)
 
        print("")
        print("TEST_logpdf(0)")
        print("tf_pdf:= : {}".format(np.array(logpdf_tf[0])))
        print("np_pdf:= : {}".format(logpdf))
        
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))
        
        
    ###################################################
    # TEST n-prediction UPDATE
    #@unittest.SkipTest
    def test_prediction_n(self):
        
        m_tf = self.m_tf
        m2 = self.m2

        factor = m_tf.grad_adjustment_factor
        m_tf.fast_computation = True
        
        n = 20
        index = 5

        ########TF
        with tf.GradientTape() as tape:
            m_tf.initialize()
            mu_tf = m_tf.mu_t(n)
            mu_tf_i = mu_tf[index]
        dmu_tf = tape.gradient(mu_tf_i, m_tf.trainable_parameters)
        dmu_tf = dmu_tf * factor
        dmu_tf = switch(dmu_tf)
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            sig_tf = m_tf.sigma2_t(n)
            sig_tf_i = sig_tf[index]
        dsig_tf = tape.gradient(sig_tf_i, m_tf.trainable_parameters)
        dsig_tf =  dsig_tf * factor
        dsig_tf = switch(dsig_tf)
        
        ##### NP
        m2.initialize()
        for i in range(n):
            _ = m2.prediction(i)
            m2.update(i)
        (mu, dmu), (sig, dsig) = m2.prediction(n)

        ####### Scipy recompute
        mu_scp = np.zeros((n + 1,1))
        sig_scp = np.zeros((n + 1,1))
        
        xt = m2.grid[n]
        for i in range(n) :
            yp = m2.X[i : n ]
            xp = m2.grid[i: n]
            gpr = GaussianProcessRegressor(kernel= m2.kernel, optimizer = None, alpha= 0.0, copy_X_train = False).fit(xp, yp)
            mu_scp[i], sig_scp[i] = gpr.predict(xt.reshape(1, -1), return_std=True)
            sig_scp[i] = sig_scp[i]**2
        
        mu_scp[-1] =  0
        sig_scp[-1] =  m2.kernel(xt)
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
        print("np_mu_fast_grad:= : {}".format(dmu[index]))

        print("")
        print("tf_sig_fast_grad:= : {}".format(np.array(dsig_tf)))
        print("np_sig_fast_grad:= : {}".format(dsig[index]))







if __name__ == '__main__':
    unittest.main()



