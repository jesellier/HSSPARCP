
import numpy as np

from bocpd.GPTS.GPTS import SPTS
from bocpd.test.GPTS_tf import SPTS_TF
import bocpd.generate_data as gd

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing

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



class Test_SPTS(unittest.TestCase):

    def setUp(self) :
     
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-500 :,]
        
        grid = (np.atleast_2d(range(len(data))) * 0.5).T

        variance = 2
        lengthscales = 0.5
        k1 = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)

        alpha= 1e-01
        scalePrior = alpha
        
        #MODEL1 = fast GPTS_TF
        m_tf = SPTS_TF(k1, scalePrior)
        m_tf.setData(data, grid)
        
        #MODEL2 = fast GPTS
        k2 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m2 = SPTS(k2, scalePrior)
        m2.setData(data, grid)
        m2.computeGradient()

        self.m_tf = m_tf
        self.m2 = m2


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
    @unittest.SkipTest
    def test_dA(self):
        
        m_tf = self.m_tf
        m2 = self.m2
        factor = m_tf.grad_adjustment_factor

        with tf.GradientTape() as tape:
            m_tf.initialize()
            out = tf.reduce_sum(tf.reduce_sum(m_tf.A,0),0)
        g1 = tape.gradient(out, m_tf.kernel.trainable_variables)
        g1 = np.array(g1) * factor[:-1]
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
        g1 = tape.gradient(out, m_tf.kernel.trainable_variables)
        g1 = np.array(g1) * factor[:-1]
        g1 = switch(g1)
        
        m2.initialize()
        g2 =  sum(m2.dSig2)
        
        print("")
        print("TEST_dSig2")
        print("tf_grad:= : {}".format(np.array(g1)))
        print("np_grad:= : {}".format(g2))

    
    ###################################################
    #@unittest.SkipTest
    def test_logpdf_0(self):
    # # TEST logpdf(0)
        
        m_tf = self.m_tf
        m2 = self.m2
        factor = m_tf.grad_adjustment_factor

        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf  = m_tf.logpdf(0)
        dlogpdf_tf = tape.gradient(logpdf_tf , m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf) * factor
        dlogpdf_tf = switch(dlogpdf_tf)

        m2.initialize()
        logpdf, dlogpdf = m2.logpdf(0)
        
        print("")
        print("TEST_logpdf(0)")
        print("tf_logpdf(0):= : {}".format(np.array(logpdf_tf)))
        print("np_logpdf(0):= : {}".format(logpdf))
        print("")

        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))

    
    ###################################################
    #@unittest.SkipTest
    def test_SSE_1(self):
    ## TEST SSE UPDATE
    
        m_tf = self.m_tf
        m2 = self.m2
        factor = m_tf.grad_adjustment_factor
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            m_tf.logpdf(0)
            m_tf.update(0)
            out = m_tf.SSE
        g1 = tape.gradient(out, m_tf.trainable_parameters[:-1])
        g1= np.array(g1) * factor[:-1]
        g1 = switch(g1)
    
        m2.initialize()
        m2.logpdf(0)
        m2.update(0)
        g2 = m2.dSSE
        
        print("")
        print("TEST_SSE_1")
        print("tf_grad:= : {}".format(np.array(g1)))
        print("np_grad:= : {}".format(g2))
        
    ###################################################
    #@unittest.SkipTest
    def test_SSE_n(self):
    ## TEST SSE UPDATE

        m2 = self.m2
        
        n = 20
     
        m2.initialize()
        for i in range(n):
            _ = m2.logpdf(i)
            m2.update(i)
        logpdf, dlogpdf = m2.logpdf(n)
    
        SSE = m2.SSE - 2 * m2._beta0
        
        y = m2.X[ : n]
        xp = m2.grid[ : n]
        xt = tf.expand_dims(m2.grid[n],1)
        
        K = m2.kernel(xp, xp)
        Kss = m2.kernel(xt, xt)
        Ks = m2.kernel(xp, xt)
        Kinv = tf.linalg.inv(K)
        
        SSE_recalc = y.T @ Kinv @ y
        
        print("")
        print("TEST_SSE_n")
        print("SSE:= : {}".format(SSE))
        print("SSE_recalc:= : {}".format(SSE_recalc[0][0]))

    
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
        print("tf_logpdf(n):= : {}".format(np.array(logpdf_tf[0])))
        print("np_logpdf(n):= : {}".format(logpdf))
        print("")

        print("tf_grad:= : {}".format(np.array(dlogpdf_tf)))
        print("np_grad:= : {}".format(dlogpdf))







if __name__ == '__main__':
    unittest.main()
    
    
    




