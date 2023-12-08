
import numpy as np

from bocpd.GPTS.GPTSCP import SPTSCP
from bocpd.test.GPTS_tf import SPTSCP_TF
import bocpd.generate_data as gd

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing

import tensorflow as tf
import gpflow.kernels as gfk

import unittest



def switch(a):
    if len(a) == 2 :
        return a[::-1]
    else :
        return np.append(a[:2][::-1], a[2:])



class Test_SPTSCP(unittest.TestCase):

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

        alpha= 1e-01
        scalePrior = alpha
        
        #MODEL1 = fast GPTS_TF
        m_tf = SPTSCP_TF(k1, scalePrior)
        m_tf.setData(data, grid)
        
        #MODEL2 = fast GPTS
        k2 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m2 = SPTSCP(k2, scalePrior)
        m2.setData(data, grid)
        
        # #MODEL3 = GPTS
        k3 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m3 = SPTSCP(k3, scalePrior)
        m3.setData(data, grid)
        m3.fast_computation = False

        self.m_tf = m_tf
        self.m2 = m2



    ###################################################
    #@unittest.SkipTest
    def test_SSE(self):
    ## TEST SSE UPDATE (dSSE[last])

        m_tf = self.m_tf
        m2 = self.m2
        
        factor = m_tf.grad_adjustment_factor
    
        with tf.GradientTape() as tape:
            m_tf.initialize()
            m_tf.logpdf(0)
            m_tf.update(0)
            sse_tf = m_tf.SSE[-1:,0]
        dsse_tf = tape.gradient(sse_tf, m_tf.trainable_parameters[:-1])
        dsse_tf= np.array(dsse_tf) * factor[:-1]
        dsse_tf = switch(dsse_tf)
        
        m2.computeGradient(True)
        m2.initialize()
        m2.logpdf(0)
        m2.update(0)
        dsse = m2.dSSE[1,:]
        
        print("")
        print("TEST_SSE")
        print("tf_grad:= : {}".format(np.array(dsse_tf)))
        print("np_grad:= : {}".format(dsse))
    
    ###################################################
    # TEST n-logpdf UPDATE
    #@unittest.SkipTest
    def test_logpdf_n(self):
        
        m_tf = self.m_tf
        m2 = self.m2
        
        index = 18
        n = 20
        factor = m_tf.grad_adjustment_factor
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            for i in range(n):
                m_tf.logpdf(i)
                m_tf.update(i)
            logpdf_tf = m_tf.logpdf(n)[index,:]   
        dlogpdf_tf = tape.gradient(logpdf_tf , m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf ) * factor
        dlogpdf_tf  = switch(dlogpdf_tf )
        
        m2.computeGradient(True)
        m2.initialize()
        for i in range(n):
            m2.logpdf(i)
            m2.update(i)
        logpdf, dlogpdf = m2.logpdf(n)
        logpdf = logpdf[index]  
        dlogpdf = dlogpdf[index]

            
        print("")
        print("TEST_logpdf(n)")
        print("tf_pdf:= : {}".format(np.array(logpdf_tf[0])))
        print("np_pdf:= : {}".format(logpdf))
        
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf )))
        print("np_grad:= : {}".format(dlogpdf))
        
    ###################################################
    # TEST n-logpdf UPDATE
    #@unittest.SkipTest
    def test_logpdf_0(self):
        
        m_tf = self.m_tf
        m2 = self.m2
        
        factor = m_tf.grad_adjustment_factor
        
        with tf.GradientTape() as tape:
            m_tf.initialize()
            logpdf_tf = m_tf.logpdf(0) 
        dlogpdf_tf = tape.gradient(logpdf_tf , m_tf.trainable_parameters)
        dlogpdf_tf = np.array(dlogpdf_tf ) * factor
        dlogpdf_tf  = switch(dlogpdf_tf )
        
        m2.computeGradient(True)
        m2.initialize()
        logpdf, dlogpdf = m2.logpdf(0)

            
        print("")
        print("TEST_logpdf(0)")
        print("tf_pdf:= : {}".format(np.array(logpdf_tf [0])))
        print("np_pdf:= : {}".format(logpdf))
        
        print("")
        print("tf_grad:= : {}".format(np.array(dlogpdf_tf )))
        print("np_grad:= : {}".format(dlogpdf))








if __name__ == '__main__':
    unittest.main()



