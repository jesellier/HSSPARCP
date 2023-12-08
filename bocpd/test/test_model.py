
import numpy as np
import tensorflow as tf
import gpflow.kernels as gfk

import tensorflow_probability as tfp
tfd = tfp.distributions

from bocpd.GPAR.GPAR import SPAR
from bocpd.test.GPAR_tf import SPAR_TF
import bocpd.generate_data as gd

from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import unittest



def switch(a):
    if len(a) == 2 :
        return a[::-1]
    else :
        return np.append(a[:-1][::-1], a[-1])


def log_space_transformation( a ) :
    return np.log( np.exp( a) - 1)

def log_space_inverse_transformation( a ) :
    return tf.math.log( tf.math.exp( a) + 1)

def log_space_gradient_factor( a):
    #transform gradient from  np.log( np.exp( a) - 1) space to log( a ) space
    return  a * np.exp(a) / ( np.exp(a) - 1)

def switch2(a):
     return np.append(a[:-1][::-1], a[-1])



class Test_Model(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-200 :,]
        
        variance = 0.2
        lengthscales = 0.5
        k1 = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
        noiseScale = 0.1
        alpha0 = 0.2
        jitter = 0.0
        
        #MODEL1 = SPAR_TF
        m_tf = SPAR_TF(k1, 1, noiseScale)
        m_tf.setData(data)
        m_tf.initialize()
        m_tf._jitter = jitter

        #MODEL2 = fast SPAR
        k2 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m2 = SPAR(k2, 1, alpha0)
        m2._jitter = jitter
        m2.setData(data)
        m2.initialize()
        m2.computeGradient()
        
        #MODEL3 = fast SPAR
        k3 = ConstantKernel(variance) * RBF(length_scale=lengthscales )
        m3 = SPAR(k3, 1, alpha0, noiseScale)
        m3._jitter = jitter
        m3.setData(data)
        m3.initialize()
        m3.computeGradient()

        self.m_tf = m_tf
        self.m2 = m2
        self.m3 = m3



    ###################################################
    #@unittest.SkipTest
    def test_parameters(self):

        m2 = self.m2
        m3 = self.m3
        
        m2.noise_trainable = False
        m3.noise_trainable = False
        
        p2 = m2.parameters
        p3 = m3.parameters
        
        self.assertAlmostEqual(len(p2), 3, places=6)
        self.assertAlmostEqual(p2[0], 0.2, places=6)
        self.assertAlmostEqual(p2[1], 0.5, places=6)
        self.assertAlmostEqual(p2[2], 0.2, places=6)
        
        self.assertAlmostEqual(len(p3), 4, places=6)
        self.assertAlmostEqual(p3[0], 0.2, places=6)
        self.assertAlmostEqual(p3[1], 0.5, places=6)
        self.assertAlmostEqual(p3[2], 0.1, places=6)
        self.assertAlmostEqual(p3[3], 0.2, places=6)
        
        tp2 = m2.trainable_parameters
        tp3 = m3.trainable_parameters
        
        self.assertAlmostEqual(len(tp2), 3, places=6)
        self.assertAlmostEqual(tp2[0], -1.60943791, places=6)
        self.assertAlmostEqual(tp2[1], -0.69314718, places=6)
        self.assertAlmostEqual(tp2[2], -1.60943791, places=6)
        
        self.assertAlmostEqual(len(tp3), 3, places=6)
        self.assertAlmostEqual(tp3[0], -1.60943791, places=6)
        self.assertAlmostEqual(tp3[1], -0.69314718, places=6)
        self.assertAlmostEqual(tp3[2], -1.60943791, places=6)
        
        m3.noise_trainable = True
        tp3 = m3.trainable_parameters
        self.assertAlmostEqual(len(tp3), 4, places=6)
        self.assertAlmostEqual(tp3[0], -1.60943791, places=6)
        self.assertAlmostEqual(tp3[1], -0.69314718, places=6)
        self.assertAlmostEqual(tp3[2], -2.30258509, places=6)
        self.assertAlmostEqual(tp3[3], -1.60943791, places=6)



class Test_gradient_conversion(unittest.TestCase):
    
    def setUp(self):
        
        def func(a) :
            return 2 *a**2
        
        self.f = func
        
        #dfunc(a)/da
        self.a = tf.Variable(5.0)
        
    #@unittest.SkipTest
    def test_norm_to_log(self):
        
        a = self.a
        f = self.f
 
        with tf.GradientTape() as tape:
           out = f(a )
        g1 = tape.gradient(out, a )
        self.assertAlmostEqual(g1.numpy(), a.numpy() * 4, places=5)
        
        # df(a)/dlog(a)
        b = tf.Variable(tf.math.log(a))
        with tf.GradientTape() as tape:
            out = f(tf.math.exp(b))
        g2 = tape.gradient(out, b)
        
        adj_factor = 1 / a 
        self.assertAlmostEqual(g1.numpy(), adj_factor.numpy() * g2.numpy(), places=5)

        #df(a)/d(log(exp(a) - 1))
        b = tf.Variable( log_space_transformation(a ))
        with tf.GradientTape() as tape:
            out = f(log_space_inverse_transformation(b))
        g3 = tape.gradient(out, b)

        #adjust
        factor = log_space_gradient_factor(a )
        self.assertAlmostEqual(g3.numpy() * factor.numpy(), g2.numpy(), places=5)
   

    ###################################################


if __name__ == '__main__':
    unittest.main()



