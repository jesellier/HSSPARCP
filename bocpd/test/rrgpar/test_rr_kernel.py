
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions


from bocpd.test.GP_base_tf import logexp_space_gradient_factor
from bocpd.test.RRGPAR_tf import  GaussianLaplaceReducedRank_TF
from bocpd.RRGPAR.gaussian_laplace_reduced_rank import  GaussianLaplaceReducedRank
import bocpd.generate_data as gd

from sklearn import preprocessing

import unittest

class Test_RRGPAR(unittest.TestCase):

    def setUp(self) :
        
        data = gd.import_data('D:/GitHub/bocpd/data/well.dat')
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data[-10 :,]
        X = data
        
        variance = 0.2
        lengthscales = 0.5

        k_tf = GaussianLaplaceReducedRank_TF(variance, lengthscales , n_features = 5, L = 3)
        k = GaussianLaplaceReducedRank(variance, lengthscales , n_features = 5, L = 3).fit()
        gradient_factor = logexp_space_gradient_factor(k_tf.parameters)
        
        self.X = X
        self.k_tf = k_tf
        self.k = k
        self.gradient_factor = gradient_factor 
        
        
    ###################################################
    #@unittest.SkipTest
    def test_lambda(self):
        
        k_tf = self.k_tf
        k = self.k
        gradient_factor = self.gradient_factor
        
        ii = 2

        with tf.GradientTape() as tape:
            out_tf =  k_tf.Λ()
            out_tf = out_tf[ii]
        dtf = tape.gradient(out_tf , k_tf.trainable_parameters)
        dtf = dtf * gradient_factor

        out, dout = k.Λ_with_grad()

        print("")
        print("tf_l:= : {}".format(out_tf))
        print("np_l:= : {}".format(out[ii]))
        print("")
        print("tf_l_grad:= : {}".format(np.array(dtf)))
        print("np_l_grad:= : {}".format(dout[:,ii]))
        

    ###################################################
    #@unittest.SkipTest
    def test_eval(self):
        
        k_tf = self.k_tf
        k = self.k
        X = self.X
        gradient_factor = self.gradient_factor

        with tf.GradientTape() as tape:
            out_tf = k_tf(X) 
            out_tf = out_tf[0,1]
        dtf = tape.gradient(out_tf , k_tf.trainable_parameters)
        dtf = dtf * gradient_factor

        out, dout = k(X, eval_gradient = True)
        
        print("")
        print("tf_k:= : {}".format(out_tf))
        print("np_k:= : {}".format(out[0,1]))
        print("")
        print("tf_k_grad:= : {}".format(np.array(dtf)))
        print("np_k_grad:= : {}".format(dout[0,1]))
        
    ###################################################
    #@unittest.SkipTest
    def test_eval_1(self):
        
        k_tf = self.k_tf
        k = self.k
        X = np.expand_dims(self.X[8],1)
        gradient_factor = self.gradient_factor

        with tf.GradientTape() as tape:
            out_tf = k_tf(X) 
            out_tf = out_tf[0,0]
        dtf = tape.gradient(out_tf , k_tf.trainable_parameters)
        dtf = dtf * gradient_factor

        out, dout = k(X, eval_gradient = True)
        
        print("")
        print("tf_k:= : {}".format(out_tf))
        print("np_k:= : {}".format(out[0,0]))
        print("")
        print("tf_k_grad:= : {}".format(np.array(dtf)))
        print("np_k_grad:= : {}".format(dout[0,0]))


        


        

if __name__ == '__main__':
    unittest.main()



