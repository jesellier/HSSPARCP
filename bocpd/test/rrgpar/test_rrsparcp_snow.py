
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

from bocpd.RRGPAR.RRGPARCP import RRSPARCP
from bocpd.RRGPAR.gaussian_laplace_reduced_rank import  GaussianLaplaceReducedRank
import bocpd.generate_data as gd

import unittest


class Test_RRSPARCP(unittest.TestCase):

    def setUp(self) :
        
        data, _ = gd.import_snowfall_data()
        X = (data - np.mean(data)) / np.std(data)
        
        eval_cutoff = 1000
        X_train = X[: eval_cutoff ]
        L = 19.86629956

        variance = 0.2
        lengthscales = 0.5
        noise_parameter  = 0.5
        prior_parameter  = 0.1

        #MODEL2 = fast RRSPAR
        k = GaussianLaplaceReducedRank(variance, lengthscales , n_features = 5, L = L)
        m = RRSPARCP(k, 1,  prior_parameter, noise_parameter )
        m.set_trainable_parameters = np.array([ 0.69779095,  0.69715908, -7.22166585,  1.01854056])
        m.maxLen = 300
        
        m.setData(X_train)
        m.computeGradient(False)
        m.initialize()

        self.m = m
        
    

    #@unittest.SkipTest
    # TEST SSE UPDATE
    def test_SSE_update(self):

        m = self.m

        n = 331
        m.noise_trainable = True
 
        ############## NP
        m.horizontal_update = True
        m.fast_computation = True
        m.initialize()
        
        #Fast calculation
        for i in range(n + 1):
            out = m.pdf(i)
        
        SSE_t = m.SSE_t

        #RECALC
        SSE_rec = np.zeros((n + 1,1))
        yp = m.X[ : n ]
        xp = m.lagMatrix[:n,:,0]

        for ii in range(1, n + 1) : 
            Kss2 = m.kernel(xp[-ii:])
            Kss2[np.diag_indices_from(Kss2)] += m.total_noise_variance
            Kinv = np.linalg.inv(Kss2)
            SSE_rec[ii]  =  yp[-ii:].T @ Kinv @ yp[-ii:]

        print("TEST_SSE("+ str(n) +")")
        for i in range(n+1):
            print("[rec, tf, np]:= :[{}".format(SSE_rec[i][0]) +  ",{}".format(SSE_t[i]) + "]")
            #self.assertAlmostEqual(SSE_rec[i][0], SSE_t[i], places=7)
            
        print("")
        print(out)
            




if __name__ == '__main__':
    unittest.main()



