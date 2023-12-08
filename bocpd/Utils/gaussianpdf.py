#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np



def gaussianpdf(
    x,
    mu,
    var,
    nargout=1,
):
    
    norm = (x - mu) ** 2
    logp = norm / var + np.log(var) +  np.log( 2 * np.pi)
    p = np.exp( - 0.5 * logp )
    
    if nargout == 2 :
        N = len(mu)
        dp = np.zeros((N, 2))

        # Derivative for mu
        dp[:, 0] = p * (x - mu) / var

        # Derivative for var
        dp[:, 1] = -0.5 * p* ( 1/var - norm / var**2)
        
        return p, dp
    
    return p



def gaussianlogpdf(
    x,
    mu,
    var,
    nargout=1,
):
    
    norm = (x - mu) ** 2
    logp = norm / var + np.log(var) +  np.log( 2 * np.pi)
    p = - 0.5 * logp
    
    if nargout == 2 :
        N = len(mu)
        dp = np.zeros((N, 2))

        # Derivative for mu
        #dlogp[:, 0]
        dp[:, 0] = (x - mu) / var

        # Derivative for var
        dp[:, 1] = -0.5 * ( 1/var - norm / var**2)
        
        return p, dp
    
    return p

if __name__ == '__main__':

    import tensorflow as tf
    import tensorflow_probability as tfp

    mu_tf = tf.Variable(0.0)
    sigma2_tf = tf.Variable( 0.2)
    x = tf.constant(-0.85759774)

    with tf.GradientTape() as tape:
        pred_scale = tf.math.sqrt(sigma2_tf )
        pdf_tf = tfp.distributions.Normal(mu_tf, pred_scale, validate_args=False, allow_nan_stats=True).prob(x)
        out = tf.math.log(pdf_tf)
        #out = pdf_tf
    dmu_tf = tape.gradient(out, mu_tf)
    
    with tf.GradientTape() as tape:
        pred_scale = tf.math.sqrt(sigma2_tf )
        pdf_tf = tfp.distributions.Normal(mu_tf, pred_scale, validate_args=False, allow_nan_stats=True).prob(x)
        out = tf.math.log(pdf_tf)
        #out = pdf_tf
    dsigma2_tf = tape.gradient(out, sigma2_tf)

    (predprobs, dpredprobs) = gaussianlogpdf(x, np.resize(mu_tf,1), np.resize(sigma2_tf,1), 2)
    
    print("tf_pdf:= : {}".format(np.array(out)))
    print("np_pdf:= : {}".format(predprobs[0]))
    
    print("tf_dmu:= : {}".format(np.array(dmu_tf)))
    print("np_dmu:= : {}".format(dpredprobs[0][0]))
    
    print("tf_dsigma2:= : {}".format(np.array(dsigma2_tf)))
    print("np_dsigma2:= : {}".format(dpredprobs[0][1]))


        
  