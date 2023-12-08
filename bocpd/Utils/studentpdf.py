#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gammaln, digamma, gamma


def studentpdf(
    x,
    mu,
    var,
    nu,
    nargout=1,
):

  #
  # p = studentpdf(x, mu, var, nu)
  #
  # Can be made equivalent to MATLAB's tpdf() by:
  # tpdf((y - mu) ./ sqrt(var), nu) ./ sqrt(var)
  # Equations found in p. 577 of Gelman

   #This form is taken from Kevin Murphy's lecture notes.
    c = np.exp(gammaln(nu / 2 + .5) - gammaln(nu / 2) ) * (nu * np.pi * var) ** (-0.5)
    p = c * (1.0 + (1.0 / (nu * var)) * (x - mu) ** 2) ** (-0.5 * (nu + 1))

    if nargout == 2:
        N = len(mu)
        dp = np.zeros((N, 3))

        error = (x - mu) / np.sqrt(var)
        sq_error = (x - mu) ** 2 / var

        # Derivative for mu
        dlogp = (1.0 / np.sqrt(var)) * ((nu + 1.0) * error) / (nu + sq_error)
        dp[:, 0] = p * dlogp

        # Derivative for var
        dlogpdprec = np.sqrt(var) - ((nu + 1) * (x - mu) * error) / (nu + sq_error)
        dp[:, 1] = -.5 * (p / var ** 1.5) * dlogpdprec

        # Derivative for nu (df)
        dlogp = digamma(nu / 2.0 + .5) - digamma(nu / 2.0) - (1.0 / nu) - np.log( 1.0 + (1.0 / nu) * sq_error) + ((nu + 1) * sq_error) / (nu ** 2 + nu * sq_error)
        dp[:, 2] = .5 * p * dlogp

        return (p, dp)
    return p




def studentlogpdf(
    x,
    mu,
    var,
    nu,
    nargout=1,
    ):


  # p = studentpdf(x, mu, var, nu)
  # Can be made equivalent to MATLAB's tpdf() by:
  # tpdf((y - mu) ./ sqrt(var), nu) ./ sqrt(var)
  # Equations found in p. 577 of Gelman

    computeDerivs = nargout == 2

    error = (x - mu) / np.sqrt(var)
    sq_error = (x - mu) ** 2 / var

  # This form is taken from Kevin Murphy's lecture notes.
    c = gammaln(nu / 2 + .5) - gammaln(nu / 2) - .5 * np.log(nu * np.pi
            * var)
    p = c + -(nu + 1) / 2 * np.log(1 + sq_error / nu)

    if computeDerivs:
        N = len(mu)
        dp = np.zeros((N, 3))

    # Derivative for mu
        dp[:, 0] = 1 / np.sqrt(var) * ((nu + 1) * error) / (nu + sq_error)
    # Derivative for var
        dlogpdprec = np.sqrt(var) - (nu + 1) * np.sqrt(var) * sq_error / (nu + sq_error)
        dp[:, 1] = -.5 * (1 / var ** 1.5) * dlogpdprec

    # Derivative for nu (df)
        dlogp = digamma(nu / 2 + .5) - digamma(nu / 2) - 1 / nu \
            - np.log(1 + 1 / nu * sq_error) + (nu + 1) * sq_error / (nu
                ** 2 + nu * sq_error)
        dp[:, 2] = .5 * dlogp

        return (p, dp)
    return p




# if __name__ == '__main__':

#     import tensorflow as tf
#     import tensorflow_probability as tfp

#     mu_tf = tf.Variable(0.0)
#     sigma2_tf = tf.Variable( 0.2)
#     x = tf.constant(-0.85759774)

#     with tf.GradientTape() as tape:
#         pred_scale = tf.math.sqrt(sigma2_tf )
#         pdf_tf = tfp.distributions.StudentT(2, mu_tf, pred_scale, validate_args=False, allow_nan_stats=True).prob(x)
#         #out = tf.math.log(pdf_tf)
#         out = pdf_tf
#     g = tape.gradient(out, sigma2_tf)

       
#     (logpredprobs, dlogpredprobs) = studentpdf(x, np.resize(mu_tf,1), np.resize(sigma2_tf,1), 2, 2)

        
   