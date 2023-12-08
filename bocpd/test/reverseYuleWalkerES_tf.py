
import tensorflow as tf
from gpflow.config import default_float


def reverseYuleWalkerES_tf(
    rho,
    r,
    minLen,
    epsilon,
    ):

    n = r.shape[0]
    assert r.shape == (n, 1)

    r = r / rho
    y = tf.zeros((n-1, 1), dtype= default_float())
    Ysub = tf.zeros((n-1, n), dtype= default_float())
    Ysub = tf.transpose(tf.experimental.numpy.vstack([r[0,0] ,  tf.zeros((n-1, 1), dtype= default_float())]))
    y = tf.reshape(r[0,0], (1,1)) 

    for k in range(n - 1):

     
        beta = 1.0 - tf.transpose(r[:k + 1, :]) @ y

        num = r[k + 1, :] - tf.transpose(r[:k + 1, :]) @ y[::-1, :]
        alpha = num / beta

        z = y - alpha * y[::-1, :]

        y =  tf.experimental.numpy.vstack([z, alpha])

        newsub = tf.transpose(tf.experimental.numpy.vstack([y ,  tf.zeros((n-k-2, 1), dtype= default_float())]))
        Ysub = tf.experimental.numpy.vstack([Ysub, newsub])
    
        if k >= minLen and tf.math.reduce_max(tf.math.abs(Ysub[k + 1, :k + 2] - Ysub[k, :k + 2])) <= epsilon:
            break
        
    Ysub = Ysub[:k + 2, :k + 2]


    return Ysub


