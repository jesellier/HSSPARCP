
import numpy as np
from bocpd.Hazards.hazards_model import HazardsModel


def logistic(x):
    return 1. / (1. + np.exp(-x))

def logit(p):
    return np.log(p) - np.log(1 - p)




class LogisticHazards(HazardsModel):
    
    def __init__(self, theta):
        super().__init__(theta)
        self._num_hazard_params  = 3
        
    @property
    def h(self) : 
        return self._hazard_params[0]
    
    @property
    def a(self) : 
        return self._hazard_params[1]
    
    @property
    def b(self) :  
        return self._hazard_params[2]


    def evaluate(self,v):
        # h(t) = h * logistic(at + b)
        # theta: [h, a, b]
        if len(v.shape) == 2:
            v = np.reshape(v, (v.shape[0], ))

        lp = logistic(self.a*v + self.b)

        lm = logistic(-self.a*v - self.b)
        H = self.h * lp
        # Derivatives
        dH = np.empty((v.shape[0], 3))
        dH[:, 0] = lp
        lp_lm_v = lp * lm * v
        dH[:, 1] = self.h * lp_lm_v
        dH[:, 2] = self.h * lp * lm

        return (np.atleast_2d(H).T, dH)
    
    
    def log_evaluate(self,v):
        # h(t) = h * logistic(at + b)
        # theta: [logit(h), a, b]
        # derived on p. 230 - 232 of DPI notebook

        if np.isscalar(v):
            v = np.asarray([v])
        T = v.shape[0]
        if len(v.shape) == 2:
            v = np.reshape(v, (T, ))

        h = logistic(self.h)
        logmh = self.loglogistic(-self.h)
        logh  = self.loglogistic(self.h)

        logisticNeg = logistic(-self.a*v - self.b)

        logH = self.loglogistic(self.a * v + self.b) + logh

        logmH = self.logsumexp(-self.a * v - self.b, logmh) + self.loglogistic(self.a * v + self.b)

        if len(logH.shape) == 1:
            logH = np.atleast_2d(logH).T

        if len(logmH.shape) == 1:
            logmH = np.atleast_2d(logmH).T

        # Derivatives

        dlogH      = np.zeros((T, 3))
        dlogH[:,0] = logistic(-self.h)
        dlogH[:,1] = logisticNeg*v
        dlogH[:,2] = logisticNeg

        dlogmH = np.zeros((T, 3))
        dlogmH[:,0] = self.dlogsumexp(-self.a * v - self.b, 0, logmh, -h)
        dlogmH[:,1] = self.dlogsumexp(-self.a * v - self.b, -v, logmh, 0) + v*logisticNeg
        dlogmH[:,2] = self.dlogsumexp(-self.a * v - self.b, -1, logmh, 0) + logisticNeg

        assert logH.shape   == (T, 1)
        assert logmH.shape  == (T, 1)
        assert dlogH.shape  == (T, 3)
        assert dlogmH.shape == (T, 3)

        return (logH, logmH, dlogH, dlogmH)
    
    
    def logsumexp(self,x, c):
        # function logZ = logsumexp(x, c)
        maxx = np.max([np.max(x),np.max(c)])
        if isinstance(maxx, np.ndarray):
            maxx[np.isnan(maxx)] = 0
            maxx[np.logical_not(np.isfinite(maxx))] = 0
        return np.log(np.exp(x - maxx) + np.exp(c - maxx)) + maxx


    def dlogsumexp(self,x,dx,c,dc):
        maxx = np.max([np.max(x), np.max(c)])
        if isinstance(maxx, np.ndarray):
            maxx[np.isnan(maxx)] = 0
            maxx[np.logical_not(np.isfinite(maxx))] = 0

        return (np.exp(x - maxx) * dx + np.exp(c - maxx) * dc) / (np.exp(x- maxx) + np.exp(c - maxx))


    def loglogistic(self,x):
    # function y = loglogistic(x)
        if isinstance(x, float):
            if x < 0:
                y = -np.log(np.exp(x) + 1.0) + x
            else:
                y = -np.log(1.0 + np.exp(-x))
        else:
            y = np.zeros_like(x)
            negx = x < 0
            nnegx = x >= 0
            y[negx] = -np.log(np.exp(x[negx]) + 1.0) + x[negx]
            y[nnegx] = -np.log(1.0 + np.exp(-x[nnegx]))
        return y


