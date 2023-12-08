import numpy as np


class BOCPD():

    def __init__(self, model, hazard_function, compute_metrics  = False):
         self.model = model
         self.model.computeGradient(False)
         self.hazard_function = hazard_function
         self.compute_metrics  = compute_metrics 
         
    def setData(self, X, grid):
        self.model.setData(X, grid)
        

    def run(self, X, grid = None, verbose = False) :

        (T, D) = X.shape

        # Set up the placeholders for each parameter
        # R(r, t) = P(runlength_t-1 = r-1|X_1:t-1).
        # maxes = np.zeros(T + 1)
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1

        # The evidence at each time step i.e. Z(t) = P(X_t | X_1: t - 1).
        Z = np.zeros((T, 1))

        # Precompute all the gpr aspects of algorithm.
        self.setData(X, grid)
        self.model.compute_metrics = self.compute_metrics
        self.model.initialize()
        
        if self.compute_metrics : 
            M0 = np.zeros((T, 1))
            M2 = np.zeros((T, 1))

        for t in range(T) :
            
            #print("t:=" + str(t) + "\;")

            # Evaluate the predictive distribution for the new datum under each of
            # the parameters.  This is the standard thing from Bayesian inference.
            predprobs = self.model.pdf(t)

            # Evaluate the hazard function for this interval
            [H, _] = self.hazard_function.evaluate(np.array(range(t + 1)))
    
            # Evaluate the growth probabilities - shift the probabilities down and to
            # the right, scaled by the hazard function and the predictive
            # probabilities.
            R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

            # Evaluate the probability that there *was* a changepoint and we're
            # accumulating the mass back down at r = 0.
            R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)
    
            # Renormalize the run length probabilities for improved numerical
            # stability.
            Z[t] =  np.sum(R[: t + 2, t + 1])
            R[: t + 2, t + 1] = R[: t + 2, t + 1] / Z[t]
            
            #compute MSE
            if self.compute_metrics :
                M0[t] = np.sum(R[0 : t + 1, t] *  self.model.abs_error_t)
                M2[t] = np.sum(R[0 : t + 1, t] *  self.model.abs_error_2_t)
      
            # Update the parameter sets for each possible run length.
            self.model.update(t)
    
            #maxes[t] = R[:, t].argmax()
            
        Z = - np.log(Z)
        nlml = sum(Z)

        if self.compute_metrics :
            return nlml, R, (Z, M2, M0)
        else :
            return nlml, R, Z
    
    

