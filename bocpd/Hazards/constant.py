

import numpy as np
from bocpd.Hazards.hazards_model import HazardsModel


class ConstantHazards( HazardsModel):
   
    def __init__(self,theta):
        if isinstance(theta, list):
            assert (len(theta)==1)
            theta = theta[0]

        assert theta >= 0
        
        super().__init__(theta)
        self._num_hazard_params  = 1


    def evaluate(self,v):
        Ht = 1 / self._hazard_params * np.ones_like(v)
        dH = - (1 / self._hazard_params**2) * np.ones_like(v)
        return (Ht, dH)
    
    def log_evaluate(self,v):
        logH = - np.log(self._hazard_params) * np.ones_like(v)
        logmH = np.log(1 -  1 / self._hazard_params) * np.ones_like(v)
        
        dlogH = - (1 / self._hazard_params) * np.ones_like(v)
        dlogmH =  (1 / (self._hazard_params  * (self._hazard_params - 1))) * np.ones_like(v)
        return (logH, logmH, dlogH, dlogmH)
