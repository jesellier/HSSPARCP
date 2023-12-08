

from bocpd.Hazards.hazards_model import HazardsModel
from gpflow.config import default_float

import tensorflow as tf



class ConstantHazards_TF( HazardsModel):
   
    def __init__(self,theta):
        if isinstance(theta, list):
            assert (len(theta)==1)
            theta = theta[0]

        assert theta >= 0
        
        super().__init__(theta)
        self._num_hazard_params  = 1
        self._hazard_params = tf.Variable(self._hazard_params, dtype=default_float())


    def evaluate(self,v):
        Ht = 1 / self._hazard_params * tf.ones_like(v, dtype=default_float())
        dH = - (1 / self._hazard_params**2) * tf.ones_like(v, dtype=default_float())
        return (Ht, dH)
    
    def log_evaluate(self,v):
        logH = - tf.math.log(self._hazard_params) * tf.ones_like(v, dtype=default_float())
        logmH = tf.math.log(1 -  1 / self._hazard_params) * tf.ones_like(v, dtype=default_float())

        return (logH, logmH)


