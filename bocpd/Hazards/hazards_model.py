

from abc import ABC, abstractmethod


class HazardsModel(ABC):
    
    
    def __init__(self,theta):
         self._hazard_params = theta
    
    @property
    def parameters(self) :
       return self._hazard_params
    
    @property
    def num_params(self) :
     return self._num_hazard_params 
    

    def set_params(self, theta) :
        self._hazard_params = theta
        
        
    @abstractmethod
    def evaluate(self,v):
        raise NotImplementedError(
            "Parameters setter. Please define in separate class to override this function."
        )
        
    @abstractmethod
    def log_evaluate(self,v):
        raise NotImplementedError(
            "Parameters setter. Please define in separate class to override this function."
        )
        


