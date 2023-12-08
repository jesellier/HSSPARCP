import numpy as np
import time

from sklearn.model_selection import ParameterGrid
from bocpd.model_base import model_class


class TSCP_GridSearch():
    
    def __init__(self, X, bocpd, grid = None):
         
         self.X = X
         self.bocpd = bocpd
         self.bocpd.setData(X, grid)
        
         self.min_func = np.inf
         self.min_theta = None
         self.verbose = True
         self.iter = 0
         
         if self.bocpd.model.mtype != model_class.TSCP :
            raise ValueError("TS_Optimizer only accept TSCP models")
            
    
    def set_param_grid(self, entry_range = np.arange(1e-01, 2, 0.5), param_grid_dic = None) :
        
        self.bocpd.compute_metrics = False
        
        if param_grid_dic is None :
            param_grid_dic = {}
            entry_dic = self.bocpd.model.trainable_parameters
            
            for key in entry_dic.keys() : 
                param_grid_dic[key] = entry_range
                
        self.param_grid = ParameterGrid(param_grid_dic)

  
    def search(self, entry_range = np.arange(1e-01, 2, 0.5), param_grid_dic = None ) :
        
        self.set_param_grid(entry_range, param_grid_dic)
        
        for p in self.param_grid :
    
            t0 = time.time()
            self.bocpd.model.set_trainable_params(p)
            nlml, _, _ = self.bocpd.run(self.X)
            
            if nlml < self.min_func :
                self.min_func = nlml
                self.min_theta = p
                
            if self.verbose :
                self.iter +=1  
                print("model_{}".format(self.bocpd.model.name) + "_iter_{}".format(self.iter))   
                print("func:= : {}".format(nlml))
                print("time:= : {}".format(time.time() - t0))
                print("")
                
        self.bocpd.model.set_trainable_params(self.min_theta)
