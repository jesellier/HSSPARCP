# -*- coding: utf-8 -*-
"""

Library to compute weighted quantiles, including the weighted median, of
numpy arrays.
"""

import numpy as np


def wquantile(data, weights, quantile, interp = 'linear'):
 
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    cum_sorted_weights = np.cumsum(weights[ind_sorted])
    Cn =  cum_sorted_weights /  cum_sorted_weights[-1]

    if interp == 'lowest' :
        return sorted_data[Cn <= quantile][-1]
    if interp == 'highest' :
        return sorted_data[Cn > quantile][0]
    else :
        return np.interp(quantile, Cn, sorted_data)
    


if __name__ == "__main__" :
    
    quantile = 0.5
    data = np.array([10, 25, 30, 35, 50, 75])
    weights = np.array([1, 1, 4, 2, 2, 1 ])
    
    print("linear")
    q = wquantile(data, weights, quantile) 
    print(q)
    print(sum(weights[data <= q]) / sum(weights))
    print(sum(weights[data > q]) / sum(weights))
    print("")
    
    print("lowest")
    q = wquantile(data, weights, quantile, interp = 'lowest') 
    print(q)
    print(sum(weights[data <= q]) / sum(weights))
    print(sum(weights[data > q]) / sum(weights))
    print("")
    
    print("highest")
    q = wquantile(data, weights, quantile, interp = 'highest') 
    print(q)
    print(sum(weights[data <= q]) / sum(weights))
    print(sum(weights[data > q]) / sum(weights))
    print("")

