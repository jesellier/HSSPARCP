from __future__ import division
import numpy as np
import pandas as pd
import copy

from sklearn import preprocessing



def import_data(loc):
    data = np.genfromtxt(
        loc,
        skip_header=1,
        skip_footer=1,
        names=True,
        dtype= None, delimiter=' ')

    return np.expand_dims(data.astype(float), 1)
    
def import_well_data():
    loc = data = 'D:/GitHub/bocpd/data/well.dat'
    data = np.genfromtxt(
        loc,
        skip_header=1,
        skip_footer=1,
        names=True,
        dtype= None, delimiter=' ')

    return (np.expand_dims(data.astype(float), 1), None)


def import_f_well_data(entry = '0'):
    loc = data = 'D:/GitHub/bocpd/data/well_f_' + entry + '.csv'
    df = pd.read_csv(loc)
    df = df.astype(float)
    data = df.values

    return (data, None)


def import_nile_data():
    loc = 'D:/GitHub/bocpd/data/nile.csv'
    df = pd.read_csv(loc, header=None )
    df[1] = df[1].astype(float)
    df[0] = df[0].astype(float)
    
    data = df.values

    return  (np.expand_dims(data[:,1],1), np.expand_dims(data[:,0],1))


def import_bee_data(do_angle_processing = True):
    loc = 'D:/GitHub/bocpd/data/bee_seq1.csv'
    df = pd.read_csv(loc)
    
    data = df.values[:,1:]
    true_cp = df.values[:, 0]
    data_original = copy.deepcopy(data)
    
    if do_angle_processing :
        data[ 1:,2] = data[ 1:,2] - data[ :-1,2]
        
        data = data[1:,:]
        true_cp = true_cp[1:]
        data_original = data_original[1:,:]
  
    return data, true_cp, data_original




def import_snowfall_data():
    loc = 'D:/GitHub/bocpd/data/whistler_data.csv'
    df = pd.read_csv(loc )

    return df.values, None 
    
    

def generate_normal_time_series(num, minl: int=50, maxl: int=1000, seed: int=100):
    np.random.seed(seed)
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn() * 10
        var = np.random.randn() * 1
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return partition, np.atleast_2d(data).T


def generate_multinormal_time_series(num, dim, minl: int=50, maxl: int=1000, seed: int=100):
    np.random.seed(seed)
    data = np.empty((1, dim), dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.standard_normal(dim) * 10
        # Generate a random SPD matrix
        A = np.random.standard_normal((dim, dim))
        var = np.dot(A, A.T)

        tdata = np.random.multivariate_normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return partition, data[1:, :]


def generate_xuan_motivating_example(minl: int=50, maxl: int=1000, seed: int=100):
    np.random.seed(seed)
    dim = 2
    num = 3
    partition = np.random.randint(minl, maxl, num)
    mu = np.zeros(dim)
    Sigma1 = np.asarray([[1.0, 0.75], [0.75, 1.0]])
    data = np.random.multivariate_normal(mu, Sigma1, partition[0])
    Sigma2 = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    data = np.concatenate(
        (data, np.random.multivariate_normal(mu, Sigma2, partition[1]))
    )
    Sigma3 = np.asarray([[1.0, -0.75], [-0.75, 1.0]])
    data = np.concatenate(
        (data, np.random.multivariate_normal(mu, Sigma3, partition[2]))
    )
    return partition, data







