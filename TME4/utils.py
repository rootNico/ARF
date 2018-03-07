import numpy as np

def filter_values(x, y, a,b):
    indexes = np.where(np.logical_or(y == a, y == b))[0]
    return np.array(x[indexes]),np.array(y[indexes])

def one_againt_others(x, y, num):
    indexes = np.where(y == num)[0]
    new_y = np.ones(y.shape[0])
    new_y[indexes] = 0
    return x,new_y