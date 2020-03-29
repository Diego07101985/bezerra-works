import numpy as np
import plot_ex1data1 as pe
import utils_ml as um
import math


def custo_regrlin_multi(theta, x, y):
    m = len(x)
    sm = np.power(x.dot(theta) - np.transpose([y]), 2)
    J = (1.0/(2*m)) * sm.sum( axis = 0 )
    return J