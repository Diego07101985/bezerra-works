import numpy as np
import utils_ml as um
# import sklearn
# from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import custo_reglin_uni as cr


def gd_reglin_uni(x, y, learning_rate, iterations):
    j_theta_0 = 0
    j_theta_1 = 0
    m = len(x)
    count = 0
    new_custo = 0
    custo = cr.custo_regrlin(j_theta_0, j_theta_1, x, y)

    while True:
        j_theta_0 = j_theta_0 - (learning_rate * (1/m *
                                                  sum([(um.hypotheses(j_theta_0, j_theta_1, np.asarray(
                                                      [x[i]]))) - y[i] for i in range(m)])))
        j_theta_1 = j_theta_1 - (learning_rate * (1/m *
                                                  sum([(um.hypotheses(j_theta_0, j_theta_1, np.asarray([x[i]])) - y[i]) * np.asarray([x[i]]) for i in range(m)])))

        new_custo = cr.custo_regrlin(j_theta_0, j_theta_1, x, y)
        count += 1
        if(count > iterations):
            custo = new_custo
            break
    return custo, [j_theta_0, j_theta_1]

