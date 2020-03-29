import numpy as np
import plot_ex1data1 as pe
import utils_ml as um
import math


def custo_regrlin(t0, t1, x, y):
    m = len(x)
    return 1/2/m * sum([math.pow((um.hypotheses(t0, t1, np.asarray([x[i]])) - y[i]), 2) for i in range(m)])
