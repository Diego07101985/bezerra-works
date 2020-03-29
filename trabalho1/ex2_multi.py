import numpy as np
import matplotlib.pyplot as plt
import plot_ex1data2 as pe
import gd_reglin_multi as gm
import normalizacao as na

properties_sell = pe.load_properties_sell()
learning_rate = 0.01

X_s = properties_sell[:,:2]
y = properties_sell[:,2]
m = len(y)


X_norm, mean, standard_deviation = na.normalizar_caracteristica(X_s)
X_norm = np.column_stack((np.ones((m,1)), X_norm))

theta,custo = gm.gd_reglin_multi(X_norm, y, learning_rate, 10)

print(theta)



