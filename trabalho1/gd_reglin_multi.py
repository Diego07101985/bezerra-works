
import numpy as np



def gd_reglin_multi(x, y, learning_rate, iterations):
    theta = np.zeros((3, 1)) 
    m = len(y)
    count = 0
    new_custo = 0
    while True:
        theta = theta - learning_rate * (1.0/m) * np.transpose(x).dot(x.dot(theta) - np.transpose([y]))
        new_custo = crm.custo_regrlin_multi(theta, x[0], y)
        print("Iteration {0} custo {1}".format(count,new_custo))
        count += 1
        if(count > (iterations-1)):
            break
    return theta,new_custo
