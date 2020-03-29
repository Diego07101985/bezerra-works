#3.2.2 Fun¸c˜ao de custo e gradiente
import numpy as np
import sigmoide as si


def funcaoCustoRegressaoLogistica(theta, z, labels_norm):
    y = labels_norm
    m = len(y)
    term_1 = y * np.transpose(np.log(si.sigmoide(np.dot(z,theta)))) 
    term_2 = (1-y)* np.transpose(np.log(1-si.sigmoide(np.dot(z,theta))))
    
    return - 1/m * (term_1 + term_2).sum()