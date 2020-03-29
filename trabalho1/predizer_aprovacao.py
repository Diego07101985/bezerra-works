import numpy as np
import sigmoide as si


def acuracia(theta, X):
    m = X.shape[0] 
    p = np.zeros((m, 1))
    sigValue = si.sigmoide( np.dot(X,theta) )
    p = sigValue >= 0.5
    return p

def predizer(values,theta):
    prob = si.sigmoide(np.dot(values,theta))
    return prob