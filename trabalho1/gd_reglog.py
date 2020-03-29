
import numpy as np
import custo_reglog as cr
import sigmoide as si



def gd_reglog(theta,exam_norm,clasz):
    learning_rate = 0.5
    iter = 500
    y = clasz
    m = len(y)

    new_custo = 0

    for i in range(iter):   
        # grad = (1./m)*(np.transpose(exam_norm).dot(si.sigmoide(exam_norm.dot(theta)) - np.transpose([y])))
        grad = (1./m) * np.dot(si.sigmoide( np.dot(exam_norm,theta) ).T - y, exam_norm).T 
        theta = theta - learning_rate * grad
        new_custo = cr.funcaoCustoRegressaoLogistica(theta,exam_norm, clasz)
        # print("Iteration {0} custo {1}".format(i,new_custo))
        # print("theta {0}".format(theta))
    return theta


    