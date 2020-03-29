import numpy as np
import sigmoide as si

def costFunctionReg(theta, exam_norm, y, lambda_reg):
    m = len(y) 

    custo = 0
    grad = np.zeros(theta.shape)

    term1 = y * np.transpose(np.log(si.sigmoide(np.dot(exam_norm,theta))))
    term2 = (1-y) * np.transpose(np.log( 1 - si.sigmoide( np.dot(exam_norm,theta))))
    regularizar = ( float(lambda_reg) / (2*m)) * np.power(theta[1:theta.shape[0]],2).sum()
    custo = -(1./m)*(term1+term2).sum() + regularizar

    grad_no_regularization = (1./m) * np.dot(si.sigmoide( np.dot(exam_norm,theta) ).T - y, exam_norm).T
    grad_reg = (1./m) ** np.dot(si.sigmoide( np.dot(exam_norm,theta) ).T - y, exam_norm).T + ( float(lambda_reg) / m ) ** theta

    grad_reg[0] = grad[0]
    return custo 
    # return custo, grad_reg.flatten()

