import numpy as np
from scipy.optimize import minimize
import linearRegCostFunction as lr


def learningCurve(X, y, Xval, yval, lambda_val):
    m = len(X)
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))

    for i in range(1,m+1):
        X_train = X[:i]
        y_train = y[:i]
        
        initial_theta = np.zeros((X.shape[1], 1))

        def costFunc(theta):
            return lr.linearRegCostFunction(X_train, y_train, theta, lambda_val, True)

        maxiter = 200
        results = minimize(costFunc, x0=initial_theta, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

        theta = results["x"]

        error_train[i-1] = lr.linearRegCostFunction(X_train, y_train, theta, 0)
        error_val[i-1]   = lr.linearRegCostFunction(Xval   , yval   , theta, 0)
                
    return error_train, error_val