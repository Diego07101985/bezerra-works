import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import linearRegCostFunction as lr


mat = scipy.io.loadmat('am-T1-dados/ex5data1.mat')

X = mat["X"]
y = mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]
Xtest = mat["Xtest"]
ytest = mat["ytest"]

m = X.shape[0]

plt.figure(figsize=(10, 7))
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show(block=False)

theta = np.array([[1] , [1]])
X_padded = np.column_stack((np.ones((m,1)), X))
J = lr.linearRegCostFunction(X_padded, y, theta, 1)

print('Cost at theta = [1 ; 1]: {:f}\n(this value should be about 303.993192)\n'.format(J))



initial_theta = np.zeros((X_padded.shape[1], 1))

lambda_val = 0

def costFunc(theta):
    return lr.linearRegCostFunction(X_padded, y, theta, lambda_val, True)

maxiter = 200
results = minimize(costFunc, x0=initial_theta, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

theta = results["x"]

plt.figure(figsize=(10, 7))
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.dot(np.column_stack((np.ones((m,1)), X)), theta), '--', linewidth=2)
