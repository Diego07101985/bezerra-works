
import numpy as np
import plot_ex2data2 as pe
import costFunctionReg as cr
import gd_reglog as  gr
import sigmoide as si
import mapFeature as mf
import plot_decision_boundary as pb
import plot_data as pd
from scipy.optimize import fmin_bfgs





microchips = pe.load_microchip_base()
# learning_rate = 1


values = microchips[:,:2]
clasz  = microchips[:,2]

m = len(clasz)

values = mf.mapFeature(values[:,0], values[:,1])
m,n = values.shape
initial_theta = np.zeros((n, 1))


#0,693

lambda_r = 1
initial_theta = np.transpose(initial_theta)
custo,theta = cr.costFunctionReg(np.transpose(initial_theta), values,clasz,lambda_r)
# print('theta  {0}'.format(theta))
print('Cost at initial theta (zeros): {:f}'.format(custo))


initial_theta = np.zeros((n, 1))
myargs=(values, clasz, lambda_r)
theta = fmin_bfgs(cr.costFunctionReg, x0=initial_theta, args=myargs)

pb.plotDecisionBoundary(theta,values,clasz)

# plt.xlabel('Microchip Test 1')
# plt.ylabel('Microchip Test 2')
# plt.title('lambda = {:f}'.format(lambda_r))

# theta  = gr.gd_reglog(initial_theta,examData_norm,clasz)
# print(theta)

# # print(np.dot(,theta))
# p = pa.predizer(np.array([1,45,35]),theta)
# a = pa.acuracia(theta, examData_norm)
# print('Predizer {0}'.format(p))
# print('Train Accuracy: {0}'.format(np.mean(p == clasz) * 100))


# myargs=(examData_norm, clasz)
# m,n = examData_norm.shape

# theta = np.array([0, 0, 0],ndmin=2)
# # result = opt.fmin_tnc(func=cr.funcaoCustoRegressaoLogistica, x0=theta, fprime=gr.gd_reglog, args=myargs)
# # custo  = cr.funcaoCustoRegressaoLogistica(result[0], examData_norm, clasz)
# # print("Result {0}".format(result))
# # print("Custo {0}".format(custo))

# theta, cost_at_theta, _, _, _, _, _ = fmin_bfgs(cr.funcaoCustoRegressaoLogistica, x0=theta, args=myargs, full_output=True)


# # myargs=(examData_norm, clasz)
# # theta = fmin(cr.funcaoCustoRegressaoLogistica, x0=initial_theta, args=myargs)

# # theta, cost_at_theta, _, _, _, _, _ = fmin_bfgs(cr.funcaoCustoRegressaoLogistica, x0=theta, args=myargs, full_output=True)
# print('Cost at theta found by fmin: {:f}'.format(cost_at_theta))


# # # Print theta to screen
# # print('Cost at theta found by fmin: {:f}'.format(cost_at_theta))
# # print('theta:'),
# # print(theta)

# myargs=(examData_norm, clasz)
# initial_theta = np.array([0, 0, 0],ndmin=2)
# result = opt.fmin_tnc(func=cr.funcaoCustoRegressaoLogistica, x0=initial_theta, fprime=gr.gd_reglog, args=myargs)
# custo  = cr.funcaoCustoRegressaoLogistica(result[0], examData_norm, clasz)

# print("Result {0}".format(result))
# print("Custo {0}".format(custo))
