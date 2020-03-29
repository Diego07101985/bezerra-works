
import numpy as np
import matplotlib.pyplot as plt
import plot_ex2data1 as pe
import normalizacao as na
import custo_reglog as cr
import gd_reglog as  gr
import predizer_aprovacao as pa
import sigmoide as si
import scipy.optimize as opt
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs

student_grade = pe.load_student_grade()
learning_rate = 1

initial_theta = np.array([0, 0, 0],ndmin=2)

values = student_grade[:,:2]
clasz  = student_grade[:,2]

m = len(clasz)


#0,693

examData_norm,label_norm,mean_examData, std_examData,mean_lb,std_lb = na.normalizar_caracteristicas(values,clasz)
examData_norm = np.column_stack((np.ones((m,1)), examData_norm))
initial_theta = np.transpose(initial_theta)

J = cr.funcaoCustoRegressaoLogistica(initial_theta, examData_norm,clasz)

print(J)

theta  = gr.gd_reglog(initial_theta,examData_norm,clasz)
print(theta)

# print(np.dot(,theta))
p = pa.predizer(np.array([1,45,35]),theta)
a = pa.acuracia(theta, examData_norm)
print('Predizer {0}'.format(p))
print('Train Accuracy: {0}'.format(np.mean(p == clasz) * 100))


myargs=(examData_norm, clasz)
m,n = examData_norm.shape

theta = np.array([0, 0, 0],ndmin=2)
# result = opt.fmin_tnc(func=cr.funcaoCustoRegressaoLogistica, x0=theta, fprime=gr.gd_reglog, args=myargs)
# custo  = cr.funcaoCustoRegressaoLogistica(result[0], examData_norm, clasz)
# print("Result {0}".format(result))
# print("Custo {0}".format(custo))

theta, cost_at_theta, _, _, _, _, _ = fmin_bfgs(cr.funcaoCustoRegressaoLogistica, x0=theta, args=myargs, full_output=True)


# myargs=(examData_norm, clasz)
# theta = fmin(cr.funcaoCustoRegressaoLogistica, x0=initial_theta, args=myargs)

# theta, cost_at_theta, _, _, _, _, _ = fmin_bfgs(cr.funcaoCustoRegressaoLogistica, x0=theta, args=myargs, full_output=True)
print('Cost at theta found by fmin: {:f}'.format(cost_at_theta))


# # Print theta to screen
# print('Cost at theta found by fmin: {:f}'.format(cost_at_theta))
# print('theta:'),
# print(theta)

myargs=(examData_norm, clasz)
initial_theta = np.array([0, 0, 0],ndmin=2)
result = opt.fmin_tnc(func=cr.funcaoCustoRegressaoLogistica, x0=initial_theta, fprime=gr.gd_reglog, args=myargs)
custo  = cr.funcaoCustoRegressaoLogistica(result[0], examData_norm, clasz)

# print("Result {0}".format(result))
# print("Custo {0}".format(custo))
