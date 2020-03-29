import visualizar_reta as vs
import numpy as np
import plot_ex1data1 as pe
import gd_reglin_uni as gd

vs.visualizar_reta()


def predict_value(x, theta):
    return theta[0] + theta[1]*np.array([x])


food_truck = pe.load_food_truck()
learning_rate = 0.01

x = food_truck['Profit in $ 10,000s']
y = food_truck['Population in City in 10,000s']

custo, theta = gd.gd_reglin_uni(x, y, learning_rate, 5000)

print("Para uma população de 35,000 habitantes , Preve um lucro de  {0}".format(
    float((predict_value(3.5, theta)*10000))))

print("Para uma população de 70,000 habitantes , Preve um lucro de  {0}".format(
    float((predict_value(7, theta)*10000))))
