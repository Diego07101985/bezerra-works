import numpy as np
import matplotlib.pyplot as plt
import plot_ex1data1 as pe
import gd_reglin_uni as gd


def visualizar_reta():
    food_truck = pe.load_food_truck()
    learning_rate = 0.01

    x = food_truck['Profit in $ 10,000s']
    y = food_truck['Population in City in 10,000s']

    custo, theta = gd.gd_reglin_uni(x, y, learning_rate, 5000)

    precision_value = theta[0] + theta[1]*x


    plt.figure(figsize=(10, 7))
    plt.plot(x, precision_value, 'b-', label="Linear Regression")
    plt.plot(x, y, 'r.', marker='x', markersize=10, label="Training data")
    plt.legend(loc=4)
    plt.xlabel("Profit in $ 10,000s", fontsize=10)
    plt.ylabel("Population in City in 10,000s", rotation=90, fontsize=10)
    plt.ylim(-5, 25)
    plt.xlim = (4, 24)
    plt.xticks(np.arange(4, 25, 2))
