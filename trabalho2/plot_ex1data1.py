import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FOOD_TRUCK = "am-T1-dados"


def load_food_truck(food_truck=FOOD_TRUCK):
    os.makedirs(food_truck, exist_ok=True)
    txt_path = os.path.join(food_truck, "ex1data1.txt")
    return pd.read_csv(txt_path, sep=",", header=None, names=["Profit in $ 10,000s", "Population in City in 10,000s"])


food_truck = load_food_truck()
food_truck.plot(kind="scatter", x="Profit in $ 10,000s", s=52, ylim=(-5, 25), xlim=(4, 24),
                xticks=np.arange(4, 25, 2), y="Population in City in 10,000s", marker='x', c='red', figsize=(10, 7))
