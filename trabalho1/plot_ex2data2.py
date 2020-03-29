import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

DATA_BASE = "am-T1-dados"


#4.1 Visualiza¸c˜ao dos Dados

def load_microchip_base(properties_sell=DATA_BASE):
    os.makedirs(properties_sell, exist_ok=True)
    txt_path = os.path.join(properties_sell, "ex2data2.txt")
    return np.loadtxt(txt_path, delimiter=",")

data = load_microchip_base()


values = data[:,:2]
clasz  = data[:,2]

pos = data[np.where(clasz == 0)]
neg = data[np.where(clasz == 1)]

plt.figure(figsize=(10, 7))
plt.plot(pos[:,0], pos[:,1],'y.',marker='o',markersize=7, label="y=1");
plt.plot(neg[:,0], neg[:,1],'k.', marker='P',markersize=7, label="y=0");
plt.legend(loc=1)

plt.ylim = (-0.8, 1.5)
plt.xlim = (-1,1.8)
plt.xticks(np.arange(-1, 1.8,0.5))
plt.xlabel("Microchip 1", fontsize=10)
plt.ylabel("Microchip 2", rotation=90, fontsize=10)


