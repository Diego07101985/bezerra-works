
import matplotlib.pyplot as plt
import numpy as np


def plotData(microchip_base, clasz):
    pos = np.where(clasz==1)
    neg = np.where(clasz==0)
    
    p1 = plt.plot(microchip_base[pos,0], microchip_base[pos,1], marker='+', markersize=9, color='k')[0]
    p2 = plt.plot(microchip_base[neg,0], microchip_base[neg,1], marker='o', markersize=7, color='y')[0]
    return plt, p1, p2