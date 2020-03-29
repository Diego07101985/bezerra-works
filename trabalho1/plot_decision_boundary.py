import mapFeature as mf
import plot_data as pd


# 4.4 Esbo¸co da fronteira de decis˜ao

def plotDecisionBoundary(theta, microchip_base , clasz):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 7))

    plt, p1, p2 = pd.plotData(microchip_base[:,1:3], clasz)
   
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros(( len(u), len(v) ))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mf.mapFeature(np.array([u[i]]), np.array([v[j]])),theta)
    z = np.transpose(z) 

    p3 = plt.contour(u, v, z, levels=[0], linewidth=2).collections[0]
    
    plt.legend((p1,p2, p3),('y = 1', 'y = 0', 'Decision Boundary'), numpoints=1, handlelength=0)
    plt.show()