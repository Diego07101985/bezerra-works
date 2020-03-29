import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import pandas as pd

DATA_BASE = "am-T1-dados"

# 3.1 Visualiza¸c˜ao dos dados

def load_student_grade(properties_sell=DATA_BASE):
    os.makedirs(properties_sell, exist_ok=True)
    txt_path = os.path.join(properties_sell, "ex2data1.txt")
    return np.loadtxt(txt_path, delimiter=",")

data = load_student_grade()


values = data[:,:2]
clasz  = data[:,2]

pos = data[np.where(clasz == 0)]
neg = data[np.where(clasz == 1)]

plt.figure(figsize=(10, 7))
plt.plot(pos[:,0], pos[:,1],'y.',marker='o',markersize=7, label="Not admitted");
plt.plot(neg[:,0], neg[:,1],'k.', marker='P',markersize=7, label="Admitted");
plt.legend(loc=4)
plt.xlabel("Exam 1 Score", fontsize=10)
plt.ylabel("Exam 2 Score", rotation=90, fontsize=10)



