import numpy as np 

# Essa fun¸c˜ao recebe a matriz de dados X de dados como parˆametro (na forma de um numpy
# array). Essa fun¸c˜ao realiza dois passos principais:
# • subtrai o valor m´edio de todas as caracter´ısticas do conjunto de dados X.
# • ap´os subtrair a m´edia, divide cada caracter´ıstica pelo seu respectivo desvio
# padr˜ao.

#  Normalizacao é uma pratica para evitar que seu algoritmo 
#  fique enviesado para as variáveis com maior ordem de grandeza.

# E z score normalization as variáveis vao resultar  em uma média proxima a 0 e um desvio padrão proximo a 1.

def normalizar_caracteristica(pmtr):
    c = len(pmtr[0])
    l = len(pmtr)
    mean = np.zeros(shape=(c), dtype=np.float64)
    std = np.zeros(shape=(c), dtype=np.float64)
    normalizar = np.copy(pmtr)

    for j in range(c):
        mean[j] = np.mean(pmtr[:,j])
        std[j] = pmtr[:,j].std()

    for i in range(l):
        for j in range(c):
            normalizar[i,j] = ((pmtr[i,j] - mean[j]) / std[j])

    return normalizar, mean , std

def normalizar_caracteristicas(pmtr, labels):
    c = len(pmtr[0])
    l = len(pmtr)

    ll = len(labels)

    mean_label = np.zeros(shape=(ll), dtype=np.float64)
    std_label = np.zeros(shape=(ll), dtype=np.float64)

    normalizar_label = np.copy(labels)

    mean_lb = np.mean(labels)
    std_lb = np.std(labels)

    for j in range(ll):
        normalizar_label[j] = ((labels[j] - mean_lb) / std_lb)
        
    mean = np.zeros(shape=(c), dtype=np.float64)
    std = np.zeros(shape=(c), dtype=np.float64)

    normalizar = np.copy(pmtr)

    for j in range(c):
        mean[j] = np.mean(pmtr[:,j])
        std[j] = pmtr[:,j].std()

    for i in range(l):
        for j in range(c):
            normalizar[i,j] = ((pmtr[i,j] - mean[j]) / std[j])
            
    return normalizar,normalizar_label, mean , std,mean_lb,std_lb


