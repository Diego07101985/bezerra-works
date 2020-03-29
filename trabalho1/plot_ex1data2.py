import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

DATA_BASE = "am-T1-dados"

# Nessa parte do trabalho, vocˆe ir´a estudar uma implementa¸c˜ao da regress˜ao
# linear com m´ultiplas vari´aveis para predizer o pre¸co de venda de im´oveis. O
# arquivo ex1data2.txt cont´em informa¸c˜oes acerca de pre¸cos de im´oveis. A
# primeira coluna corresponde ao tamanho do im´ovel (em p´es quadrados1
# ). A
# segunda coluna corresponde `a quantidade de dormit´orios no im´ovel em quest˜ao.
# A terceira coluna corresponde ao pre¸co do im´ovel.


def load_properties_sell(properties_sell=DATA_BASE):
    os.makedirs(properties_sell, exist_ok=True)
    txt_path = os.path.join(properties_sell, "ex1data2.txt")
    return np.loadtxt(txt_path, delimiter=",")




