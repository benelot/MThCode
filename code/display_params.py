""" Displays parameter of a certain model

Part of master thesis Segessenmann J. (2020)
"""

import pickle

params = pickle.load(open('../models/FRNN_tanh_big.pkl', 'rb'))

for keys, values in params.items():
    print(f'{keys}: {values}')
