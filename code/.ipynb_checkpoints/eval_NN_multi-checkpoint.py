""" Comparison of various NNs

Part of master thesis Segessenmann J. (2020)
"""
import utilities as util
import models
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ids = []
titles = ['0', '30', '90', '120', '150']
for i in range(len(titles)):
    ids.append('model_hidden_' + str(titles[i]))

ids = ['pat03_hidden_0', 'pat03_hidden_60']
util.plot_multi_boxplots(ids, x='id', y='correlation')
#util.plot_multi_boxplots(ids, x='hidden_size', y='mse')
#util.plot_multi_boxplots(ids, x='hidden_size', y='mae')
#util.plot_multi_scatter(ids, save_name='hidden_size__sigmoid')







