""" Main file

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

util.plot_optim(model_id='gradient_norm')

"""
preds = [[] for i in range(6)]
trues = [[] for i in range(6)]
names = ['tanh', 'tanh_bias', 'relu', 'relu_bias', 'sig', 'sig_bias']
results = [[] for i in range(6)]

for i, name in enumerate(names):
    preds[i] = pd.read_csv('../results/FRNN_parallel_big_' + name + '_preds.csv').values
    trues[i] = pd.read_csv('../results/FRNN_parallel_big_' + name + '_true.csv').values
    results[i] = pickle.load(open('../results/FRNN_parallel_big_' + name + '_results.pkl', 'rb'))

df = pd.DataFrame()
for i in range(len(names)):
    df = df.append(pd.DataFrame(results[i]), ignore_index=True)
util.box_plots(df, x='Non-Linearity', y='Correlation', ylim=(0, 1), hue='Bias')

util.scatter_plots(names, preds, trues)


idx = [13, 24, 46, 58]
for i, name in enumerate(names):
    stridx= []
    for k in range(len(idx)):
        stridx.append(name + ', node: ' + str(idx[k]))
    util.prediction_plots(stridx, preds[i][-1000:, idx], trues[i][-1000:, idx], save_name=name)

for i, name in enumerate(names):
    params = pickle.load(open('../models/FRNN_parallel_big_' + name + '.pkl', 'rb'))
    util.plot_weights(params, linewidth=.0)
"""








