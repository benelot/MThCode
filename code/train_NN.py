""" Train a NN

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
paramses = []
model_type = [['parallel', 1000],
              ['general', 20]]

for i, val in enumerate(model_type):
    ids.append('test_' + val[0])

    paramses.append({'id': ids[-1],
                     'model_type': val[0],
                     'path2data': '../data/ID07_35h.mat',
                     # model parameters ------------------------
                     'channel_size': 20,
                     'reverse_nodes': False,
                     'hidden_size': 10,
                     'lambda': 0,
                     'non-linearity': 'relu',
                     'bias': False,
                     # train parameters -------------------------
                     'sample_begin': int(15 * 60 * 512),
                     'sample_size': 300,  # int(2 * 512),
                     'window_size': 30,
                     'normalization': True,
                     'epochs': val[1],
                     'lr_decay': None})

util.train_parallel(paramses[0])
util.train(paramses[1])


for i, val in enumerate(ids):
    if i == 0:
        #util.plot_optimization(val)
        util.make_prediction_parallel(val)
        util.make_distances(val)
        util.plot_prediction(val, [2, 6, 14, 18])
        util.plot_weights(val, linewidth=0)
    else:
        util.plot_optimization(val)
        util.make_prediction(val)
        util.make_distances(val)
        util.plot_prediction(val, [2, 6, 14, 18])
        util.plot_weights(val, linewidth=0)

util.plot_multi_boxplots(ids, x='id', y='correlation', save_name='parallel', ylim=(0, 1))

