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
model_type = ['is', 'in', 'as']
for i, val in enumerate(model_type):
    ids.append('test_' + val)

    params = {'id': 'mtype_rv_relu_' + val,
              'model_type': val,
              'path2data': '../data/ID01_1h.mat',
              # model parameters ------------------------
              'channel_size': 20,
              'reverse_nodes': True,
              'hidden_size': 20,
              'lambda': 0.5,
              'non-linearity': 'relu',
              'bias': False,
              # train parameters -------------------------
              'sample_size': 3000,
              'window_size': 50,
              'normalization': True,
              'epochs': 12,
              'lr_decay': 7}

    util.train(params)

for i, val in enumerate(ids):
    util.plot_optimization(val)
    util.make_prediction(val)
    util.make_distances(val)
    util.plot_prediction(val, [2, 8, 12, 17])
    util.plot_weights(val, linewidth=0)


util.plot_multi_boxplots(ids, x='id', y='correlation', save_name='mtype_rv_relu')

