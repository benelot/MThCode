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
model_type = [['relu'],
              ['sigmoid0'],
              ['sigmoid1'],
              ['sigmoid2'],
              ['sigmoid3'],
              ['linear']]

for i, val in enumerate(model_type):
    ids.append('test_' + val[0])

    params = {'id': ids[-1],
              'model_type': 'general',
              'path2data': '../data/ID07_32h.mat',
              # model parameters ------------------------
              'channel_size': 20,
              'reverse_nodes': False,
              'hidden_size': 10,
              'lambda': 0,
              'non-linearity': val[0],
              'bias': False,
              # train parameters -------------------------
              'sample_begin': 0,
              'sample_size': 2*512,
              'window_size': 30,
              'normalization': True,
              'epochs': 20,
              'lr_decay': 5}

    util.train(params)


for i, val in enumerate(ids):
    util.plot_optimization(val)
    util.make_prediction(val)
    util.make_distances(val)
    util.plot_prediction(val, [2, 6, 14, 18])
    util.plot_weights(val, linewidth=0)

util.plot_multi_boxplots(ids, x='id', y='correlation', save_name='test_models', ylim=(0, 1))

