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
model_type = ['norm01_sigm']
for i, val in enumerate(model_type):
    ids.append('test_' + val)

    params = {'id': 'test_' + val,
              'model_type': 'in',
              'path2data': '../data/ID02_1h.mat',
              # model parameters ------------------------
              'channel_size': 20,
              'reverse_nodes': False,
              'hidden_size': 20,
              'lambda': 0.5,
              'non-linearity': 'sigm',
              'bias': False,
              # train parameters -------------------------
              'sample_begin': 0,
              'sample_size': 3000,
              'window_size': 30,
              'normalization': True,
              'epochs': 12,
              'lr_decay': 7}

    util.train(params)

for i, val in enumerate(ids):
    util.plot_optimization(val)
    util.make_prediction(val)
    util.make_distances(val)
    util.plot_prediction(val, [2, 6, 14, 18])
    util.plot_weights(val, linewidth=0)

util.plot_multi_boxplots(ids, x='id', y='correlation', save_name='norm01', ylim=(0, 1))

