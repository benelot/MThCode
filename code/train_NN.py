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
model_type = [['0_single', 'corr', 'linear', False, 0],
              ['1_linear', 'general', 'linear', False, 0],
              ['2_nonlinear', 'general', 'relu', False, 0],
              ['3_hidden5', 'general', 'relu', False, 5],
              ['4_hidden20', 'general', 'relu', False, 20],
              ['5_reversed', 'general', 'relu', True, 20]]

for i, val in enumerate(model_type):
    ids.append('complexity_' + val[0])

    params = {'id': ids[-1],
              'model_type': val[1],
              'path2data': '../data/ID07_35h.mat',
              # model parameters ------------------------
              'channel_size': 20,
              'reverse_nodes': val[3],
              'hidden_size': val[4],
              'lambda': 0,
              'non-linearity': val[2],
              'bias': False,
              # train parameters -------------------------
              'sample_begin': int(15*60*512),
              'sample_size': int(2*512),
              'window_size': 30,
              'normalization': True,
              'epochs': 20,
              'lr_decay': 30}

    util.train(params)


for i, val in enumerate(ids):
    if val == 'complexity_0_single':
        util.make_prediction(val)
        util.make_distances(val)
        util.plot_prediction(val, [2, 6, 14, 18])
    else:
        util.plot_optimization(val)
        util.make_prediction(val)
        util.make_distances(val)
        util.plot_prediction(val, [2, 6, 14, 18])
        util.plot_weights(val, linewidth=0)

util.plot_multi_boxplots(ids, x='id', y='correlation', save_name='complexity', ylim=(0, 1))

