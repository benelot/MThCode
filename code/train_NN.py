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
window_sizes = [30, 50, 200, 500, 1000]
for i, val in enumerate(window_sizes):
    ids.append('wsize_' + str(val))
    params = {'id': 'wsize_' + str(val),
              'path2data': '../data/ID01_1h.mat',
              # model parameters ------------------------
              'channel_size': 60,
              'hidden_size': 60,
              'lambda': 0.8,
              'non-linearity': 'sigmoid',
              'bias': False,
              # train parameters -------------------------
              'sample_size': 5000,
              'window_size': val,
              'normalization': True,
              'epochs': 16,
              'lr_decay': 7}
    util.train(params)

for i, val in enumerate(ids):
    util.plot_optimization(val)
    util.make_prediction(val)
    util.make_distances(val)
    util.plot_prediction(val, [5, 26, 46, 54], lim_nr_samples=2000)
    util.plot_weights(val, linewidth=0)


util.plot_multi_boxplots(ids, x='id', y='correlation')

