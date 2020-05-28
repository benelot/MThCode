""" Main file

Part of master thesis Segessenmann J. (2020)
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import time
import torch
import torch.nn as nn
import pickle
from os import path

import utilities as util
import models
import preprocess as pre

params = {'id': 'autocorr',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 60,
          'hidden_size': 0,
          'lambda': 0.5,
          'non-linearity': 'tanh',
          'bias': False,
          # train parameters -------------------------
          'sample_size': 3000,
          'window_size': 10,
          'normalization': True,
          'epochs': 20,
          'lr_decay': 7}

train_set, test_set = util.data_loader(params=params, train_portion=1, windowing=False)

n_clusters = 5

acf, conf = pre.acf(train_set.numpy(), n_lags=1000)
df = pre.node_reduction(acf, n_clusters)
plt.figure(figsize=(12, 8))
sns.set_style('whitegrid')
plt.plot([0, 1000], [conf[1], conf[1]], color='black', ls='--')
plt.plot([0, 1000], [conf[0], conf[0]], color='black', ls='--')
plt.xlim(0, 1000)
sns.lineplot(x='sample', y='value', data=df, hue='cluster', palette='colorblind')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Autocorrelation')
plt.savefig('../doc/figures/preprocess_acf')




