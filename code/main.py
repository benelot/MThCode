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
          'sample_size': 10000,
          'window_size': 10,
          'normalization': True,
          'epochs': 20,
          'lr_decay': 7}

train_set, test_set = util.data_loader(params=params, train_portion=1, windowing=False)

spect, freq = pre.fft(train_set.numpy(), fs=512, filter=True)

fig, ax = plt.subplots()
ax.plot(freq, spect[:, 0])
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Frequency domain (spectrum) magnitude')
ax.set_xlim(0, 50)
plt.show()


