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
          'reversed_nodes': False,
          'hidden_size': 0,
          'lambda': 0.5,
          'non-linearity': 'tanh',
          'bias': False,
          # train parameters -------------------------
          'sample_size': 100000,
          'window_size': 10,
          'normalization': True,
          'epochs': 20,
          'lr_decay': 7}

util.print_params('mtype_relu_is')







