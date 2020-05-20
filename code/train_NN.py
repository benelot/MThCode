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


for i in range(2):
    a = i*60
    params = {'id': 'pat01_hidden_' + str(a),
              'path2data': '../data/ID01_1h.mat',
              # model parameters ------------------------
              'channel_size': 60,
              'hidden_size': a,
              'lambda': 0.5,
              'non-linearity': 'sigmoid',
              'bias': False,
              # train parameters -------------------------
              'sample_size': 2000,
              'window_size': 50,
              'normalization': True,
              'epochs': 15,
              'lr_decay': 7}

    util.train(params)