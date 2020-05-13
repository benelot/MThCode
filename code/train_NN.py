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


for i in range(3):
    params = {'id': 'test_0',
              'path2data': '../data/ID02_1h.mat',
              # model parameters ------------------------
              'visible_size': 60,
              'hidden_size': 60,
              'lambda': 0.5,
              'non-linearity': 'tanh',
              'bias': False,
              # train parameters -------------------------
              'sample_size': 500,
              'window_size': 50,
              'normalization': True,
              'epochs': 15,
              'lr_decay': 7}

    util.train(params)