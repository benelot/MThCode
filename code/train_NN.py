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

params0 = {'id': 'test_0',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 60,
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

params1 = {'id': 'test_1',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 60,
          'hidden_size': 60,
          'lambda': 0.5,
          'non-linearity': 'sigmoid',
          'bias': False,
          # train parameters -------------------------
          'sample_size': 500,
          'window_size': 50,
          'normalization': True,
          'epochs': 15,
          'lr_decay': 7}

params2 = {'id': 'test_2',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 60,
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

util.train(params0)
util.train(params1)
util.train(params2)
