""" Comparison of various NNs

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
titles = ['0', '30', '60', '90', '120']
for i in range(len(titles)):
    ids.append('Hidden_size_' + str(titles[i]))


util.plot_multi_boxplots(ids, x='hidden_size', y='correlation', hue='train_set')
util.plot_multi_boxplots(ids, x='hidden_size', y='mse', hue='train_set')
util.plot_multi_boxplots(ids, x='hidden_size', y='mae', hue='train_set')
util.plot_multi_scatter(ids, save_name='hidden_size')







