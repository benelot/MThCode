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
titles = [0.1, 0.3, 0.5, 0.7, 0.9]
for i in range(5):
    ids.append('model_lambda_' + str(titles[i]))
"""
util.plot_multi_boxplots(ids, x='lambda', y='correlation')
util.plot_multi_boxplots(ids, x='lambda', y='mse')
util.plot_multi_boxplots(ids, x='lambda', y='mae')
"""
util.plot_multi_scatter(ids, save_name='lambda')







