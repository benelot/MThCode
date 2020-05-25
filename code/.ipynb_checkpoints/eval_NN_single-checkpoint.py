""" Basic evaluation of one single NN

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

#id = 'model_lambda_0.1'
ids = []
titles = ['0', '30', '90', '120', '150']
for i in range(len(titles)):
    ids.append('model_hidden_' + str(titles[i]))

ids = ['pat01_hidden_0', 'pat01_hidden_60']
for i, val in enumerate(ids):
    util.plot_optimization(val)
    util.make_prediction(val)
    util.make_distances(val)
    #util.plot_prediction(val, [5, 26, 46, 54])#, lim_nr_samples=2000)
    util.plot_weights(val, linewidth=0)
    print(f'id: {val} done')
