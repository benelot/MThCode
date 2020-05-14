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
titles = ['0', '30', '60', '90', '120']
for i in range(len(titles)):
    ids.append('Hidden_size_' + str(titles[i]))

"""
util.plot_optimization(id)
"""
for i, id in enumerate(ids):
    util.make_prediction(id)
    util.make_distances(id)

"""
df = pd.DataFrame(util.make_distances(id, train_set=True))
print('---------')
print('Mean Corr.: ' + str(df['correlation'].mean()))
print('Mean MSE.: ' + str(df['mse'].mean()))
print('Mean MAE.: ' + str(df['mae'].mean()))


df = pd.DataFrame(util.make_distances(id))
print('---------')
print('Mean Corr.: ' + str(df['correlation'].mean()))
print('Mean MSE.: ' + str(df['mse'].mean()))
print('Mean MAE.: ' + str(df['mae'].mean()))


util.plot_prediction(id, [5, 26, 46, 54], lim_nr_samples=2000)

util.plot_weights(id, linewidth=0)


"""