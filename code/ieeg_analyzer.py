import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy import fftpack
from sklearn.preprocessing import MinMaxScaler
import time
import torch
import torch.nn as nn
import pickle
from os import path
from pandas.plotting import autocorrelation_plot
import sklearn as sk
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator

import utilities as util
import models

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

pca = PCA(n_components=12)
pca_result = pca.fit_transform(train_set.numpy().T)
print(pca_result.shape)

# Plot the 1st principal component aginst the 2nd and use the 3rd for color
fig, ax = plt.subplots()
ax.scatter(pca_result[:, 0], pca_result[:, 1])
ax.set_xlabel('1st principal component')
ax.set_ylabel('2nd principal component')
fig.subplots_adjust()
plt.show()

from sklearn.cluster import KMeans
ax = plt.figure(figsize=(8, 8)).gca()
max_n_clusters = 30
n_clusters = list(range(1, max_n_clusters))
score = []
for i, val in enumerate(n_clusters):
    kmeans = KMeans(n_clusters=val)
    clusters = kmeans.fit(pca_result)
    score.append(clusters.inertia_/params['channel_size'])
plt.plot(n_clusters, score)
plt.xlabel('Number of clusters')
plt.ylabel('Mean squared distance')
plt.xlim((0, max_n_clusters))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


n_clusters = 5
km = KMeans(n_clusters=n_clusters)
clusters = km.fit(pca_result)

df = pd.DataFrame(pca_result)
df['label'] = clusters.labels_

fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.scatterplot(x=0, y=1, data=df, hue='label', palette='colorblind')
ax.set_xlabel('1st principal component')
ax.set_ylabel('2nd principal component')


def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices


X = train_set.numpy()
sample_size = train_set.shape[0]
node_size = train_set.shape[1]

df = pd.DataFrame()
df['sample'] = np.tile(np.arange(0, sample_size), node_size)
df['node'] = np.repeat(np.arange(0, node_size), sample_size)
df['cluster'] = np.repeat(list(clusters.labels_), sample_size)

fig, ax = plt.subplots(figsize=(8, 8))
#for i in range(n_clusters):
#    sns.lineplot(data=df[all_indices(0, list(clusters.labels_))])
plt.show()
