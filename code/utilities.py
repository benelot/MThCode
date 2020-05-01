# Utilities for iEEG connectome
# Segessenmann J. 2020

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


def data_loader(path: str, nr_channels: int, nr_samples: int, window_size: int, train_portion=0.8, normalization=True):
    """ Loads and prepares ieeg data for learning.

    """
    data_mat = loadmat(path)
    data = data_mat['EEG'][:nr_channels, :nr_samples].transpose()

    # Normalization
    if normalization:
        sc = MinMaxScaler(feature_range=(-1, 1))
        sc.fit(data)
        data = sc.transform(data)

    # To tensor
    data = torch.FloatTensor(data)

    # Split data into training and test set
    train_set = data[:int(train_portion * nr_samples), :]
    test_set = data[int(train_portion * nr_samples):, :]

    # Prepare data for learning
    X_train, X_test = [], []
    for i in range(train_set.shape[0] - window_size):
        X_train.append(train_set[i:i + window_size, :])
    for i in range(test_set.shape[0] - window_size):
        X_test.append(test_set[i:i + window_size, :])

    return X_train, X_test


def plot_weights(W, channel_size, vmax=1, linewidth=.5, absolute=False, path=None):
    vmin = -vmax
    cmap = 'RdBu'
    hticklabels = np.arange(channel_size, W.shape[0], 1)

    if absolute:
        vmin = 0
        cmap = 'Blues'
        W = np.abs(W)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(nrows=W.shape[0], ncols=W.shape[0])
    cbar_ax = fig.add_axes([.925, .125, .02, .755])  # l,b,w,h

    ax0 = fig.add_subplot(gs[:channel_size, :channel_size])
    ax0.set_ylabel('to visible nodes')
    ax0.get_xaxis().set_visible(False)

    ax1 = fig.add_subplot(gs[:channel_size, channel_size:])
    ax1.get_xaxis().set_visible(False), ax1.get_yaxis().set_visible(False)
    ax1.set_ylabel('to hidden nodes')
    ax1.set_xlabel('from visible nodes')

    ax2 = fig.add_subplot(gs[channel_size:, :channel_size])
    ax2.set_ylabel('to hidden nodes')
    ax2.set_xlabel('from visible nodes')

    ax3 = fig.add_subplot(gs[channel_size:, channel_size:])
    ax3.get_yaxis().set_visible(False)
    ax3.set_xlabel('from hidden nodes')

    sns.heatmap(W[:channel_size, :channel_size], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth,
                ax=ax0)
    sns.heatmap(W[:channel_size, channel_size:], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth,
                ax=ax1)
    sns.heatmap(W[channel_size:, :channel_size], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth,
                ax=ax2, yticklabels=hticklabels)
    sns.heatmap(W[channel_size:, channel_size:], cmap=cmap, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, linewidths=linewidth,
                ax=ax3, xticklabels=hticklabels)

    fig.text(0.08, 0.65, 'to visible node', va='center', ha='center', rotation='vertical')
    fig.text(0.08, 0.27, 'to hidden node', va='center', ha='center', rotation='vertical')
    fig.text(0.35, 0.08, 'from visible node', va='center', ha='center')
    fig.text(0.77, 0.08, 'from hidden node', va='center', ha='center')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()

    if path is not None:
        fig.savefig(path)

