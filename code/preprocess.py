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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator

import utilities as util
import models


def node_reduction(data: np.ndarray, n_clusters: int, max_n_clusters=20, n_components=12):
    """ Reduces node dimension to n_clusters.

    :param data: Data of size (n_samples, n_nodes)
    :param n_clusters: Number of clusters
    :param max_n_clusters: Maximum number of clusters for evaluation
    :param n_components: Number of PCA components
    :return: df: DataFrame
    """
    n_samples = data.shape[0]
    n_nodes = data.shape[1]

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data.T)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.set_title('First 3 principal components')
    plt.tight_layout()
    fig.savefig('../doc/figures/preprocess_pca.png')

    # Check score of K-Means for max_n_clusters
    n_clusters_list = list(range(1, max_n_clusters + 1))
    score = []
    for i, val in enumerate(n_clusters_list):
        km = KMeans(n_clusters=val)
        clusters = km.fit(pca_result)
        score.append(clusters.inertia_ / n_nodes)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    plt.plot(n_clusters_list, score)
    plt.xlabel('Number of clusters')
    plt.ylabel('Mean squared distance')
    plt.title('Distance to nearest cluster center')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig('../doc/figures/preprocess_kmeans_n_clusters.png')

    # Perform K-Means on data
    km = KMeans(n_clusters=n_clusters)
    clusters = km.fit(pca_result)

    df = pd.DataFrame(pca_result)
    df['Cluster'] = clusters.labels_
    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(x=0, y=1, data=df, s=50, hue='Cluster', style='Cluster', palette='colorblind')
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.title('Clustering of the first two princ. components')
    plt.tight_layout()
    fig.savefig('../doc/figures/preprocess_kmeans_result.png')

    # Make DataFrame
    df = pd.DataFrame()
    df['sample'] = np.tile(np.arange(0, n_samples), n_nodes)
    df['node'] = np.repeat(np.arange(0, n_nodes), n_samples)
    df['cluster'] = np.repeat(list(clusters.labels_), n_samples)
    df['value'] = data.T.flatten()

    return df


def acf(data: np.ndarray, n_lags, partial=False):
    """ Auto Correlation Function.

    :param data: Array of size (n_samples, n_nodes)
    :param n_lags: Lag size [samples]
    :param partial: Choose for partial or non-partial
    :return:
    """
    n_samples = data.shape[0]
    n_nodes = data.shape[1]

    lags = [i for i in range(n_lags)]
    acf_list = []

    if partial:
        for i in range(n_nodes):
            corr = [1. if lag == 0 else np.corrcoef(data[lag:, i], data[:-lag, i])[0][1] for lag in lags]
            acf_list.append(np.array(corr))
            acf_array = np.array(acf_list).T
    else:
        for i in range(n_nodes):
            mean = np.mean(data[:, i])
            var = np.var(data[:, i])
            data_p = data[:, i] - mean
            corr = [1. if lag == 0 else np.sum(data_p[lag:] * data_p[:-lag]) / n_samples / var for lag in lags]
            acf_list.append(np.array(corr))
            acf_array = np.array(acf_list).T

    return acf_array


def fft(data: np.ndarray, fs: float, filter=False):
    n_samples = data.shape[0]
    n_nodes = data.shape[1]

    spect = np.zeros((n_samples, n_nodes))
    for i in range(n_nodes):
        spect[:, i] = np.abs(fftpack.fft(data[:, i]))
    freq = fftpack.fftfreq(n_samples) * fs

    # Cut negavite frequencies
    spect_pos = spect[:len(freq[freq >= 0])]
    freq_pos = freq[freq >= 0]

    if filter:
        filtered = np.zeros((spect_pos.shape[0], n_nodes))
        for i in range(n_nodes):
            filtered[:, i] = savgol_filter(np.abs(spect_pos[:, i]), 49, 3)

        return filtered, freq_pos

    return spect_pos, freq_pos


