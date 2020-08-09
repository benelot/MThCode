import copy

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy import fftpack
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats.kde import gaussian_kde
from matplotlib.ticker import MaxNLocator
from matplotlib import animation, rc
import hdf5storage
import pwlf
import time
from IPython.display import display, clear_output
from sklearn.preprocessing import StandardScaler
from obspy.signal import cross_correlation
import copy


def node_reduction(data: np.ndarray, n_clusters: int, max_n_clusters=20, n_components=12, sample_labels=None, plot=True):
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

    if plot:
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

    if plot:
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

    if plot:
        fig = plt.figure(figsize=(5, 5))
        palette = sns.color_palette('muted', n_clusters)
        sns.scatterplot(x=0, y=1, data=df, s=50, hue='Cluster', style='Cluster', palette=palette)
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
    if sample_labels is not None:
        df['sample_label'] = np.tile(sample_labels, n_nodes)

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

    # Confidence
    conf = (1.96/np.sqrt(n_samples), -1.96/np.sqrt(n_samples))

    return acf_array, conf


def fft(data: np.ndarray, fs: float, downscale_factor=None):
    """

    :param data:
    :param fs:
    :param downscale_factor:
    :return:
    """
    n_samples = data.shape[0]
    n_nodes = data.shape[1]

    spect = np.zeros((n_samples, n_nodes))
    for i in range(n_nodes):
        spect[:, i] = np.abs(fftpack.fft(data[:, i]))
    freq = fftpack.fftfreq(n_samples) * fs

    # Cut negative frequencies
    spect_pos = spect[:len(freq[freq >= 0])]
    freq_pos = freq[freq >= 0]

    if downscale_factor is not None:
        spect_resampled = []
        for i in range(n_nodes):
            spect_filtered = savgol_filter(np.abs(spect_pos[:, i]), int(n_samples/500)-1, 3)
            spect_resampled.append(signal.resample_poly(spect_filtered, up=3, down=3*downscale_factor))
        freq_resampled = signal.resample_poly(freq_pos, up=3, down=3*downscale_factor)

        return np.array(spect_resampled).T, freq_resampled

    return spect_pos, freq_pos


def distribution(data: np.ndarray, xlim: tuple=None):
    """

    :param data:
    :param xlim:
    :return:
    """
    n_nodes = data.shape[1]

    if xlim is None:
        xlim = (np.min(data), np.max(data))
    kde_x = np.linspace(xlim[0], xlim[1], 500)
    density = []
    for i in range(n_nodes):
        kde = gaussian_kde(data[:, i])
        density.append(kde(kde_x))

    return kde_x, np.array(density).T


def plot_distribution(data: np.ndarray, xlim: tuple=None, n_clusters: int=6, qq_plot: bool=True, title: str=''):
    print('Status: Get distributions.')
    kde_x, kde = distribution(data, xlim=xlim)

    print('Status: Reduction of nodes.')
    df_kde = node_reduction(kde, n_clusters=n_clusters, sample_labels=kde_x, plot=False)

    sns.set_style('whitegrid')
    palette = sns.color_palette('muted', n_clusters)

    if qq_plot:
        normal = np.sort(np.random.normal(size=data.shape[0]))
        df_qq = node_reduction(np.sort(data, axis=0), n_clusters=n_clusters, sample_labels=normal)

        fig = plt.figure(figsize=(8, 16))
        gs = fig.add_gridspec(nrows=int(n_clusters / 2 + 1), ncols=2)
        ax = [[] for i in range(n_clusters)]
        ax[0] = fig.add_subplot(gs[:1, :])
        sns.lineplot(x='sample_label', y='value', data=df_kde, hue='cluster', palette=palette, ax=ax[0])
        ax[0].set_xlabel('Magnitude')
        ax[0].set_ylabel('Density')
        ax[0].set_title(title)

        row = np.repeat(np.arange(1, int(n_clusters / 2 + 2)), 2)
        col = np.tile(np.array([[0, 1], [1, 2]]), 3)
        for i in range(n_clusters):
            ax[i] = fig.add_subplot(gs[row[i]:row[i + 2], col[0, i]:col[1, i]])
            for k in range(df_qq.shape[0]):
                if df_qq['cluster'][k] == i:
                    node = df_qq['node'][k]
                    break
            sns.scatterplot(x='sample_label', y='value', data=df_qq.where(df_qq['node'] == node),
                            ax=ax[i], label='from cluster ' + str(i), edgecolor=None, color=palette[i])
            ax[i].set_xlabel('Theoretical quantiles')
            ax[i].set_ylabel('Sample quantiles')
            ax[i].legend()
        plt.tight_layout()

    else:
        plt.figure(figsize=(6, 4))
        sns.lineplot(x='sample_label', y='value', data=df_kde, hue='cluster', palette=palette)
        plt.xlabel('Magnitude')
        plt.ylabel('Density')
        plt.title('PCA on density distributions')

    print('Status: Save plot.')
    plt.savefig('../doc/figures/preprocess_distribution_' + title + '.png')


def acf_plot(data: np.ndarray, n_clusters=5, n_lags=1000):
    """

    :param data:
    :param n_clusters:
    :param n_lags:
    :return:
    """
    acf, conf = acf(data, n_lags=n_lags)
    df = node_reduction(acf, n_clusters)
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    plt.xlim(0, n_lags)
    sns.lineplot(x='sample', y='value', data=df, hue='cluster', palette='colorblind')
    plt.plot([0, n_lags], [conf[1], conf[1]], color='black', ls='--')
    plt.plot([0, n_lags], [conf[0], conf[0]], color='black', ls='--')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.title('Autocorrelation')
    plt.savefig('../doc/figures/preprocess_acf.png')


def fft_plot(data: np.ndarray, n_clusters=5):
    """

    :param data:
    :param n_clusters:
    :return:
    """
    spect, freq = pre.fft(train_set.numpy(), fs=512, downscale_factor=20)
    df = pre.node_reduction(spect, n_clusters, sample_labels=freq)
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    # for i in range(60):
    #    plt.plot(freq, spect[:, i], color='tab:blue')
    sns.lineplot(x='sample_label', y='value', data=df, hue='cluster', palette='colorblind')

    plt.plot([120, 120], [0, 800], color='black', ls='--')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [-]')
    title = ['High frequency spectrum', 'Low frequency spectrum']
    save_name = ['fft_high', 'fft_low']
    xlim = [150, 25]
    for i in range(2):
        plt.xlim(0, xlim[i])
        plt.ylim(0, 800)
        plt.title(title[i])
        if i == 1:
            plt.plot([.5, .5], [0, 800], color='black', ls='--')
        plt.savefig('../doc/figures/preprocess_' + save_name[i])


def anim_connectome(patient_ID: str, hour: str, sperseg: float, soverlap: float, save=False, anim=True, fps=10):
    """

    """
    # Import signal
    ID = patient_ID
    h = hour
    data = hdf5storage.loadmat('../data/ID' + ID + '_' + h + '.mat')
    info = loadmat('../data/ID' + ID + '_info.mat')
    data = data['EEG']
    fs = float(info['fs'])

    nperseg = int(sperseg * fs)
    noverlap = int(soverlap * fs)

    corrmats = np.zeros((data.shape[0], data.shape[0], int((data.shape[1] - nperseg) / (nperseg - noverlap))))
    for i in range(corrmats.shape[2]):
        corrmats[:, :, i] = np.corrcoef(data[:, int(i * (nperseg - noverlap)):int(i * (nperseg - noverlap) + nperseg)])
    corr_val = np.mean(np.mean(np.abs(corrmats), axis=0), axis=0)

    if save:
        np.save('../data/corrmat_ID' + ID + '_' + h + '.npy', corrmats)
        np.save('../data/corrmean_ID' + ID + '_' + h + '.npy', corr_val)

    if anim:
        # equivalent to rcParams['animation.html'] = 'html5'
        rc('animation', html='html5')

        def update(frame):
            level.set_array(corrmats[:, :, frame].ravel())
            dot.set_data(t[frame], corr_val[frame])
            line.set_data([t[frame], t[frame]], [0, 30])
            return level

        corr = np.ones((corrmats.shape[0], corrmats.shape[0]))
        sns.set_style('whitegrid')

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2)

        ax0 = fig.add_subplot(gs[:1, :])
        f, t, Sxx = signal.spectrogram(data[0, :], fs, nperseg=nperseg, noverlap=noverlap)
        t = t[:-1] / 60
        ax0.pcolormesh(t, f, Sxx[:, :-1], vmin=0, vmax=20, cmap='viridis')
        ax0.set_title('ID' + ID + ' h' + h + ' Spectrogram of Channel 00')
        ax0.set_ylabel('Frequency [Hz]')
        ax0.set_ylim(0, 30)
        ax0.set_xlabel('Time [min]')
        line, = ax0.plot([], [], color='red')

        ax1 = fig.add_subplot(gs[1:, :1])
        level = ax1.pcolormesh(corr, cmap='bwr', vmin=-1, vmax=1)
        ax1.invert_yaxis()
        ax1.set_title('Correlation matrix')

        ax2 = fig.add_subplot(gs[1:, 1:])
        ax2.plot(t, corr_val)
        ax2.set_title('Mean abs. correlation')
        ax2.set_xlim(0, t[-1])
        ax2.set_ylim(0, np.max(corr_val) + 0.2)
        ax2.set_xlabel('Time [min]')
        dot, = ax2.plot([], [], 'o', color='red')

        ani = animation.FuncAnimation(fig, update, frames=corrmats.shape[2], interval=20)

        ani.save('../doc/figures/corr_ID' + ID + '_' + h + '.gif', writer='imagemagick', fps=fps)


def nonlin_corr_metric(x: np.ndarray, y: np.ndarray, t_shift: int):
    """
    :param x:
    :param y:
    :param t_shift: In samples
    :param save_name:
    :return:
    """
    break_locs = np.linspace(-5, 5, 50)
    h2_shift = []  # Array with correlation per time shift
    r2_shift = []  # Array with correlation per time shift

    for i in range(2 * t_shift):
        # Cut data to time shift
        y_data = y[2 * t_shift - i: len(y)-i]
        x_data = x[t_shift: -t_shift]
        # Normalize data
        y_norm = (y_data - np.mean(y_data)) / np.std(y_data)
        x_norm = (x_data - np.mean(x_data)) / np.std(x_data)
        # Non-linear fit
        g = pwlf.PiecewiseLinFit(x_norm, y_norm)
        g.fit_with_breaks(break_locs)
        expect = np.mean(np.abs(y_norm - g.predict(x_norm)))  # Uniform distribution
        h2_shift.append(1 - expect**2 / np.var(y_norm))
        # Linear fit
        r2_shift.append(np.corrcoef(x_norm, y_norm)[0, 1]**2)

    h2 = h2_shift[int(np.argmax(np.asarray(h2_shift)))]
    h2_dt = int(np.argmax(np.asarray(h2_shift)) - t_shift)  # dt for best correlation from x to y
    r2 = r2_shift[int(np.argmax(np.asarray(r2_shift)))]
    r2_dt = int(np.argmax(np.asarray(r2_shift)) - t_shift)  # dt for best correlation from x to y

    return r2, h2, r2_dt, h2_dt, r2_shift, h2_shift


def plot_nonlin_corr(r2, r2_dt, h2=None, h2_dt=None, save_name='default'):
    fs = 512

    if h2 is not None:
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2)
        cmap = 'viridis'

        ax0 = fig.add_subplot(gs[:1, :1])
        sns.heatmap(r2, cmap=cmap)
        ax0.set_title('r2 linear correlation')
        ax0.get_xaxis().set_visible(False)

        ax1 = fig.add_subplot(gs[:1, 1:])
        sns.heatmap(h2, cmap=cmap)
        ax1.set_title('h2 non-linear correlation')
        ax1.get_xaxis().set_visible(False), ax1.get_yaxis().set_visible(False)

        ax2 = fig.add_subplot(gs[1:, :1])
        vlim = np.max(np.abs(r2_dt))
        sns.heatmap(r2_dt / fs * 1000, cmap='bwr', vmin=-vlim, vmax=vlim)
        ax2.set_title('r2 time shift [ms]')

        ax3 = fig.add_subplot(gs[1:, 1:])
        vlim = np.max(np.abs(h2_dt))
        sns.heatmap(h2_dt / fs * 1000, cmap='bwr', vmin=-vlim, vmax=vlim)
        ax3.set_title('h2 time shift [ms]')
        ax3.get_yaxis().set_visible(False)

    else:
        fig = plt.figure(figsize=(8, 10))
        gs = fig.add_gridspec(2, 1)
        cmap = 'viridis'

        ax0 = fig.add_subplot(gs[:1, :1])
        sns.heatmap(r2, cmap=cmap)
        ax0.set_title('r2 linear correlation')
        ax0.get_xaxis().set_visible(False)

        ax2 = fig.add_subplot(gs[1:, :1])
        vlim = np.max(np.abs(r2_dt))
        sns.heatmap(r2_dt / fs * 1000, cmap='bwr', vmin=-vlim, vmax=vlim)
        ax2.set_title('r2 time shift [ms]')

    plt.tight_layout()
    plt.savefig('../doc/figures/fc_' + save_name + '.png')
    plt.close()


def nonlin_corr(patient_id: str, time_begin: list, duration: float,
                           t_shift=0.2, plot_name=None):
    """

    :param patient_id:
    :param time_begin: List with [hour, minute].
    :param duration: In seconds.
    :param t_shift: In seconds.
    :param plot_name:
    :return:
    """
    # Load and prepare data
    data_mat = loadmat('../data/' + patient_id + '_' + str(time_begin[0]) + 'h.mat')
    info_mat = loadmat('../data/' + patient_id + '_info.mat')
    fs = float(info_mat['fs'])
    sample_begin = int(time_begin[1] * 60 * fs)
    sample_end = sample_begin + int(duration * fs)
    data_raw = data_mat['EEG'][:, sample_begin:sample_end].transpose()
    if fs != 512:
        data_raw = signal.resample(data_raw, num=int(data_raw.shape[0] / fs * 512), axis=0)

    # Compute correlation
    h2 = np.zeros((data_raw.shape[1], data_raw.shape[1]))
    r2 = np.zeros((data_raw.shape[1], data_raw.shape[1]))
    h2_dt = np.zeros((data_raw.shape[1], data_raw.shape[1]))
    r2_dt = np.zeros((data_raw.shape[1], data_raw.shape[1]))
    h2_shift_list = []
    r2_shift_list = []

    start_time = time.time()

    for i in range(data_raw.shape[1]):
        for j in range(data_raw.shape[1]):
            r2[i, j], h2[i, j], r2_dt[i, j], h2_dt[i, j], r2_shift, h2_shift = nonlin_corr_metric(
                data_raw[:, i], data_raw[:, j], t_shift=int(t_shift * fs))
            h2_shift_list.append(h2_shift)
            r2_shift_list.append(r2_shift)
        t_rem = (time.time() - start_time) / (i + 1) * (data_raw.shape[1] - i + 1)
        clear_output()
        print(f'Computed columns: {i}/{data_raw.shape[1]} | Time remaining [min]: {t_rem / 60:.3}')

    save_name = '../data/fc_' + patient_id + '_' + str(time_begin[0]) + 'h' + str(time_begin[1]) + 'min'
    np.save(save_name + '_r2.npy', r2)
    np.save(save_name + '_r2_dt.npy', r2_dt)
    np.save(save_name + '_h2.npy', h2)
    np.save(save_name + '_h2_dt.npy', h2_dt)

    if plot_name is not None:
        plot_nonlin_corr(r2, h2, r2_dt, h2_dt, save_name=plot_name)


def lin_corr(patient_id: str, time_begin: list, duration: float, t_lag=0.7, critical_corr=0.7):
    """

    :param patient_id:
    :param time_begin: List with [hour, minute].
    :param duration: In seconds.
    :param t_lag: In seconds.
    :param critical_corr:
    :return:
    """
    # Load and prepare data
    data_mat = loadmat('../data/' + patient_id + '_' + str(time_begin[0]) + 'h.mat')
    info_mat = loadmat('../data/' + patient_id + '_info.mat')
    fs = float(info_mat['fs'])
    sample_begin = int(time_begin[1] * 60 * fs)
    sample_end = sample_begin + int(duration * fs)
    data_raw = data_mat['EEG'][:, sample_begin:sample_end].transpose()

    n_lag = int(t_lag * fs)
    factor = np.exp(-1)

    # Compute normalized cross correlation (NCC)
    cctl = np.zeros((data_raw.shape[1], data_raw.shape[1], (n_lag * 2) + 1))
    for from_ in range(data_raw.shape[1]):
        for to_ in range(data_raw.shape[1]):
            x = data_raw[:, to_]
            y = data_raw[n_lag:-n_lag, from_]
            cctl[from_, to_, :] = cross_correlation.correlate_template(x, y)

    # Calculate peak cross correlation (cc) and corresponding time lag (tl)
    sign = np.sign(np.max(cctl, axis=2) - np.abs(np.min(cctl, axis=2)))
    cc = np.multiply(np.max(np.abs(cctl), axis=2), sign)
    mask = np.where(np.abs(cc) > critical_corr, 1, np.nan)
    tl_n = np.argmax(np.abs(cctl), axis=2)
    tl = (tl_n - n_lag) * mask / fs * 1000  # in [ms]
    tl_no_mask = (tl_n - n_lag) / fs * 1000  # in [ms], used for plots

    # Calculate mean tau
    # Tile and stack values for future operations
    tl_n_stacked = np.dstack([tl_n] * cctl.shape[2])
    arg_tau_stacked = factor * np.dstack([cc] * cctl.shape[2])
    mask_stacked = np.dstack([np.where(np.abs(cc) > critical_corr, 1, 0)] * cctl.shape[2])
    t_indices_tiled = np.tile(np.arange(0, cctl.shape[2]), (cctl.shape[0], cctl.shape[0], 1))
    # Get indices of values close to factor of peak cross correlation
    close_indices = np.isclose(cctl, arg_tau_stacked, rtol=1e-1) * t_indices_tiled
    # Create mask to separate negative and positive tau
    higher_tau_mask = np.where(close_indices - tl_n_stacked > 0, 1, 0)
    lower_tau_mask = np.where((tl_n_stacked - close_indices > 0) & (close_indices != 0), 1, 0)
    # Eliminate possible third occurrence of np.isclose() to factor
    higher_edge_indices = np.where(np.diff(higher_tau_mask) == -1, 1, 0) * t_indices_tiled[:, :, :-1]
    higher_edge_indices = np.min(np.where(higher_edge_indices == 0, np.inf, higher_edge_indices), axis=2)
    higher_third_occ_mask = np.where(t_indices_tiled > np.dstack([higher_edge_indices] * cctl.shape[2]), 0, 1)
    lower_edge_indices = np.where(np.diff(lower_tau_mask) == 1, 1, 0) * t_indices_tiled[:, :, :-1]
    lower_edge_indices = np.max(lower_edge_indices, axis=2)
    lower_third_occ_mask = np.where(t_indices_tiled < np.dstack([lower_edge_indices] * cctl.shape[2]), 0, 1)
    # Apply masks (apply mask for critical correlation separately to get all taus for plots)
    higher_tau_masked_all = close_indices * higher_tau_mask * higher_third_occ_mask
    higher_tau_masked = higher_tau_masked_all * mask_stacked
    lower_tau_masked_all = close_indices * lower_tau_mask * lower_third_occ_mask
    lower_tau_masked = lower_tau_masked_all * mask_stacked
    # Compute median along time lag axis and ignore zero entries
    higher_tau = np.ma.median(np.ma.masked_where(higher_tau_masked == 0, higher_tau_masked), axis=2).filled(0)
    lower_tau = np.ma.median(np.ma.masked_where(lower_tau_masked == 0, lower_tau_masked), axis=2).filled(0)
    # Get taus without mask for critical correlation for plots
    higher_tau_all = np.ma.median(np.ma.masked_where(higher_tau_masked_all == 0, higher_tau_masked_all)
                                  , axis=2).filled(0)
    higher_tau_all = (higher_tau_all - n_lag) / fs * 1000  # in [ms]
    lower_tau_all = np.ma.median(np.ma.masked_where(lower_tau_masked_all == 0, lower_tau_masked_all)
                                 , axis=2).filled(0)
    lower_tau_all = (lower_tau_all - n_lag) / fs * 1000  # in [ms]
    # Calculate mean distance for tau to cc
    tau_n = (higher_tau - lower_tau) / 2
    tau = np.where(tau_n == 0, np.nan, tau_n) / fs * 1000  # in [ms]

    # Additional masks for plots (diagonal, upper triangle, ...)
    tl_masked = tl.copy()
    cc_masked = cc.copy()
    np.fill_diagonal(tl_masked, np.nan)
    np.fill_diagonal(cc_masked, np.nan)
    cc_masked[np.triu_indices(cc_masked.shape[0], k=1)] = np.nan

    # Plot cc, tl and tau
    # General settings
    sns.set_style('white')
    fig = plt.figure(figsize=(10, 13))
    gs = fig.add_gridspec(3, 2)
    cmap_div = copy.copy(mpl.cm.get_cmap('seismic'))
    cmap_div.set_bad('dimgrey')
    cmap_uni = copy.copy(mpl.cm.get_cmap('viridis'))
    cmap_uni.set_bad('dimgrey')

    # Subplot: Peak cross correlation
    ax0 = fig.add_subplot(gs[:1, :1])
    sns.heatmap(cc_masked, cmap=cmap_div, vmin=-1, vmax=1)
    ax0.set_title('Peak cross correlation')
    ax0.set_xlabel('Node idx'), ax0.set_ylabel('Node idx')

    # Subplot: Histogram of peak cross correlation
    ax1 = fig.add_subplot(gs[:1, 1:])
    sns.distplot(cc_masked, kde=False)
    ymin, ymax = ax1.get_ylim()
    xmin, xmax = ax1.get_xlim()
    label = 'Critical corr. = +/- ' + str(critical_corr)
    plt.plot([critical_corr, critical_corr], [ymin, ymax], linestyle='--', color='black', label=label)
    plt.plot([-critical_corr, -critical_corr], [ymin, ymax], linestyle='--', color='black')
    ax1.set_xlim(xmin, xmax), ax1.set_ylim(ymin, ymax)
    plt.legend()
    ax1.set_title('Peak cross correlation histogram')
    ax1.set_xlabel('Peak cross correlation [-]'), ax1.set_ylabel('Nr. of occurrence [-]')

    # Subplot: Time lag
    ax2 = fig.add_subplot(gs[1:2, :1])
    vlim = np.nanmax(np.abs(tl))
    sns.heatmap(tl_masked, cmap=cmap_div, vmin=-vlim, vmax=vlim)
    ax2.set_title('Corresponding time lag [ms]')
    ax2.set_xlabel('Node idx'), ax2.set_ylabel('Node idx')

    # Subplot: Histogram of time lag
    ax3 = fig.add_subplot(gs[1:2, 1:])
    sns.distplot(tl_masked, kde=False)
    ax3.set_title('Time lag histogram')
    ax3.set_xlabel('Time [ms]'), ax3.set_ylabel('Nr. of occurrence [-]')

    # Subplot: Tau
    ax4 = fig.add_subplot(gs[2:, :1])
    sns.heatmap(tau, cmap=cmap_uni)
    ax4.set_title('Corresponding tau [ms]')
    ax4.set_xlabel('Node idx'), ax4.set_ylabel('Node idx')

    # Subplot: Histogram of tau
    ax5 = fig.add_subplot(gs[2:, 1:])
    sns.distplot(np.diagonal(tau), kde=False, label='Auto correlated')
    auto_corr = tau.copy()
    auto_corr[np.diag_indices(auto_corr.shape[0])] = np.nan
    sns.distplot(auto_corr, kde=False, label='Cross correlated')
    ax5.set_title('Tau histogram'), plt.legend()
    ax5.set_xlabel('Time [ms]'), ax5.set_ylabel('Nr. of occurrence [-]')

    plt.tight_layout()
    save_name = patient_id + '_' + str(time_begin[0]) + 'h' + str(time_begin[1]) + 'm'
    plt.savefig('../doc/figures/cc_' + save_name + '.png')
    plt.close()

    # t vector for plots
    plt.figure(figsize=(8, 5))
    t = np.arange(0, cctl.shape[2])
    t = (t - n_lag) / fs * 1000
    n0 = 44  # Base node
    N = 7  # Number of line plots
    peaks_x, peaks_y, taus_x_0, taus_x_1, taus_y = [], [], [], [], []
    for i in range(N):
        n1 = n0 + i  # Reference node
        plt.plot(t, cctl[n0, n1, :], label='Nodes 0 - ' + str(i))
        peaks_x.append(tl_no_mask[n0, n1])
        peaks_y.append(cc[n0, n1])
        taus_x_0.append(higher_tau_all[n0, n1])
        taus_x_1.append(lower_tau_all[n0, n1])
        taus_y.append(cc[n0, n1] * factor)
    plt.scatter(peaks_x, peaks_y, color='black', marker='d', label='Peak', zorder=N + 1)
    plt.scatter(taus_x_0, taus_y, color='black', marker='<', label='Right tau', zorder=N + 1)
    plt.scatter(taus_x_1, taus_y, color='black', marker='>', label='Left tau', zorder=N + 1)
    ymin, ymax = plt.gca().get_ylim()
    plt.plot([-t_lag*1000, t_lag*1000], [critical_corr, critical_corr],
             color='black', linestyle=':', label='Critical corr.')
    plt.plot([-t_lag*1000, t_lag*1000], [-critical_corr, -critical_corr], color='black', linestyle=':')
    plt.ylim(ymin, ymax)
    plt.xlabel('Time lag [ms]'), plt.ylabel('NCC [-]')
    plt.title('Normalized cross correlation: examples'), plt.legend(loc='upper right')
    plt.xlim(-t_lag * 1000, t_lag * 1000), plt.grid()
    save_name = patient_id + '_' + str(time_begin[0]) + 'h' + str(time_begin[1]) + 'm'
    plt.savefig('../doc/figures/cctl_' + save_name + '.png')
    plt.close()


def determine_sample_size(patient_id: list=None, time_begin: list=None, max_sample_size: float=None, dt: float=None,
                          save_name='default', load_name=None):
    """

    :param patient_id: List with string or multiple strings.
    :param time_begin: List with list with [hour, minute] or multiple lists.
    :param max_sample_size: In seconds.
    :param dt: In seconds.
    :param save_name: String.
    :param load_name: If not None, only plot is applied.
    :return:
    """
    if load_name is None:
        corr_dts = []
        t_size = np.arange(dt, max_sample_size, dt).tolist()
        for i, id_ in enumerate(patient_id):
            print('Computes job ' + str(i) + '/' + str(len(patient_id) - 1))
            # Load and prepare data
            data_mat = loadmat('../data/' + id_ + '_' + str(time_begin[i][0]) + 'h.mat')
            info_mat = loadmat('../data/' + id_ + '_info.mat')
            fs = float(info_mat['fs'])
            sample_begin = int(time_begin[i][1] * 60 * fs)
            sample_end = sample_begin + int(max_sample_size * fs)
            data_raw = data_mat['EEG'][:, sample_begin:sample_end].transpose()

            # Get correlation matrices
            corr = np.zeros((len(t_size) + 1, data_raw.shape[1], data_raw.shape[1]))
            corr_dt = np.zeros((len(t_size), data_raw.shape[1], data_raw.shape[1]))
            sc = StandardScaler()
            for j, t in enumerate(t_size):
                data_norm = sc.fit_transform(data_raw[:int(t * fs), :])
                corr[j + 1, :, :] = np.corrcoef(data_norm.T)
                corr_dt[j, :, :] = corr[j + 1, :, :] - corr[j, :, :]

            corr_dts.append(np.sum(np.sum(np.abs(corr_dt), axis=1), axis=1))

        # Make DataFrame
        df = pd.DataFrame()
        for i, id_ in enumerate(patient_id):
            sub_df = pd.DataFrame()
            sub_df['unique_id'] = (np.ones(len(t_size)) * i).astype(int)
            sub_df['Patient ID'] = [id_ for _ in range(len(t_size))]
            sub_df['dt'] = [dt for _ in range(len(t_size))]
            time_begin_str = id_ + ': ' + str(time_begin[i][0]) + 'h' + str(time_begin[i][1]) + 'm'
            sub_df['time_begin'] = [time_begin_str for _ in range(len(t_size))]
            sub_df['corr_dt'] = corr_dts[i]
            sub_df['t_size'] = t_size
            df = df.append(sub_df, ignore_index=True)
        df.to_pickle('../data/sample_size_det_' + save_name + '.pkl')

    else:
        df = pd.read_pickle('../data/sample_size_det_' + load_name + '.pkl')

    # Plot results
    sns.set_style('white')
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='t_size', y='corr_dt', data=df, hue='Patient ID')
    plt.xlabel('Sample size [s]'), plt.ylabel('Sum of abs. weight changes [-]')
    plt.title('Weight change per ' + str(df['dt'][0]) + ' sec.')
    plt.ylim(0, 500), plt.xlim(df['t_size'].min(), df['t_size'].max())
    # plt.ylim(0, df.quantile(0.97)['corr_dt']), plt.xlim(df['t_size'].min(), df['t_size'].max())
    plt.savefig('../doc/figures/sample_size_det_' + save_name + '.png')
    plt.close()

