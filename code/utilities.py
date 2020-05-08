""" Various functions

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import time
import torch
import torch.nn as nn
import pickle

import models


def train(params: dict):
    """ Train and saves model with parameters params.

    """
    # Load data
    X_train, X_test = data_loader(params)

    # Define model, criterion and optimizer
    model = models.FRNN(params)
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    temp_loss = np.zeros([len(X_train), params['channel_size']])
    epoch_loss = np.zeros([params['epochs'], params['channel_size']])

    start_time = time.time()
    for epoch in range(params['epochs']):
        if epoch is not 0 and epoch % params['lr_decay'] is 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        for T, X in enumerate(X_train):
            optimizer.zero_grad()
            prediction = model(X)
            loss = criterion(prediction, X[-1, :])
            temp_loss[T, :] = loss.detach()
            torch.autograd.backward(loss.mean())
            optimizer.step()
        epoch_loss[epoch, :] = np.mean(temp_loss, axis=0)
        print(f'Epoch: {epoch} | Loss: {np.mean(temp_loss):.4}')

    total_time = time.time() - start_time
    print(f'Time [min]: {total_time / 60:.3}')

    # Plot loss
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.plot(np.max(epoch_loss, axis=1), c='tab:blue', ls='--', label='max/min')
    ax.plot(np.min(epoch_loss, axis=1), c='tab:blue', ls='--')
    ax.plot(np.quantile(epoch_loss, 0.25, axis=1), c='tab:purple', label='quartile')
    ax.plot(np.quantile(epoch_loss, 0.75, axis=1), c='tab:purple')
    ax.plot(np.median(epoch_loss, axis=1), c='tab:red', ls='--', lw=3, label='median')
    ax.set_title(f'Computation time [min]: {total_time / 60:.3}')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid()
    fig.savefig('../doc/figures/losses_' + params['name'] + '.png')

    # Save model and params to file
    torch.save(model.state_dict(), '../models/' + params['name'] + '.pth')
    pickle.dump(params, open('../models/' + params['name'] + '.pkl', 'wb'))


def evaluation(model_name: str, ch_show: list, plot_signal=True, plot_W=True,
               plot_scatter=True, plot_box=True, eval_train=False):
    """ Evaluates model an returns evaluation values.

    """
    # Load parameters
    params = pickle.load(open('../models/' + model_name + '.pkl', 'rb'))

    # Load data
    X_train, X_test = data_loader(params)

    # Get trained model
    model = models.FRNN(params)
    model.load_state_dict(torch.load('../models/' + params['name'] + '.pth'))

    # Evaluate model
    model.eval()

    with torch.no_grad():
        test_preds = np.zeros((len(X_test), params['channel_size']))
        test_true = np.zeros((len(X_test), params['channel_size']))
        test_corr = []
        for T, X in enumerate(X_test):
            predictions = model(X)
            test_preds[T, :] = predictions.numpy()
            test_true[T, :] = X[-1, :].numpy()
        for i in range(params['channel_size']):
            test_corr.append(np.corrcoef(test_preds[:, i], test_true[:, i])[0, 1])
        print(f'Mean test correlation: {np.mean(test_corr)}')

        if eval_train:
            train_preds = np.zeros((len(X_train), params['channel_size']))
            train_true = np.zeros((len(X_train), params['channel_size']))
            train_corr = []
            with torch.no_grad():
                for T, X in enumerate(X_train):
                    predictions = model(X)
                    train_preds[T, :] = predictions.numpy()
                    train_true[T, :] = X[-1, :].numpy()
                for i in range(params['channel_size']):
                    train_corr.append(np.corrcoef(train_preds[:, i], train_true[:, i])[0, 1])
                print(f'Mean train correlation: {np.mean(train_corr)}')

    if plot_signal is True:
        fig, ax = plt.subplots(len(ch_show), figsize=(8, len(ch_show) * 2.5))
        fig.tight_layout(pad=3)
        for i, ch_out in enumerate(ch_show):
            ax[i].set_title(f'Channel: {ch_out}')
            ax[i].plot(test_true[:, ch_out], label='true')
            ax[i].plot(test_preds[:, ch_out], label='predicted')
            ax[i].set_xlabel('Samples [-]')
            ax[i].set_ylabel('Magn. [-]')
            ax[i].legend()
        fig.subplots_adjust(hspace=.8)
        fig.savefig('../doc/figures/pred_' + params['name'] + '.png')

    if plot_scatter is True:
        idx = []
        for i in range(int(len(ch_show) / 2)):
            for k in range(int(len(ch_show) / 2)):
                idx.append([i, k])
        fig, ax = plt.subplots(int(len(ch_show) / 2), int(len(ch_show) / 2), figsize=(10, 10))
        fig.tight_layout(pad=5)
        for n, ch_out in enumerate(ch_show):
            lim_max = np.max(test_true[:, ch_out])
            lim_min = np.min(test_true[:, ch_out])
            ax[idx[n][0]][idx[n][1]].set_title(f'Channel: {ch_out}')
            ax[idx[n][0]][idx[n][1]].scatter(test_true[:, ch_out], test_preds[:, ch_out], s=1)
            ax[idx[n][0]][idx[n][1]].plot([lim_min, lim_max], [lim_min, lim_max], ls="--", color='red')
            ax[idx[n][0]][idx[n][1]].set_xlabel('True iEEG signal [-]')
            ax[idx[n][0]][idx[n][1]].set_ylabel('Predicted iEEG signal [-]')
        fig.savefig('../doc/figures/scatter_' + params['name'] + '.png')

    if plot_box is True:
        plt.figure(figsize=(5, 10))
        plt.title('Correlation per channel')
        plt.grid()
        plt.ylim([0.5, 1])
        plt.boxplot(test_corr)
        plt.savefig('../doc/figures/box_' + params['name'] + '.png')

    if plot_W is True:
        W = model.W.weight.data.numpy()
        plot_weights(W=W, params=params, vmax=.5, linewidth=.0, save2path='../doc/figures/weights_' + params['name'] + '.png')

    if eval_train:
        return test_corr, train_corr
    else:
        return test_corr


def data_loader(params: dict, train_portion=0.8, windowing=True):
    """ Loads and prepares iEEG data for NN model.

    """
    data_mat = loadmat(params['path2data'])
    data = data_mat['EEG'][:params['channel_size'], :params['sample_size']].transpose()

    # Normalization
    if params['normalization']:
        sc = MinMaxScaler(feature_range=(-1, 1))
        sc.fit(data)
        data = sc.transform(data)

    # To tensor
    data = torch.FloatTensor(data)

    # Split data into training and test set
    train_set = data[:int(train_portion * params['sample_size']), :]
    test_set = data[int(train_portion * params['sample_size']):, :]

    if windowing is False:
        return train_set, test_set

    # Windowing
    X_train, X_test = [], []
    for i in range(train_set.shape[0] - params['window_size']):
        X_train.append(train_set[i:i + params['window_size'], :])
    for i in range(test_set.shape[0] - params['window_size']):
        X_test.append(test_set[i:i + params['window_size'], :])

    return X_train, X_test


def plot_weights(W: float, params: dict, vmax=1, linewidth=.5, absolute=False, save2path=None):
    """ Plots a heat map of weight matrix W.

    """
    vmin = -vmax
    cmap = 'RdBu'
    ch = params['channel_size']
    hticklabels = np.arange(ch, W.shape[0], 1)

    if absolute:
        vmin = 0
        cmap = 'Blues'
        W = np.abs(W)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(nrows=W.shape[0], ncols=W.shape[0])
    cbar_ax = fig.add_axes([.925, .125, .02, .755])  # l,b,w,h

    ax0 = fig.add_subplot(gs[:ch, :ch])
    ax0.set_ylabel('to visible nodes')
    ax0.get_xaxis().set_visible(False)

    ax1 = fig.add_subplot(gs[:ch, ch:])
    ax1.get_xaxis().set_visible(False), ax1.get_yaxis().set_visible(False)
    ax1.set_ylabel('to hidden nodes')
    ax1.set_xlabel('from visible nodes')

    ax2 = fig.add_subplot(gs[ch:, :ch])
    ax2.set_ylabel('to hidden nodes')
    ax2.set_xlabel('from visible nodes')

    ax3 = fig.add_subplot(gs[ch:, ch:])
    ax3.get_yaxis().set_visible(False)
    ax3.set_xlabel('from hidden nodes')

    sns.heatmap(W[:ch, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth,
                ax=ax0)
    sns.heatmap(W[:ch, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth,
                ax=ax1)
    sns.heatmap(W[ch:, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth,
                ax=ax2, yticklabels=hticklabels)
    sns.heatmap(W[ch:, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, linewidths=linewidth,
                ax=ax3, xticklabels=hticklabels)

    fig.text(0.08, 0.65, 'to visible node', va='center', ha='center', rotation='vertical')
    fig.text(0.08, 0.27, 'to hidden node', va='center', ha='center', rotation='vertical')
    fig.text(0.35, 0.08, 'from visible node', va='center', ha='center')
    fig.text(0.77, 0.08, 'from hidden node', va='center', ha='center')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    if save2path is not None:
        fig.savefig(save2path)

