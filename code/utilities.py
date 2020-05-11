""" Various functions

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
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


def predict(model_name: str, train_set=False):
    """ Tests model an returns predictions and correlations.

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
        for T, X in enumerate(X_test):
            predictions = model(X)
            test_preds[T, :] = predictions.numpy()
            test_true[T, :] = X[-1, :].numpy()

        pd.DataFrame(test_preds).to_csv('../results/' + params['name'] + '_preds.csv')
        pd.DataFrame(test_true).to_csv('../results/' + params['name'] + '_true.csv')
        if train_set is False:
            return test_preds, test_true
        else:
            train_preds = np.zeros((len(X_train), params['channel_size']))
            train_true = np.zeros((len(X_train), params['channel_size']))
            with torch.no_grad():
                for T, X in enumerate(X_train):
                    predictions = model(X)
                    train_preds[T, :] = predictions.numpy()
                    train_true[T, :] = X[-1, :].numpy()

    pd.DataFrame(train_preds).to_csv('../results/' + params['name'] + '_preds-tr.csv')
    pd.DataFrame(train_true).to_csv('../results/' + params['name'] + '_true-tr.csv')
    return test_preds, test_true, train_preds, train_true


def evaluate(model_name: str, preds: np.ndarray, true: np.ndarray):
    """ Computes Correlation, MSE, MAE for evaluation.

    """
    # Load parameters
    params = pickle.load(open('../models/' + model_name + '.pkl', 'rb'))

    corr = []
    # Calculate distances
    for i in range(params['channel_size']):
        corr.append(np.corrcoef(preds[:, i], true[:, i])[0, 1])
    mse = np.mean((preds - true) ** 2, axis=0)
    mae = np.mean(np.abs(preds - true), axis=0)

    results = {'Name': [params['name'] for i in range(params['channel_size'])],
               'Node': [i for i in range(params['channel_size'])],
               'Non-Linearity': [params['nonlinearity'] for i in range(params['channel_size'])],
               'Bias': [params['bias'] for i in range(params['channel_size'])],
               'Recurrence': [params['lambda'] for i in range(params['channel_size'])],
               'Correlation': corr,
               'MSE': mse,
               'MAE': mae}

    pickle.dump(results, open('../results/' + params['name'] + '_results.pkl', 'wb'))

    return results


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


def plot_weights(params: dict, vmax=1, linewidth=.5, absolute=False):
    """ Plots a heat map of weight matrix W.

    """
    # Get trained model
    model = models.FRNN(params)
    model.load_state_dict(torch.load('../models/' + params['name'] + '.pth'))
    W = model.W.weight.data.numpy()

    vmin = -vmax
    cmap = 'RdBu'
    ch = params['channel_size']
    #hticklabels = np.arange(ch, W.shape[0], 1)

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

    sns.heatmap(W[:ch, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax0)
    sns.heatmap(W[:ch, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax1)
    sns.heatmap(W[ch:, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax2)
    sns.heatmap(W[ch:, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, linewidths=linewidth, ax=ax3)

    fig.text(0.08, 0.65, 'to visible node', va='center', ha='center', rotation='vertical')
    fig.text(0.08, 0.27, 'to hidden node', va='center', ha='center', rotation='vertical')
    fig.text(0.35, 0.08, 'from visible node', va='center', ha='center')
    fig.text(0.77, 0.08, 'from hidden node', va='center', ha='center')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.savefig('../doc/figures/weights_' + params['name'] + '.png')


def box_plots(results: pd.DataFrame, x: str, y: str, hue=None, ylim=None):
    """ Makes and saves box plots of results.

    """
    plt.figure(figsize=(10, 8))
    sns.set_style('darkgrid')
    ax = sns.boxplot(x=x, y=y, data=results, hue=hue)
    ax.set(xlabel='', ylabel=y)
    sns.set_style('darkgrid')
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.savefig('../doc/figures/boxplot_' + x + '.png')


def scatter_plots(titles: list, preds: list, trues: list, save_name='default'):
    """ Makes and saves scatter plots of predictions.

    """
    sns.set_style('darkgrid')
    ax = [[] for i in range(len(titles))]

    fig = plt.figure(figsize=(10, int(len(ax) / 2) * 5))
    for i in range(len(ax)):
        ax[i] = fig.add_subplot(int(len(ax)/2), 2, i + 1)
        ax[i] = plt.scatter(preds[i], trues[i], s=.01)
        ax[i] = plt.plot([-1, 1], [-1, 1], ls="--", color='red')
        plt.axis([-1, 1, -1, 1])
        plt.ylabel('Predicted value')
        plt.xlabel('True value')
        plt.title(titles[i])
    plt.tight_layout()
    plt.savefig('../doc/figures/scatter_' + save_name + '.png')


def prediction_plots(titles: list, pred: np.ndarray, true: np.ndarray, save_name='default'):
    """ Makes and saves line plots of predictions.

    """
    sns.set_style('darkgrid')
    ax = [[] for i in range(len(titles))]
    fig = plt.figure(figsize=(10, int(len(ax)) * 3))
    for i in range(len(ax)):
        ax[i] = fig.add_subplot(int(len(titles)), 1, i + 1)
        ax[i] = plt.plot(pred[:, i])
        ax[i] = plt.plot(true[:, i], ls="--", color='tab:red')
        plt.ylabel('Magn. [-]')
        plt.xlabel('Time steps [Samples]')
        plt.title(titles[i])
    plt.tight_layout()
    plt.savefig('../doc/figures/prediction_' + save_name + '.png')


def corr_map(params: dict, save_name='default'):
    """ Makes and saves heat map of electrode correlation in train set.

    """
    params = pickle.load(open('../models/' + params['name'] + '.pkl', 'rb'))
    train_set, test_set = data_loader(params, windowing=False)

    df = pd.DataFrame(train_set[:2000, :].numpy())
    corr = df.corr()

    fig, ax = plt.subplots()
    plt.title('Node correlation over 2000 samples')
    sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, square=True)
    fig.savefig('../doc/figures/corr_' + save_name + '.png')

