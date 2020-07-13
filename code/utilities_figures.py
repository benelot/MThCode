""" Various functions

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import pickle
from os import path

import models
import utilities_train as utrain


def plot_train_test(id_: str, pred_nodes: list, lim_nr_samples=None, train_set=False):
    plot_optimization(id_)
    plot_prediction(id_, pred_nodes, lim_nr_samples=lim_nr_samples, train_set=train_set)
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    if params['model_type'] != 'single':
        plot_weights(id_)


def plot_optimization(id_: str):
    """ Makes and saves evaluation plots of optimization to ../figures/.

        Saves:
            Figure "optim_[...]"
    """
    eval_optimization = pickle.load(open('../models/' + id_ + '/eval_optimization.pkl', 'rb'))

    epochs = eval_optimization['loss'].shape[0]
    nodes = eval_optimization['loss'].shape[1]
    m = np.zeros((4, epochs*nodes))
    m[0, :] = np.repeat(np.arange(0, epochs), nodes)
    m[1, :] = np.tile(np.arange(0, nodes), epochs)
    m[2, :] = eval_optimization['loss'].reshape(1, -1)
    m[3, :] = np.repeat(eval_optimization['grad_norm'], nodes)

    df = pd.DataFrame(m.T, columns=['Epoch', 'Node', 'Loss', 'Grad norm'])
    df[['Epoch', 'Node']] = df[['Epoch', 'Node']].astype('int32')
    df['Node'] = df['Node'].astype('str')

    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(nrows=2, ncols=2)
    ax = [[], [], []]
    ax[0] = fig.add_subplot(gs[:1, :1])
    ax[0] = sns.lineplot(x='Epoch', y='Loss', data=df)
    plt.xlim(0, epochs - 1)
    ax[1] = fig.add_subplot(gs[:1, 1:])
    ax[1] = sns.lineplot(x='Epoch', y='Grad norm', data=df)
    plt.xlim(0, epochs - 1)
    ax[2] = fig.add_subplot(gs[1:, :])
    ax[2] = sns.barplot(x='Node', y='Loss', data=df.where(df['Epoch'] == epochs - 1), color='tab:blue')
    plt.ylabel('Mean loss of last epoch')
    every_nth = 2
    for n, label in enumerate(ax[2].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.suptitle('Optimization of model ' + id_)
    fig.savefig('../doc/figures/optim_' + eval_optimization['id_'] + '.png')
    plt.close()


def plot_weights(id_: str, vmax=1, linewidth=0, absolute=False):
    """ Makes and saves a heat map of weight matrix W to ../figures/.

        Saves:
            Figure "weights_[...]"
    """
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    # Get trained model
    model = None
    if params['model_type'] == 'general':
        model = models.general_RNN(params)
    elif params['model_type'] == 'parallel':
        model = models.parallel_RNN(params)
    else:
        print('Error: No valid model type.')

    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth'))
    W = model.W.weight.data.numpy()

    vmin = -vmax
    cmap = 'bwr'
    ch = params['visible_size']

    if absolute:
        vmin = 0
        cmap = 'Blues'
        W = np.abs(W)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(nrows=W.shape[0], ncols=W.shape[0])
    cbar_ax = fig.add_axes([.92, .11, .02, .77])  # x-pos,y-pos,width,height

    if W.shape[0] == ch:
        ax0 = fig.add_subplot(gs[:ch, :ch])
        sns.heatmap(W[:ch, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, linewidths=linewidth, ax=ax0)
        ax0.set_ylabel('to visible nodes')
        ax0.set_xlabel('from visible nodes')
        ax0.set_title('Weight matrix of ' + params['id'])

    else:
        ax0 = fig.add_subplot(gs[:ch, :ch])
        ax0.get_xaxis().set_visible(False)
        ax1 = fig.add_subplot(gs[:ch, ch:])
        ax1.get_xaxis().set_visible(False), ax1.get_yaxis().set_visible(False)
        ax2 = fig.add_subplot(gs[ch:, :ch])

        ax3 = fig.add_subplot(gs[ch:, ch:])
        ax3.get_yaxis().set_visible(False)

        sns.heatmap(W[:ch, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax0)
        sns.heatmap(W[:ch, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax1)
        sns.heatmap(W[ch:, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax2)
        sns.heatmap(W[ch:, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, linewidths=linewidth, ax=ax3)

        pos_to_vis = 0.8 / W.shape[0] * params['hidden_size'] + 0.8 / W.shape[0] * (ch / 2) + 0.1
        pos_to_hid = 0.8 / W.shape[0] * (params['hidden_size'] / 2) + 0.1
        pos_from_vis = 0.8 / W.shape[0] * (ch / 2) + 0.1
        pos_from_hid = 0.8 / W.shape[0] * ch + 0.8 / W.shape[0] * (params['hidden_size'] / 2) + 0.1
        fig.text(0.08, pos_to_vis, 'to visible node', va='center', ha='center', rotation='vertical')
        fig.text(0.08, pos_to_hid, 'to hidden node', va='center', ha='center', rotation='vertical')
        fig.text(pos_from_vis, 0.06, 'from visible node', va='center', ha='center')
        fig.text(pos_from_hid, 0.06, 'from hidden node', va='center', ha='center')
        fig.subplots_adjust(hspace=0.8, wspace=0.8)

    plt.suptitle('Weight matrix of model ' + params['id_'])
    fig.savefig('../doc/figures/weights_' + id_ + '.png')
    plt.close()


def plot_prediction(id_: str, nodes_idx: list, lim_nr_samples=None, train_set=False, nodes2path=False):
    """ Makes and saves line plots of predictions to ../figures/.

        Saves:
            Figure "prediction_[...]"
    """
    # Get data
    eval_prediction = pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb'))
    if train_set:
        if lim_nr_samples is None:
            lim_nr_samples = eval_prediction['train_pred'].shape[0]
        pred = eval_prediction['train_pred'][-lim_nr_samples:, nodes_idx]
        true = eval_prediction['train_true'][-lim_nr_samples:, nodes_idx]
    else:
        if lim_nr_samples is None:
            lim_nr_samples = eval_prediction['test_pred'].shape[0]
        pred = eval_prediction['test_pred'][-lim_nr_samples:, nodes_idx]
        true = eval_prediction['test_true'][-lim_nr_samples:, nodes_idx]

    # Make plot
    sns.set_style('darkgrid')
    ax = [[] for i in range(len(nodes_idx))]
    fig = plt.figure(figsize=(10, int(len(ax)) * 3))
    for i in range(len(ax)):
        ax[i] = fig.add_subplot(int(len(nodes_idx)), 1, i + 1)
        ax[i] = plt.plot(pred[:, i], color='tab:red', ls='--', label='Predicted values')
        ax[i] = plt.plot(true[:, i], color='tab:blue', label='True values')
        plt.ylabel('Magn. [-]')
        plt.xlabel('Time steps [Samples]')
        plt.title('Node ' + str(nodes_idx[i]))
        plt.xlim(left=0)
        plt.legend(loc='upper right')
    plt.suptitle('Predictions of model ' + id_)
    fig.subplots_adjust(hspace=.7)

    nodes_str = ''
    if nodes2path:
        for _, node in enumerate(nodes_idx):
            nodes_str = nodes_str + '_' + str(node)
    plt.savefig('../doc/figures/prediction_' + id_ + nodes_str + '.png')
    plt.close()


def plot_multi_boxplots(ids: list, x: str, y: str, hue=None, ylim=None, save_name=None, split_id=False):
    """ Makes and saves box plots of results to ../figures/.

        Saves:
            Figure "boxplot_[...]"
    """
    df = pd.DataFrame()
    for idx, id_ in enumerate(ids):
        eval_distance = pickle.load(open('../models/' + id_ + '/eval_distances.pkl', 'rb'))
        df = df.append(pd.DataFrame(eval_distance), ignore_index=True)
        train_exists = path.exists('../models/' + id_ + '/eval_distances_train.pkl')
        if train_exists:
            eval_distance = pickle.load(open('../models/' + id_ + '/eval_distances_train.pkl', 'rb'))
            df = df.append(pd.DataFrame(eval_distance), ignore_index=True)

    if split_id:
        id1, id2 = [], []
        for i, val in enumerate(df['id']):
            id1.append(df['id'][i][:-7])
            id2.append(df['id'][i][-6:])
        df['architecture'] = id1
        df['time'] = id2

    plt.figure(figsize=(10, 8))
    sns.set_style('darkgrid')
    ax = sns.boxplot(x=x, y=y, data=df, hue=hue)
    ax.set(xlabel=x, ylabel=y)
    if ylim:
        plt.ylim(ylim)
    if save_name is None:
        save_name = y + '_of_' + x
    plt.title(y + ' of ' + x)
    plt.savefig('../doc/figures/boxplots_' + save_name + '.png')
    plt.close()


def plot_multi_scatter(ids: list, save_name='default'):
    """ Makes and saves scatter plots of predictions to ../figures/.

        Saves:
            Figure "scatter_[...]"
    """
    eval_predictions = []
    for idx, id_ in enumerate(ids):
        eval_predictions.append(pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb')))

    sns.set_style('darkgrid')
    ax = [[] for i in range(len(ids))]

    fig = plt.figure(figsize=(10, int(np.ceil(len(ax) / 2)) * 5))
    for i in range(len(ax)):
        pred = eval_predictions[i]['test_pred']
        true = eval_predictions[i]['test_true']
        ax[i] = fig.add_subplot(int(np.ceil(len(ax)/2)), 2, i + 1)
        ax[i] = plt.scatter(pred, true, s=.01)
        ax[i] = plt.plot([-1, 1], [-1, 1], ls="--", color='red')
        plt.axis([-1, 1, -1, 1])
        plt.ylabel('Predicted value')
        plt.xlabel('True value')
        plt.title(ids[i])
    plt.tight_layout()
    plt.savefig('../doc/figures/scatter_' + save_name + '.png')
    plt.close()


def plot_corr_map(id_: str, size_of_samples=2000, save_name='default'):
    """ Makes and saves heat map of electrode correlation in train set to ../figures/.

        Saves:
            Figure "corr_[...]"
    """
    train_set, test_set = utrain.data_loader(id_, windowing=False)

    df = pd.DataFrame(train_set[:size_of_samples, :].numpy())
    corr = df.corr()

    fig, ax = plt.subplots()
    plt.title('Node correlation over 2000 samples')
    sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, square=True)
    fig.savefig('../doc/figures/corr_' + save_name + '.png')
    plt.close()


def mean_weights(ids: list, hidden=True, save_name='default'):
    """

    """
    id1, id2 = [], []
    mean_, mean_abs = [], []
    for i, id_ in enumerate(ids):
        params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
        # Get trained model
        model = None
        if params['model_type'] == 'general':
            model = models.general_RNN(params)
        elif params['model_type'] == 'parallel':
            model = models.parallel_RNN(params)
        else:
            print('Error: No valid model type.')

        model.load_state_dict(torch.load('../models/' + id_ + '/model.pth'))
        W = model.W.weight.data.numpy()
        mean_.append(np.mean(W))
        mean_abs.append(np.mean(np.abs(W)))
        id1.append(id_[:-7])
        id2.append(id_[-6:])

        if hidden is False:
            ch = params['channel_size']
            if params['reverse_nodes'] is True:
                ch = ch * 2
            mean_[i] = np.mean(W[:ch, :ch])
            mean_abs[i] = np.mean(np.abs(W[:ch, :ch]))

    df = pd.DataFrame()
    df['Architecture'] = id1
    df['Time'] = id2
    df['Mean abs. weight'] = mean_abs

    plt.figure(figsize=(10, 8))
    sns.set_style('darkgrid')
    ax = sns.barplot(x='Mean abs. weight', y='Architecture', hue='Time', data=df)
    ax.set(xlabel='Mean abs. weight', ylabel='Architecture')
    ax.set_title('Mean abs. weight')
    plt.savefig('../doc/figures/barplots_meanabs_' + save_name + '.png')
    plt.close()
