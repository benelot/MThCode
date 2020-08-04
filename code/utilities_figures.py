""" Various functions

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import pickle
from os import path

import models
import utilities_train as utrain


def plot_train_test(id_: str, pred_nodes: list, lim_nr_samples=None):
    plot_optimization(id_)
    plot_prediction(id_, pred_nodes, lim_nr_samples=lim_nr_samples)
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

    sns.set_style('whitegrid')
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
    model = models.GeneralRNN(params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth', map_location=device))
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
        ax0.set_title('Weight matrix of ' + params['id_'])

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


def plot_prediction(id_: str, nodes_idx: list, lim_nr_samples=None, nodes2path=False):
    """ Makes and saves line plots of predictions to ../figures/.

        Saves:
            Figure "prediction_[...]"
    """
    # Get data
    eval_prediction = pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb'))
    if lim_nr_samples is None:
        lim_nr_samples = eval_prediction['true'].shape[0]
    pred = eval_prediction['prediction'][-lim_nr_samples:, nodes_idx]
    true = eval_prediction['true'][-lim_nr_samples:, nodes_idx]

    # Make plot
    sns.set_style('whitegrid')
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

    plt.figure(figsize=(10, 8))
    sns.set_style('whitegrid')
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

    sns.set_style('whitegrid')
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


def mean_weights(ids: list, hidden=True, diagonal=True, save_name='default'):
    """

    """
    id1, id2, id3 = [], [], []
    mean_abs = []
    patient_id = []
    brain_state = []
    batch_size = []

    for i, id_ in enumerate(ids):
        params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
        # Get trained model
        model = models.GeneralRNN(params)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('../models/' + id_ + '/model.pth', map_location=device))
        W = model.W.weight.data.numpy()

        if id_[5:7] == 'ID11a':  # [7:12]
            patient_id.append('ID11a')
        elif id_[5:7] == 'ID11b':
            patient_id.append('ID11b')
        else:
            patient_id.append(params['patient_id'])
        brain_state.append(params['brain_state'])
        batch_size.append(params['batch_size'])

        W_abs = np.abs(W)
        if hidden is False:
            ch = params['visible_size']
            W_abs = np.abs(W[:ch, :ch])
        if diagonal is False:
            np.fill_diagonal(W_abs, 0)
        mean_abs.append(np.mean(W_abs))

    df = pd.DataFrame()
    df['Patient ID'] = patient_id
    df['Pos. in sleep cylce'] = brain_state
    df['Mean abs. weight'] = mean_abs
    df['Batch size'] = batch_size

    with sns.color_palette('colorblind', 3):
        plt.figure(figsize=(10, 8))
        sns.set_style('whitegrid')
        ax = sns.barplot(x='Mean abs. weight', y='Batch size', hue='Pos. in sleep cylce', data=df, orient='h')
        ax.set(xlabel='Mean abs. weight', ylabel='Batch size')
        ax.set_title('Mean abs. weight')
        ax.set_xlim(left=0.04)
    plt.savefig('../doc/figures/barplots_meanabs_' + save_name + '.png')
    plt.close()


def plot_weighted_prediction(id_, node_idx, max_duration=.5):
    # Parameter
    fs = 512

    # Get model
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    model = models.GeneralRNN(params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth', map_location=device))
    W = model.W.weight.data.numpy()
    w = W[node_idx, :]
    b = model.W.bias.data.numpy()
    b = b[node_idx]

    # Get prediction
    eval_prediction = pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb'))
    if max_duration * fs >= eval_prediction['prediction'].shape[0]:
        max_duration = int((eval_prediction['prediction'].shape[0] - 1) / fs)
    prediction = eval_prediction['prediction'][-int(max_duration * fs):, node_idx]
    true = eval_prediction['true'][-int(max_duration * fs):, node_idx]

    # Get data
    data = utrain.pre_process(params=params).numpy()
    data = data[-int(max_duration * fs):, :]
    data[:, node_idx] = prediction

    sns.set_style('white')
    fig = plt.figure(figsize=(12, 4.3))
    gs = fig.add_gridspec(1, 5)

    ax0 = fig.add_subplot(gs[:, :2])
    sns.heatmap(W, cmap='seismic', vmin=-1, vmax=1)
    ax0.add_patch(mpl.patches.Rectangle((0, node_idx), data.shape[1], 1, fill=False, edgecolor='black', lw=3))
    ax0.set_xlabel('From node'), ax0.set_ylabel('To node'), ax0.set_title('Weight matrix')

    ax1 = fig.add_subplot(gs[:, 2:])
    t = np.arange(0, data.shape[0] / fs, 1 / fs)
    cmap = mpl.cm.get_cmap('seismic')
    for i in range(data.shape[1] - 1):
        color = cmap(w[i] / 2 + 0.5)
        alpha = np.abs(w[i]) / np.max(np.abs(w))
        plt.plot(t, data[:, i], color=color, alpha=alpha)
    plt.plot(t, prediction, color='black', linestyle=':', label='predicted')
    plt.plot(t, true, color='black', label='true')
    ax1.set_xlabel('Time [s]'), ax1.set_ylabel('Membrane potential u(t) [a.U.]')
    ax1.set_xlim(0, t[-1]), ax1.set_title('Contribution to prediction of node ' + str(node_idx)), plt.legend()
    plt.tight_layout()
    plt.savefig('../doc/figures/contribution_' + id_ + '_node_' + str(node_idx) + '.png')
    plt.close()

