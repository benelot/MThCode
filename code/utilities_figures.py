""" Various functions

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import pickle
from os import path
import random

import models
import utilities_train as utrain


def plot_train_test(id_: str, n_nodes=None, node_idx=None):
    plot_optimization(id_)
    plot_prediction(id_, n_nodes=n_nodes, node_idx=node_idx)
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    if params['model_type'] != 'single':
        plot_weights(id_, plot_cbar=False)


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

    sns.set_style('ticks')
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(nrows=12, ncols=11)
    ax = [[], [], []]

    ax[0] = fig.add_subplot(gs[:5, :5])
    ax[0] = sns.lineplot(x='Epoch', y='Loss', data=df, color='black')
    ax[0].set_ylabel('Loss (MAE) [-]'), ax[0].set_xlabel('Epoch [Nr.]')
    plt.xlim(0, epochs - 1)

    ax[1] = fig.add_subplot(gs[:5, 7:])
    ax[1] = sns.lineplot(x='Epoch', y='Grad norm', data=df, color='black')
    ax[1].set_ylabel('Gradient norm [-]'), ax[1].set_xlabel('Epoch [Nr.]')
    plt.xlim(0, epochs - 1)

    ax[2] = fig.add_subplot(gs[6:, :])
    ax[2] = sns.barplot(x='Node', y='Loss', data=df.where(df['Epoch'] == epochs - 1), color='gray', edgecolor='black')
    ax[2].set_ylabel('Mean loss of\nEpoch 250'), ax[2].set_xlabel('Node index [-]')

    every_nth = 5
    ax[2].xaxis.set_ticks(np.arange(0, eval_optimization['loss'].shape[1], every_nth))
    # for n, label in enumerate(ax[2].xaxis.get_ticklabels()):
    #     if n % every_nth != 0:
    #         label.set_visible(False)
    for i in range(len(ax)):
        ax[i].spines['right'].set_visible(False), ax[i].spines['top'].set_visible(False)
    fig.savefig('../doc/figures/optim_' + eval_optimization['id_'] + '.png', dpi=300)
    plt.close()


def plot_weights(id_: str, vmax=1, linewidth=0, absolute=False, plot_cbar=True):
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
    cmap = 'seismic'
    ch = params['visible_size']
    #mpl.rcParams.update({'font.size': 15})

    # Temporary
    #sns.heatmap(W, vmin=vmin, vmax=vmax, annot=True, cmap='seismic')
    #plt.xlabel('From node [Nr.]'), plt.ylabel('To node [Nr.]'), plt.title('Strong coupling (weight decay)')
    #plt.savefig('W_strong_coupling_wd.png')

    if absolute:
        vmin = 0
        cmap = 'Blues'
        W = np.abs(W)

    sns.set_style('ticks')

    if W.shape[0] == ch:
        if params['artificial_signal'][0] is False:
            fig = plt.figure(figsize=(3, 3))
            gs = fig.add_gridspec(nrows=W.shape[0] + 10, ncols=W.shape[0] + 10)
            cbar_ax = fig.add_axes([.92, .11, .02, .77])  # x-pos,y-pos,width,height
            ax0 = fig.add_subplot(gs[5:ch+5, 5:ch+5])
            sns.heatmap(W[:ch, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, linewidths=linewidth, ax=ax0,
                        linecolor='grey', yticklabels=10, xticklabels=10)
            ax0.set_ylabel('To node index [-]')
            ax0.set_xlabel('From node index [-]')

            for _, spine in ax0.spines.items():
                spine.set_visible(True)
            cbar = ax0.collections[0].colorbar
            cbar.set_ticks([-1, 0, 1])
            cbar.ax.tick_params(size=0)

        else:
            print('Mean weight of ' + params['id_'] + ': ' + str(np.mean(W)))
            fig = plt.figure(figsize=(3, 3))
            sns.set_style('white')
            sns.heatmap(W[:ch, :ch], vmin=-1, vmax=1, cmap='seismic', annot=True, fmt='.2f', cbar=False)
            plt.ylabel('Weight index [-]'), plt.xlabel('Weight index [-]')
            plt.tight_layout()
            ax3 = plt.gca()
            for _, spine in ax3.spines.items():
                spine.set_visible(True)
            if params['artificial_signal'][1] is False:
                plt.savefig('figures/fig_Ch2_simul_W_strong.png', dpi=300)
            else:
                plt.savefig('figures/fig_Ch2_simul_W_weak.png', dpi=300)

    else:
        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(nrows=W.shape[0], ncols=W.shape[0])
        cbar_ax = fig.add_axes([.92, .11, .02, .77])  # x-pos,y-pos,width,height
        ax0 = fig.add_subplot(gs[:ch, :ch])
        ax0.get_xaxis().set_visible(False)
        ax1 = fig.add_subplot(gs[:ch, ch:])
        ax1.get_xaxis().set_visible(False), ax1.get_yaxis().set_visible(False)
        ax2 = fig.add_subplot(gs[ch:, :ch])

        ax3 = fig.add_subplot(gs[ch:, ch:])
        ax3.get_yaxis().set_visible(False)

        ticklabels = 7
        sns.heatmap(W[:ch, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax0,
                    yticklabels=ticklabels)
        sns.heatmap(W[:ch, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax1)
        sns.heatmap(W[ch:, :ch], cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, linewidths=linewidth, ax=ax2,
                    yticklabels=ticklabels, xticklabels=ticklabels)
        sns.heatmap(W[ch:, ch:], cmap=cmap, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, linewidths=linewidth, ax=ax3,
                    xticklabels=ticklabels)

        pos_to_vis = 0.8 / W.shape[0] * params['hidden_size'] + 0.8 / W.shape[0] * (ch / 2) + 0.08
        pos_to_hid = 0.8 / W.shape[0] * (params['hidden_size'] / 2) + 0.1
        pos_from_vis = 0.8 / W.shape[0] * (ch / 2) + 0.12
        pos_from_hid = 0.8 / W.shape[0] * ch + 0.8 / W.shape[0] * (params['hidden_size'] / 2) + 0.12
        fig.text(0.04, pos_to_vis, 'To visible node index [-]', va='center', ha='center', rotation='vertical')
        fig.text(0.04, pos_to_hid, 'To hidden node index [-]', va='center', ha='center', rotation='vertical')
        fig.text(pos_from_vis, 0.03, 'From visible node index [-]', va='center', ha='center')
        fig.text(pos_from_hid, 0.03, 'From hidden node index [-]', va='center', ha='center')
        fig.subplots_adjust(hspace=3, wspace=3)  # 0.8
        for _, spine in ax0.spines.items():
            spine.set_visible(True)
        for _, spine in ax1.spines.items():
            spine.set_visible(True)
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
        for _, spine in ax3.spines.items():
            spine.set_visible(True)
        ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

    fig.savefig('../doc/figures/weights_' + id_ + '.png', dpi=300)
    plt.close()

    if plot_cbar:
        fig, ax4 = plt.subplots(figsize=(0.5, 1))
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cbar = ax4.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='seismic'), ax=ax4, pad=.05, fraction=1)
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.tick_params(size=0)
        ax4.axis('off')
        fig.savefig('../doc/figures/cbar.png', dpi=300)
        plt.close()


def plot_prediction(id_: str, node_idx=None, t_lim=5, n_nodes=6, offset=1):
    """ Makes and saves line plots of predictions to ../figures/.

        Saves:
            Figure "prediction_[...]"
    """
    # Get data
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    fs = params['resample']
    eval_distance = pickle.load(open('../models/' + id_ + '/eval_distances.pkl', 'rb'))
    corr = eval_distance['correlation']
    eval_prediction = pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb'))
    pred = eval_prediction['prediction']
    true = eval_prediction['true']

    if node_idx is not None:
        n_nodes = len(node_idx)
    else:
        if n_nodes == 'all':
            n_nodes = params['visible_size']
        node_idx = sorted(random.sample([i for i in range(true.shape[1])], n_nodes))
    t = np.arange(true.shape[0] / fs - t_lim, true.shape[0] / fs, 1 / fs)  # in sec.
    offset_array = np.linspace(0, (n_nodes - 1) * offset, n_nodes)

    sns.set_style('white')
    fig = plt.figure(figsize=(4, int(0.2 * params['visible_size'])))
    #fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(1, 6)

    ax0 = fig.add_subplot(gs[:, :5])
    plt.plot(t, pred[-int(t_lim * fs):, node_idx] + offset_array, color='red', label='Predicted')
    plt.plot(t, true[-int(t_lim * fs):, node_idx] + offset_array, color='black', label='LFP', lw=.7)
    # ((data - 0.5) * 2 * 3 * np.std(old)) + np.mean(old) // 71.40147, -0.009330093
    plt.plot([t[-1] - 0.05, t[-1] - 0.05], [offset_array[-1] + offset, offset_array[-1] + offset + 0.733],
             color='black', lw='2')
    ax0.text(t[-1] - .4, offset_array[-1] + offset + 0.2, '100 $\mu$V', rotation=90, fontsize=8)
    ax0.spines['right'].set_visible(False), ax0.spines['top'].set_visible(False)
    ax0.set_yticks((offset_array + np.mean(true[-int(t_lim * fs):, node_idx[0]])).tolist())
    ax0.set_yticklabels([str(i) for i in node_idx])
    ax0.set_ylabel('Node index [-]')
    ax0.set_xlim(t[0], t[-1]), ax0.set_ylim(bottom=np.mean(true[-int(t_lim * fs):, node_idx[0]]) - offset)
    ax0.set_xlabel('Time [s]')

    ax1 = fig.add_subplot(gs[:, 5:])
    plt.barh(offset_array + 0.5, width=np.asarray(corr)[node_idx], height=.3, color='gray', edgecolor='black', linewidth=.7)
    ax1.spines['right'].set_visible(False), ax1.spines['top'].set_visible(False)
    ax1.set_yticklabels([]), ax1.set_xlabel('$r$ [-]')
    ax1.set_ylim(ax0.get_ylim())
    ax1.set_xlim(0, 1)

    plt.savefig('../doc/figures/pred_' + id_ + '.png', dpi=300)
    plt.close()


def plot_performance(ids: list, save_name: str):
    """ Makes and saves box plots of results to ../figures/.

        Saves:
            Figure "boxplot_[...]"
    """
    df = pd.DataFrame()
    for _, id_ in enumerate(ids):
        eval_distance = pickle.load(open('../models/' + id_ + '/eval_distances.pkl', 'rb'))
        for i in range(len(eval_distance['id_'])):
            if 'ID11a' in eval_distance['id_'][i]:
                eval_distance['patient_id'][i] = 'P11a'
            elif 'ID11b' in eval_distance['id_'][i]:
                eval_distance['patient_id'][i] = 'P11b'
            elif 'ID07' in eval_distance['id_'][i]:
                eval_distance['patient_id'][i] = 'P7'
            elif 'ID08' in eval_distance['id_'][i]:
                eval_distance['patient_id'][i] = 'P8'
        df = df.append(pd.DataFrame(eval_distance), ignore_index=True)

    sns.set_style('ticks')
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(nrows=12, ncols=12)
    colors = ['tab:red', 'purple', 'tab:blue']
    ylims = [(0, 0.13), (0, 0.02), (0.7, 1)]
    ylabels = ['MAE [-]', 'MSE [-]', 'Correlation [-]']
    metrics = ['mae', 'mse', 'correlation']
    gs_subs = [gs[:5, :5], gs[:5, 7:], gs[7:, :5]]
    axs = [[], [], []]
    for i, ax in enumerate(axs):
        ax = fig.add_subplot(gs_subs[i])
        ax = sns.boxplot(x=df['patient_id'], y=df[metrics[i]], data=df, hue=df['brain_state'], palette=colors)
        # Change edgecolor
        c_edge = 'black'
        for k, artist in enumerate(ax.artists):
            artist.set_edgecolor(c_edge)
            for l in range(k * 6, k * 6 + 6):
                ax.lines[l].set_color(c_edge)
                ax.lines[l].set_mfc(c_edge)
                ax.lines[l].set_mec(c_edge)
        ax.set_ylim(ylims[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('')
        ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
        if i == 2:
            patches = []
            for k in range(3):
                patches.append(mpl.patches.Patch(color=colors[k]))
            plt.legend(frameon=False, bbox_to_anchor=(1, 1.05), loc='upper left',
                       labels=['First', 'Second', 'Third'], title='NREM segment', handles=patches)
        else:
            plt.legend([], [], frameon=False)

    plt.savefig('../doc/figures/performance_' + save_name + '.png', dpi=300)
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
    :math: ``\\dot{\\bar{u}}`
    """

    mean_abs = []
    mse, mae, corr = [], [], []
    patient_id = []
    brain_state = []
    batch_size = []

    for i, id_ in enumerate(ids):
        params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
        distances = pickle.load(open('../models/' + id_ + '/eval_distances.pkl', 'rb'))
        mse.append(np.median(distances['mae']))
        mae.append(np.median(distances['mse']))
        corr.append(np.median(distances['correlation']))
        # Get trained model
        model = models.GeneralRNN(params)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('../models/' + id_ + '/model.pth', map_location=device))
        W = model.W.weight.data.numpy()
        if 'ID11a' in id_:
            patient_id.append('P11a')
        elif 'ID11b' in id_:
            patient_id.append('P11b')
        elif 'ID07' in id_:
            patient_id.append('P7')
        elif 'ID08' in id_:
            patient_id.append('P8')
        brain_state.append(params['brain_state'])
        batch_size.append(params['batch_size'])
        # Get absolute weights
        W_abs = np.abs(W)
        if hidden is False:
            ch = params['visible_size']
            W_abs = np.abs(W[:ch, :ch])
        if diagonal is False:
            np.fill_diagonal(W_abs, 0)
        mean_abs.append(np.mean(W_abs))

    #return mean_abs, mse, mae, corr

    # Normalizing over bars to first bar
    n_brain_states = 3
    mean_abs_mat = np.reshape(np.asarray(mean_abs), (-1, n_brain_states))
    first_bars = np.reshape(np.repeat(mean_abs_mat[:, 0], n_brain_states), (-1, n_brain_states))
    mean_abs = (mean_abs_mat / first_bars * 100).flatten().tolist()

    df = pd.DataFrame()
    df['Patient ID'] = patient_id
    df['NREM phases'] = brain_state
    df['Mean abs. weight'] = mean_abs
    df['MAE'] = mae
    df['Batch size'] = batch_size

    sns.set_style('ticks')
    fig = plt.figure(figsize=(6, 3))
    colors = ['tab:red', 'purple', 'tab:blue']
    metrics = ['mae', 'mse', 'correlation']
    ax = sns.barplot(x='Patient ID', y='Mean abs. weight', hue='NREM phases', data=df, palette=colors)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    #ax.set_ylim(85, 100)
    ax.set_ylabel('Mean weights $|W|$ relative\n to first segment [%]')
    ax.set_xlabel('')

    patches = []
    for k in range(3):
        patches.append(mpl.patches.Patch(color=colors[k]))
    plt.legend(frameon=False, bbox_to_anchor=(1, 1.05), loc='upper left',
               labels=['First', 'Second', 'Third'], title='NREM segment', handles=patches)

    plt.tight_layout()
    plt.savefig('../doc/figures/barplots_meanabs_' + save_name + '.png', dpi=300)
    plt.close()


def plot_weighted_prediction(id_, node_idx, max_duration=.5):
    # Get model
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    fs = params['resample']
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


def plot_sudden_lack_of_input(id_: str, custom_test_set: dict=None, window_size_t=1, interrupt_t=.5):
    """ Tests model an returns and saves predicted values.

        If the prediction set is not the training set, pass a custom_test_set dictionary containing:
            'time_begin', 'duration', 'batch_size'

        Returns and saves:
            ../model/eval_prediction.pkl
    """
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and parameters
    print('Status: Load and process data for prediction.')
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    fs = params['resample']
    window_size = int(window_size_t * fs)
    interrupt = int(interrupt_t * fs)
    if custom_test_set is None:
        data_pre = utrain.pre_process(params=params)
    else:
        data_pre = utrain.pre_process(params=params, custom_test_set=custom_test_set)
    data_set = utrain.iEEG_DataSet(data_pre, window_size)
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)

    # Make model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.TestGeneralRNN(params)
    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth', map_location=device))
    model = model.to(device)

    # Evaluate model
    model.eval()

    print('Status: Start prediction with cuda = ' + str(torch.cuda.is_available()) + '.')
    with torch.no_grad():
        for X, y in data_generator:
            X, y = X.to(device), y.to(device)
            u_hist, r_hist = model(X, interrupt)
            u_hist = np.array(u_hist)
            r_hist = np.array(r_hist)
            X_hist = X[0, :, :].numpy().copy()
            break

    # Plot
    plt.figure(figsize=(9, 6))
    t = np.linspace(0, u_hist.shape[0] / fs, u_hist.shape[0])
    plt.plot(t, u_hist[:, 2], color='tab:blue', label='Node 2, predicted')
    plt.plot(t, X_hist[:, 2], color='tab:blue', linestyle=':', label='Node 2, true')
    plt.plot(t, u_hist[:, 50], color='tab:red', label='Node 50, predicted')
    plt.plot(t, X_hist[:, 50], color='tab:red', linestyle=':', label='Node 50, true')
    plt.plot(t, u_hist[:, 65], color='tab:green', label='Node 2, predicted')
    plt.plot(t, X_hist[:, 65], color='tab:green', linestyle=':', label='Node 2, true')
    plt.plot([interrupt_t, interrupt_t], [0.2, 0.78], color='black', linestyle='--')
    plt.legend(), plt.xlim(t[0], t[-1]), plt.ylim(0.2, 0.78)
    plt.xlabel('Time [s]'), plt.ylabel('Membrane potential [a. U.]')
    plt.title('Prediction with sudden lack of input')
    plt.savefig('../doc/figures/lack_of_input.png')
    plt.close()
