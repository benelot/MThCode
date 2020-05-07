""" Main file for evaluating NNs

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

import utilities as util
import models

plot_signal = True
plot_weights = True
plot_scatter = True
plot_box = True
ch_show = [12, 27, 33, 56]  # even number of channels

# Load parameters
params = pickle.load(open('../../models/FRNN_tanh_big.pkl', 'rb'))

# Load data
X_train, X_test = util.data_loader(params)

# Get trained model
model = models.FRNN(params)
model.load_state_dict(torch.load('../models/' + params['name'] + '.pth'))

# Evaluate model
model.eval()

# Make rotation list
ch_out_rotation = []
for i in range(10):
    inner_rot = []
    for k in range(int(params['channel_size'] / 10)):
        inner_rot.append(i+k*10)
    ch_out_rotation.append(inner_rot)

# predict test data
test_preds = np.zeros((len(X_test), params['channel_size']))
test_true = np.zeros((len(X_test), params['channel_size']))
with torch.no_grad():
    for _, ch_out in enumerate(ch_out_rotation):
        print(f'Computing channel: {ch_out}')
        model.make_gate(ch_out)
        for smpl_idx, X in enumerate(X_test):
            Y = model(X)
            test_preds[smpl_idx, ch_out] = Y[ch_out].numpy()
            test_true[smpl_idx, ch_out] = X[-1, ch_out].numpy()


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
    for i in range(int(len(ch_show)/2)):
        for k in range(int(len(ch_show)/2)):
            idx.append([i, k])
    fig, ax = plt.subplots(int(len(ch_show)/2), int(len(ch_show)/2), figsize=(10, 10))
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
    corr = []
    for i in range(params['channel_size']):
        corr.append(np.corrcoef(test_preds[:, i], test_true[:, i])[0, 1])
    print(f'Mean correlation: {np.mean(corr)}')
    plt.figure(figsize=(5, 10))
    plt.title('Correlation per channel')
    plt.grid()
    plt.ylim([0.5, 1])
    plt.boxplot(corr)
    plt.savefig('../doc/figures/corr_' + params['name'] + '.png')


if plot_weights is True:
    W = model.W.weight.data.numpy()
    util.plot_weights(W=W, params=params, vmax=.5, linewidth=.0,
                      save2path='../doc/figures/weights_' + params['name'] + '.png')
