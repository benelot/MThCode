""" Main file for evaluating NNs

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

import utilities as util
import models


# Load parameters
params = pickle.load(open('../models/FRNN_tanh.pkl', 'rb'))  # rotearlyopt normal

# Load data
X_train, X_test = util.data_loader(params)

# Get trained model
model = models.FRNN(params)
model.load_state_dict(torch.load('../models/' + params['name'] + '.pth'))

# Evaluate model
model.eval()

ch = [20, 39, 55]
model.make_gate(ch)
Y_preds = []
Y = []
for idx, X in enumerate(X_test):
    with torch.no_grad():
        Y_all = model(X).numpy()
        Y_preds.append(Y_all[ch])
        Y.append(X[-1, ch].numpy())

preds = np.asarray(Y_preds)
Y_test_np = np.asarray(Y)

fig, ax = plt.subplots(len(ch), figsize=(8, len(ch) * 2.5))
fig.tight_layout(pad=3)
for i in range(len(ch)):
    ax[i].set_title('Channel ' + str(ch[i]))
    ax[i].plot(Y_test_np[:, i], label='true')
    ax[i].plot(preds[:, i], label='predicted')
    ax[i].set_xlabel('Samples [-]')
    ax[i].set_ylabel('Magn. [-]')
    ax[i].legend()

fig.subplots_adjust(hspace=.8)
fig.savefig('../doc/figures/pred_' + params['name'] + '.png')

W = model.W.weight.data.numpy()
util.plot_weights(W=W, params=params, vmax=.5, linewidth=.0,
                  save2path='../doc/figures/weights_' + params['name'] + '.png')
