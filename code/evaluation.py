# Main file for evaluation
# Segessenmann J. (2020)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import utilities as util
import models


path2data = '../data/ID02_1h.mat'
path2model = './FRNN_model.pth'
nr_channels = 10
hidden_size = 10
nr_samples = 1500
window_size = 50

X_train, X_test = util.data_loader(path=path2data, nr_channels=nr_channels, nr_samples=nr_samples,
                                   window_size=window_size)

model = models.FRNN(visible_size=nr_channels, hidden_size=hidden_size)
model.load_state_dict(torch.load(path2model))

model.eval()

ch = [0, 5, 9]
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
plt.show()
fig.savefig('../doc/figures/_predictions.png')

util.plot_weights(W=model.W.weight.data.numpy(), channel_size=nr_channels, vmax=0.8,
                  path='../doc/figures/_weights.png')
