""" Trains and saves model

Part of master thesis Segessenmann J. (2020)
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import pickle

import utilities as util
import models


params = {'name': 'FRNN_parallel_big_tanh_bias',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 60,
          'hidden_size': 60,
          'lambda': 0.5,
          'nonlinearity': 'tanh',
          'bias': True,
          # train parameters -------------------------
          'sample_size': 20000,
          'window_size': 50,
          'normalization': True,
          'epochs': 20,
          'lr_decay': 8,
          # old parameters ---------------------------
          'loss_samples': 5,
          'epochs_per_cycle': 1,
          'cycles': 7}

# Make rotation list
ch_out_rotation = list(range(params['channel_size']))

# Load data
X_train, X_test = util.data_loader(params)

# Define model, criterion and optimizer
model = models.FRNN_parallel(params, ch_out_rotation)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
start_time = time.time()

temp_loss = np.zeros([len(X_train), len(ch_out_rotation)])
epoch_loss = np.zeros([params['epochs'], len(ch_out_rotation)])

for epoch in range(params['epochs']):
    if epoch is not 0 and epoch % params['lr_decay'] is 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
    for smpl_idx, X in enumerate(X_train):
        optimizer.zero_grad()
        loss = []
        Y_pred = model(X)
        #Y_true = X[-1, :].repeat(len(ch_out_rotation)).view(-1, params['channel_size'])
        for setup, pos in enumerate(ch_out_rotation):
            loss.append(criterion(Y_pred[setup, pos], X[-1, pos])) #Y_true[setup, pos]))
            temp_loss[smpl_idx, setup] = loss[-1].item()
        torch.autograd.backward(loss)
        optimizer.step()
    epoch_loss[epoch, :] = np.mean(temp_loss, axis=0)
    print(f'Epoch: {epoch} | Loss: {np.mean(temp_loss):.4}')

total_time = time.time() - start_time
print(f'Time [min]: {total_time/60:.3}')

# Plot loss
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.plot(np.max(epoch_loss, axis=1), c='tab:blue', ls='--', label='max/min')
ax.plot(np.min(epoch_loss, axis=1), c='tab:blue', ls='--')
ax.plot(np.quantile(epoch_loss, 0.25, axis=1), c='tab:purple', label='quartile')
ax.plot(np.quantile(epoch_loss, 0.75, axis=1), c='tab:purple')
ax.plot(np.median(epoch_loss, axis=1), c='tab:red', ls='--', lw=3, label='median')
ax.set_title(f'Losses over {len(epoch_loss)} epochs and {total_time/60:.3} minutes')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid()
fig.savefig('../doc/figures/losses_' + params['name'] + '.png')

# Save model and params to file
torch.save(model.state_dict(), '../models/' + params['name'] + '.pth')
pickle.dump(params, open('../models/' + params['name'] + '.pkl', 'wb'))
