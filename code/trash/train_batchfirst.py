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

params = {'name': 'FRNN__batchfirst',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 10,
          'hidden_size': 10,
          'lambda': 0.5,
          'nonlinearity': 'tanh',
          'bias': False,
          # train parameters -------------------------
          'sample_size': 1000,
          'window_size': 50,
          'normalization': True,
          'loss_samples': 5,
          'epochs_per_cycle': 1,
          'cycles': 7}

# Make rotation list
ch_out_rotation = []
for i in range(int(params['channel_size'] / 2)):
    ch_out_rotation.append([i, i + int(params['channel_size'] / 2)])

# Load data
X_train, X_test = util.data_loader(params)

# Define model, criterion and optimizer
model = models.FRNN(params)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
start_time = time.time()

epochs = 10
loss = []

epoch_loss = np.zeros([epochs, len(ch_out_rotation)])
batch_size = 100
nr_batches = 7
listloss = []
temp_loss = np.zeros([nr_batches, len(ch_out_rotation)])  # len(X_train)

for epoch in range(epochs):
    if epoch is not 0 and epoch % 8 is 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2

    for batch in range(nr_batches):
        for rot_pos, ch_out in enumerate(ch_out_rotation):
            optimizer.zero_grad()
            loss = []
            model.make_gate(ch_out)
            for smpl in range(batch_size):
                Y_pred = model(X_train[smpl + batch_size * batch])
                loss.append(criterion(Y_pred[ch_out], X_train[smpl + 100 * batch][-1, ch_out]))
                listloss.append(loss[-1].item())

            temp_loss[batch, rot_pos] = np.mean(listloss)
            torch.autograd.backward(loss)
            optimizer.step()
    epoch_loss[epoch, :] = np.mean(temp_loss, axis=0)
    print(f'Epoch: {epoch} | Loss: {np.mean(temp_loss):.4}')

total_time = time.time() - start_time
print(f'Time [min]: {total_time / 60:.3}')

# Plot loss
fig, ax = plt.subplots(1, figsize=(5, 5))
for i in range(len(ch_out_rotation)):
    ax.plot(epoch_loss[:, i], label='Rot. pos. ' + str(i))
    ax.legend()
ax.set_title('Losses')
ax.set_ylabel('Loss')
ax.set_xlabel('Cycle')
ax.grid()
fig.tight_layout()
fig.savefig('../doc/figures/losses_' + params['name'] + '.png')

# Save model and params to file
torch.save(model.state_dict(), '../models/' + params['name'] + '.pth')
pickle.dump(params, open('../models/' + params['name'] + '.pkl', 'wb'))
