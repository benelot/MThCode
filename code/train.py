""" Main file for training NNs

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


params = {'name': 'FRNN_02',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 10,
          'hidden_size': 10,
          'lambda': 0.5,
          'nonlinearity': 'relu',
          'bias': False,
          # train parameters -------------------------
          'sample_size': 1500,
          'window_size': 50,
          'normalization': True,
          'epochs_per_cycle': 1,
          'cycles': 2}

# Make rotation list
ch_out_rotation = []
for i in range(int(params['channel_size']/2)):
    ch_out_rotation.append([i, i+int(params['channel_size']/2)])

# Load data
X_train, X_test = util.data_loader(params)

# Define model, criterion and optimizer
model = models.FRNN(params)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
start_time = time.time()
cycle_loss = []

for cycle_nr in range(params['cycles']):
    if cycle_nr is 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003
    for rot_pos, ch_out in enumerate(ch_out_rotation):
        temp_loss = []
        epoch_loss = []
        model.make_gate(ch_out)

        for epoch in range(params['epochs_per_cycle']):
            for idx, X in enumerate(X_train):
                optimizer.zero_grad()
                Y_pred = model(X)
                loss = criterion(Y_pred[ch_out], X[-1, ch_out])
                loss.backward()
                optimizer.step()
                temp_loss.append(loss.item())

            epoch_loss.append(np.mean(np.asarray(temp_loss)))
            print(f'Cycle: {cycle_nr} | Rot. pos.: {rot_pos} | Epoch: {epoch} | Loss: {epoch_loss[epoch]:.4}')

        if cycle_nr is 0:
            cycle_loss.append(np.asarray(epoch_loss))
        else:
            cycle_loss[rot_pos] = np.concatenate((cycle_loss[rot_pos], np.asarray(epoch_loss)), 0)

total_time = time.time() - start_time
print(f'Time [min]: {total_time/60:.3}')

# Plot loss
fig, ax = plt.subplots(1, figsize=(5, 5))
for i in range(len(ch_out_rotation)):
    ax.plot(cycle_loss[i], label='Rot. pos. ' + str(i))
    ax.legend()
ax.set_title('Losses')
ax.set_ylabel('Loss')
ax.set_xlabel('Cycle')
ax.grid()
fig.tight_layout()
plt.show()
fig.savefig('../doc/figures/_losses.png')

# Save model and params to file
torch.save(model.state_dict(), '../models/' + params['name'] + '.pth')
pickle.dump(params, open('../models/' + params['name'] + '.pkl', 'wb'))
