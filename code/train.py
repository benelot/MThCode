# Main file for training
# Segessenmann J. (2020)

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn

import utilities as util
import models


path2data = '../data/ID02_1h.mat'
path2model = './FRNN_model.pth'
nr_channels = 20
hidden_size = 20
nr_samples = 2000
window_size = 50

epochs = 1
rotations = 6

ch_out_rotation = []
for i in range(int(nr_channels/2)):
    ch_out_rotation.append([i, i + int(nr_channels/2)])

X_train, X_test = util.data_loader(path=path2data, nr_channels=nr_channels, nr_samples=nr_samples,
                                   window_size=window_size)

model = models.FRNN(visible_size=nr_channels, hidden_size=hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
rot_loss = []

for rot_nr in range(rotations):
    if rot_nr is 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003
    for rot, ch_out in enumerate(ch_out_rotation):
        temp_loss = []
        epoch_loss = []
        model.make_gate(ch_out)

        for epoch in range(epochs):
            for idx, X in enumerate(X_train):
                optimizer.zero_grad()
                Y_pred = model(X)
                loss = criterion(Y_pred[ch_out], X[-1, ch_out])
                loss.backward()
                optimizer.step()
                temp_loss.append(loss.item())

            epoch_loss.append(np.mean(np.asarray(temp_loss)))
            print(f'Rotation: {rot} Epoch: {epoch} Loss: {epoch_loss[epoch]}')

        if rot_nr is 0:
            rot_loss.append(np.asarray(epoch_loss))
        else:
            rot_loss[rot] = np.concatenate((rot_loss[rot], np.asarray(epoch_loss)), 0)

total_time = time.time() - start_time
print('Time [min]: ' + str(total_time / 60))

fig, ax = plt.subplots(1, figsize=(5, 5))
for i in range(len(ch_out_rotation)):
    ax.plot(rot_loss[i], label='rot. ' + str(i))
    ax.legend()
ax.set_title('Losses')
ax.set_ylabel('loss')
ax.set_xlabel('cycle')
ax.grid()
fig.tight_layout()
fig.savefig('../doc/figures/_losses.png')

torch.save(model.state_dict(), path2model)


