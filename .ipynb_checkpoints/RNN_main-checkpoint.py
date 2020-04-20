# RNN for iEEG data
# data from  http://ieeg-swez.ethz.ch/
# Segessenmann J. 2020

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------------------------
# Dataloader
# -------------------------------------------------------------------------------------
nr_channels = 25
nr_samples = 10000
data_mat, info_mat = loadmat('./data/ID01_1h.mat'), loadmat('./data/ID01_info.mat')
data, fs = data_mat['EEG'][:nr_channels, :nr_samples].transpose(), info_mat['fs']

sc = MinMaxScaler(feature_range=(-1, 1))
sc.fit(data)
data_norm = sc.transform(data)

data_norm = torch.FloatTensor(data_norm)

train_portion = 0.8
train_set = data_norm[:int(train_portion * nr_samples), :]
test_set = data_norm[int(train_portion * nr_samples):, :]

window_size = 100
ch_input = 18  # channels used as inputs (others are outputs)
X_train, Y_train, X_test, Y_test = [], [], [], []
for i in range(train_set.shape[0] - window_size):
    X_train.append(train_set[i:i+window_size, :ch_input])
    Y_train.append(train_set[i+window_size-1, ch_input:])
for i in range(test_set.shape[0] - window_size):
    X_test.append(test_set[i:i+window_size, :ch_input])
    Y_test.append(test_set[i+window_size-1, ch_input:])

# -------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------
class RNN(nn.Module):
    def __init__(self, input_size=ch_input, hidden_size=50, output_size=nr_channels-ch_input):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        # Initialize h0 (hidden state)
        self.hidden = torch.zeros(1, 1, self.hidden_size)

    def forward(self, X):
        out_rnn, self.hidden = self.rnn(X.view(X.shape[0], 1, X.shape[1]), self.hidden)
        Y = self.linear(out_rnn.view(len(X), -1))
        return Y[-1, :]  # last value of each output channel

# -------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------
model = RNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 2

start_time = time.time()
temp_loss = []
epoch_loss = []

for epoch in range(epochs):
    for idx, X in enumerate(X_train):
        optimizer.zero_grad()
        model.hidden = torch.zeros(1, 1, model.hidden_size)
        Y_pred = model(X)
        loss = criterion(Y_pred, Y_train[idx])
        loss.backward()
        optimizer.step()
        temp_loss.append(loss.item())

    epoch_loss.append(np.mean(np.asarray(temp_loss)))
    print(f'Epoch: {epoch} Loss: {epoch_loss[epoch]}')

total_time = time.time() - start_time
print(f'Time for training [s]: {total_time / 60}')

plt.figure()
plt.plot(np.asarray(epoch_loss))
plt.xlabel('epoch'), plt.ylabel('loss')
plt.show()

# -------------------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------------------
model.eval()

Y_preds = []
for idx, X in enumerate(X_test):
    with torch.no_grad():
        model.hidden = torch.zeros(1, 1, model.hidden_size)
        Y_preds.append(model(X).numpy())

preds = np.asarray(Y_preds)
Y_test_np = np.asarray([idx.numpy() for idx in Y_test])
print(preds.shape)
print(Y_test_np.shape)

plt.figure()
plt.plot(Y_test_np[599:1000, 0])
plt.plot(preds[599:1000, 0])
plt.show()

model.rnn.all_weights()
model.linear.bi