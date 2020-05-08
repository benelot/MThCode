""" Main file

Part of master thesis Segessenmann J. (2020)
"""
import utilities as util

params = {'name': 'FRNN_try',
          'path2data': '../data/ID02_1h.mat',
          # model parameters ------------------------
          'channel_size': 20,
          'hidden_size': 20,
          'lambda': 0.5,
          'nonlinearity': 'tanh',
          'bias': True,
          # train parameters -------------------------
          'sample_size': 1000,
          'window_size': 50,
          'normalization': True,
          'epochs': 5,
          'lr_decay': 8}

#util.train(params)
test_corr, train_corr = util.evaluation('FRNN_try', [2, 4, 8, 14], eval_train=True)

print(test_corr)
print(train_corr)
