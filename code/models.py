""" NN models

Part of master thesis Segessenmann J. (2020)
"""

import numpy as np
import torch
import torch.nn as nn


class GeneralRNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['visible_size']
        self.full_size = self.hidden_size + self.visible_size

        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])

        # Define non-linearity
        if params['af'] == 'relu':
            self.phi = torch.relu
        elif params['af'] == 'tanh':
            self.phi = torch.tanh
        elif params['af'] == 'linear':
            def linear(in_):
                return in_
            self.phi = linear
        else:
            print('Error; No valid activation function.')

        # Make gate Lambda
        self.recurrence = params['lambda']
        self.Lambda = torch.cat((torch.ones(self.visible_size, self.visible_size) * self.recurrence,
                                 torch.ones(self.visible_size, self.hidden_size)), 1)
        for idx in range(self.visible_size):
            self.Lambda[idx, idx] = 1

    def forward(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize r and i nodes (X.shape[0] = n_batches, X, shape[1] = n_time_steps)
        R = torch.zeros((X.shape[0], self.visible_size, self.full_size), dtype=torch.float32).to(device)
        Y = torch.zeros((X.shape[0], self.visible_size, self.full_size), dtype=torch.float32).to(device)
        Lambda_batch = self.Lambda.repeat(X.shape[0], 1).view(X.shape[0], self.visible_size, self.full_size).to(device)
        # Forward path
        for t in range(X.shape[1]):
            Y[:, :, :self.visible_size] = X[:, t, :].repeat(self.visible_size, 1)\
                .view(self.visible_size, X.shape[0], self.visible_size).transpose(0, 1)
            U = torch.mul(Lambda_batch, self.W(R)) + torch.mul((1 - Lambda_batch), Y)
            R = self.phi(U)
        return torch.diagonal(U[:, :, :self.visible_size], dim1=-1, dim2=-2)


class SingleLayer(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        # Parameters
        self.visible_size = params['visible_size']

        # Create FC layer
        self.W = nn.Linear(self.visible_size, self.visible_size, bias=params['bias'])

        # Define non-linearity
        if params['af'] == 'relu':
            self.phi = torch.relu
        elif params['af'] == 'tanh':
            self.phi = torch.tanh
        elif params['af'] == 'linear':
            def linear(in_):
                return in_
            self.phi = linear
        else:
            print('Error; No valid activation function.')

    def forward(self, x):
        r = self.phi(x)
        u = self.W(r)
        return u
