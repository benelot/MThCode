""" NN models

Part of master thesis Segessenmann J. (2020)
"""

import numpy as np
import torch
import torch.nn as nn


class parallel_RNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
        if params['reverse_nodes'] is True:
            self.visible_size = self.visible_size * 2
        self.full_size = self.visible_size + self.hidden_size
        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])
        # Define non-linearity
        exec_str = 'self.phi = torch.' + params['non-linearity']
        exec(exec_str)
        # Make gate Lambda
        self.Lambda = torch.cat((torch.zeros(self.visible_size, self.visible_size),
                                 torch.ones(self.visible_size, self.hidden_size)), 1)
        for idx in range(self.visible_size):
            self.Lambda[idx, idx] = 1
            if params['reverse_nodes'] is True:
                self.Lambda[idx, idx + int(self.visible_size/2)] = 1

    def forward(self, X):
        # Initialize r and i nodes
        n_window = int(X.shape[1]/self.visible_size)
        vs = self.visible_size
        Lambda_parallel = torch.from_numpy(np.tile(self.Lambda.numpy(), (n_window, 1)))
        R = torch.zeros((X.shape[1], self.full_size), dtype=torch.float32)
        Y = torch.zeros((X.shape[1], self.full_size), dtype=torch.float32)
        out = torch.zeros((X.shape[1]))
        # Forward path
        for t in range(X.shape[0]):
            X_t_stacked = torch.from_numpy(np.tile(X[t, :], (vs, 1)))
            for i in range(n_window):
                Y[i*vs : i*vs+vs, :vs] = X_t_stacked[:, i*vs : i*vs+vs]
            U = torch.mul(Lambda_parallel, self.W(R)) + torch.mul((1 - Lambda_parallel), Y)
            R = self.phi(U)
        for i in range(n_window):
            out[i*vs : i*vs+vs] = torch.diag(U[i * vs: i * vs + vs, :vs])
        return out


class IS_RNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
        if params['reverse_nodes'] is True:
            self.visible_size = self.visible_size * 2
        self.full_size = self.visible_size + self.hidden_size
        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])
        # Define non-linearity
        exec_str = 'self.phi = torch.' + params['non-linearity']
        exec(exec_str)
        # Make gate Lambda
        self.recurrence = params['lambda']
        self.Lambda = torch.cat((torch.ones(self.visible_size, self.visible_size) * self.recurrence,
                                 torch.ones(self.visible_size, self.hidden_size)), 1)
        for idx in range(self.visible_size):
            self.Lambda[idx, idx] = 1
            if params['reverse_nodes'] is True:
                self.Lambda[idx, idx + int(self.visible_size/2)] = 1

    def forward(self, X):
        # Initialize r and i nodes
        R = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        Y = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        # Forward path
        for t in range(X.shape[0]):
            Y[:, :self.visible_size] = X[t, :].repeat(self.visible_size).view(-1, self.visible_size)
            U = torch.mul(self.Lambda, self.W(R)) + torch.mul((1 - self.Lambda), Y)
            R = self.phi(U)  #1 / (1 + torch.exp(-U*4))  #
        return torch.diag(U[:, :self.visible_size])


class IN_RNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
        if params['reverse_nodes'] is True:
            self.visible_size = self.visible_size * 2
        self.full_size = self.visible_size + self.hidden_size
        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])
        # Define non-linearity
        exec_str = 'self.phi = torch.' + params['non-linearity']
        exec(exec_str)
        # Make gate Lambda
        self.Lambda = torch.cat((torch.zeros(self.visible_size, self.visible_size),
                                 torch.ones(self.visible_size, self.hidden_size)), 1)
        for idx in range(self.visible_size):
            self.Lambda[idx, idx] = 1
            if params['reverse_nodes'] is True:
                self.Lambda[idx, idx + int(self.visible_size/2)] = 1

    def forward(self, X):
        # Initialize r and i nodes
        R = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        Y = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        # Forward path
        for t in range(X.shape[0]):
            Y[:, :self.visible_size] = X[t, :].repeat(self.visible_size).view(-1, self.visible_size)
            U = torch.mul(self.Lambda, self.W(R)) + torch.mul((1 - self.Lambda), Y)
            R = self.phi(U)  #1 / (1 + torch.exp(-U*4))  #
        return torch.diag(U[:, :self.visible_size])


class AS_RNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
        if params['reverse_nodes'] is True:
            self.visible_size = self.visible_size * 2
        self.full_size = self.visible_size + self.hidden_size
        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])
        # Define non-linearity
        exec_str = 'self.phi = torch.' + params['non-linearity']
        exec(exec_str)
        # Make gate Lambda
        self.recurrence = params['lambda']
        self.Lambda = torch.ones(self.full_size)
        self.Lambda[:self.visible_size] = self.recurrence

    def forward(self, X):
        # Initialize r and i nodes
        R = torch.zeros(self.full_size, dtype=torch.float32)
        Y = torch.zeros(self.full_size, dtype=torch.float32)
        # Forward path
        for t in range(X.shape[0]):
            Y[:self.visible_size] = X[t, :]
            U = torch.mul(self.Lambda, self.W(R)) + torch.mul((1 - self.Lambda), Y)
            R = self.phi(U)  #1 / (1 + torch.exp(-U*4))  #
        return U[:self.visible_size]


class general_RNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
        if params['reverse_nodes'] is True:
            self.visible_size = self.visible_size * 2
        self.full_size = self.visible_size + self.hidden_size

        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])

        # Define non-linearity
        if params['non-linearity'] == 'relu':
            self.phi = torch.relu
        elif params['non-linearity'] == 'sigmoid0':
            def sigmoid(in_):
                return 1 / (1 + torch.exp(-4 * (in_ - 0.5)))
            self.phi = sigmoid
        elif params['non-linearity'] == 'sigmoid1':
            def sigmoid(in_):
                return 1 / (1 + torch.exp(-6 * (in_ - 0.5)))
            self.phi = sigmoid
        elif params['non-linearity'] == 'sigmoid2':
            def sigmoid(in_):
                return 1 / (1 + torch.exp(-8 * (in_ - 0.5)))
            self.phi = sigmoid
        elif params['non-linearity'] == 'sigmoid3':
            def sigmoid(in_):
                return 1 / (1 + torch.exp(-10 * (in_ - 0.5)))
            self.phi = sigmoid
        elif params['non-linearity'] == 'linear':
            def linear(in_):
                return in_
            self.phi = linear

        # Make gate Lambda
        self.recurrence = params['lambda']
        self.Lambda = torch.cat((torch.ones(self.visible_size, self.visible_size) * self.recurrence,
                                 torch.ones(self.visible_size, self.hidden_size)), 1)
        for idx in range(self.visible_size):
            self.Lambda[idx, idx] = 1
            if params['reverse_nodes'] is True:
                self.Lambda[idx, idx + int(self.visible_size/2)] = 1

    def forward(self, X):
        # Initialize r and i nodes
        R = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        Y = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        # Forward path
        for t in range(X.shape[0]):
            Y[:, :self.visible_size] = X[t, :].repeat(self.visible_size).view(-1, self.visible_size)
            U = torch.mul(self.Lambda, self.W(R)) + torch.mul((1 - self.Lambda), Y)
            R = self.phi(U)
        return torch.diag(U[:, :self.visible_size])
