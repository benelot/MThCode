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
        self.visible_size = params['visible_size']
        self.full_size = self.visible_size + self.hidden_size
        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])
        # Define non-linearity
        if params['af'] == 'linear':
            def linear(in_):
                return in_
            self.phi = linear
        else:
            exec_str = 'self.phi = torch.' + params['af']
            exec(exec_str)
        # Make gate Lambda
        self.Lambda = torch.cat((torch.zeros(self.visible_size, self.visible_size),
                                 torch.ones(self.visible_size, self.hidden_size)), 1)
        for idx in range(self.visible_size):
            self.Lambda[idx, idx] = 1

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
        self.flag = False  # flag for AF between r -> u
        self.hidden_size = params['hidden_size']
        self.visible_size = params['visible_size']
        self.full_size = self.visible_size + self.hidden_size

        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])

        # Define non-linearity
        if params['af'] == 'relu':
            self.phi = torch.relu
        elif params['af'] == 'relu_1n':
            def relu_1n(in_):
                relu = 0.5 * in_ + 0.5
                relu[relu < 0] = 0
                return relu
            self.phi = relu_1n
            self.flag = True
        elif params['af'] == 'tanh':
            self.phi = torch.tanh
        elif params['af'] == 'sigmoid':
            def sigmoid(in_):
                return 2 * (1 / (1 + torch.exp(-2 * in_))) -1
            self.phi = sigmoid
            self.flag = False
        elif params['af'] == 'linear':
            def linear(in_):
                return in_
            self.phi = linear

        # Make gate Lambda
        self.recurrence = params['lambda']
        self.Lambda = torch.cat((torch.ones(self.visible_size, self.visible_size) * self.recurrence,
                                 torch.ones(self.visible_size, self.hidden_size)), 1)
        for idx in range(self.visible_size):
            self.Lambda[idx, idx] = 1

    def forward(self, X):
        # Initialize r and i nodes
        R = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        Y = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        # Forward path
        if self.flag is False:
            for t in range(X.shape[0]):
                Y[:, :self.visible_size] = X[t, :].repeat(self.visible_size).view(-1, self.visible_size)
                U = torch.mul(self.Lambda, self.W(R)) + torch.mul((1 - self.Lambda), Y)
                R = self.phi(U)
        elif self.flag is True:
            for t in range(X.shape[0]):
                Y[:, :self.visible_size] = X[t, :].repeat(self.visible_size).view(-1, self.visible_size)
                U = torch.mul(self.Lambda, (2 * self.W(R) - 1)) + torch.mul((1 - self.Lambda), Y)
                R = self.phi(U)
        return torch.diag(U[:, :self.visible_size])


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


class TestGeneralRNN(nn.Module):
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

    def forward(self, X, t_recurrent):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize r and i nodes (X.shape[0] = n_batches, X, shape[1] = n_time_steps)
        R = torch.zeros((X.shape[0], self.visible_size, self.full_size), dtype=torch.float32).to(device)
        Y = torch.zeros((X.shape[0], self.visible_size, self.full_size), dtype=torch.float32).to(device)
        Lambda_batch = self.Lambda.repeat(X.shape[0], 1).view(X.shape[0], self.visible_size, self.full_size).to(device)
        u_hist = []
        r_hist = []
        # Forward path
        for t in range(X.shape[1]):
            if t < t_recurrent:
                Y[:, :, :self.visible_size] = X[:, t, :].repeat(self.visible_size, 1)\
                    .view(self.visible_size, X.shape[0], self.visible_size).transpose(0, 1)
                U = torch.mul(Lambda_batch, self.W(R)) + torch.mul((1 - Lambda_batch), Y)
                R = self.phi(U)

                # Get output for history
                u = torch.diagonal(U[0, :self.visible_size, :self.visible_size])
                r = torch.diagonal(R[0, :self.visible_size, :self.visible_size])
                if self.hidden_size > 0:
                    u = torch.cat([u, U[0, 0, self.visible_size:]])
                    r = torch.cat([r, R[0, 0, self.visible_size:]])
                u_hist.append(u[:self.visible_size].numpy().copy())
                r_hist.append(r[:self.visible_size].numpy().copy())
            else:
                if t == t_recurrent:
                    r = torch.diagonal(R[0, :self.visible_size, :self.visible_size])
                    if self.hidden_size > 0:
                        r = torch.cat([r, R[0, 0, self.visible_size:]])
                u = self.W(r)
                r = self.phi(u)
                u_hist.append(u[:self.visible_size].numpy().copy())
                r_hist.append(r[:self.visible_size].numpy().copy())
        return u_hist, r_hist


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
