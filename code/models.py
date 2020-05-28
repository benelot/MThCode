""" NN models

Part of master thesis Segessenmann J. (2020)
"""

import numpy as np
import torch
import torch.nn as nn

class RNN_experimental(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
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

    def forward(self, X):
        # Initialize r and i nodes
        R = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        I = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        # Forward path
        for t in range(X.shape[0]):
            I[:, :self.visible_size] = X[t, :].repeat(self.visible_size).view(-1, self.visible_size)
            U = torch.mul(self.Lambda, self.W(R)) + torch.mul((1 - self.Lambda), I)
            #R = self.phi(U)
            R = (1 / (1 + torch.exp(-U))) + U*0.3
        return U


class IS_RNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
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

    def forward(self, X):
        # Initialize r and i nodes
        R = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        I = torch.zeros((self.visible_size, self.full_size), dtype=torch.float32)
        # Forward path
        for t in range(X.shape[0]):
            I[:, :self.visible_size] = X[t, :].repeat(self.visible_size).view(-1, self.visible_size)
            U = torch.mul(self.Lambda, self.W(R)) + torch.mul((1 - self.Lambda), I)
            R = self.phi(U)  #1 / (1 + torch.exp(-U*4))  #
        return torch.diag(U[:, :self.visible_size])


class IN_RNN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
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
