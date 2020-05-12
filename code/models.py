""" NN models

Part of master thesis Segessenmann J. (2020)
"""

import numpy as np
import torch
import torch.nn as nn


class FRNN_old(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        # Parameters
        self.name = params['name']
        self.visible_size = params['channel_size']
        self.full_size = params['channel_size'] + params['hidden_size']
        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])
        # Define non-linearity
        exec_str = 'self.nonlinearity = torch.' + params['nonlinearity']
        exec(exec_str)
        # Initialize gate s
        self.Lambda = torch.ones(self.full_size, dtype=torch.float32)
        self.recurrence = params['lambda']

    def make_gate(self, channel_pos_out: list, new_recurrence=None):
        # Define input channels
        channel_pos_in = list(range(0, self.visible_size))
        if type(channel_pos_out) is list:
            for _, val in enumerate(channel_pos_out):
                channel_pos_in.remove(val)
        elif type(channel_pos_out) is int:
            channel_pos_in.remove(channel_pos_out)
        # Optional new recurrence factor
        if new_recurrence is not None:
            self.recurrence = new_recurrence
        # Define gate s
        self.Lambda = torch.ones(self.full_size, dtype=torch.float32)
        self.Lambda[channel_pos_in] = self.recurrence

    def forward(self, X):
        # Initialize r and i nodes
        r = torch.zeros(self.full_size, dtype=torch.float32)
        i = torch.zeros(self.full_size, dtype=torch.float32)
        #u = torch.zeros(X.shape[0], self.full_size, dtype=torch.float32)
        # Forward path
        for idx in range(X.shape[0]):
            i[:self.visible_size] = X[idx, :]
            u = torch.mul(self.Lambda, self.W(r)) + torch.mul((1 - self.Lambda), i)
            r = self.nonlinearity(u)
        return u[:self.visible_size]


class FRNN_parallel(nn.Module):
    def __init__(self, params: dict, ch_out_positions):
        super().__init__()
        # Parameters
        self.name = params['name']
        self.hidden_size = params['hidden_size']
        self.visible_size = params['channel_size']
        self.full_size = params['channel_size'] + params['hidden_size']
        self.out_pos = len(ch_out_positions)
        # Create FC layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=params['bias'])
        # Define non-linearity
        exec_str = 'self.nonlinearity = torch.' + params['nonlinearity']
        exec(exec_str)
        # Make gate Lambda
        self.recurrence = params['lambda']
        self.Lambda = torch.cat((torch.ones(self.out_pos, self.visible_size) * self.recurrence,
                                 torch.ones(self.out_pos, self.hidden_size)), 1)
        for setup, node in enumerate(ch_out_positions):
            self.Lambda[setup, node] = 1

    def forward(self, X):
        # Initialize r and i nodes
        r = torch.zeros((self.out_pos, self.full_size), dtype=torch.float32)
        i = torch.zeros((self.out_pos, self.full_size), dtype=torch.float32)
        # Forward path
        for t in range(X.shape[0]):
            i[:, :self.visible_size] = X[t, :].repeat(self.out_pos).view(-1, self.visible_size)
            u = torch.mul(self.Lambda, self.W(r)) + torch.mul((1 - self.Lambda), i)
            r = self.nonlinearity(u)
        return u[:, :self.visible_size]


class FRNN(nn.Module):
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
            R = self.phi(U)
        return torch.diag(U[:, :self.visible_size])
