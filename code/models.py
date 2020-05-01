# Models for iEEG connectome
# Segessenmann J. 2020

import numpy as np
import torch
import torch.nn as nn


class FRNN(nn.Module):
    def __init__(self, visible_size: int, hidden_size: int, recurrence=0.5):
        super().__init__()
        # Parameters
        self.visible_size = visible_size
        self.full_size = self.visible_size + hidden_size
        # Create FC Layer
        self.W = nn.Linear(self.full_size, self.full_size, bias=False)
        # Initialize gate s
        self.Lambda = torch.ones(self.full_size, dtype=torch.float32)
        self.recurrence = recurrence

    def make_gate(self, channel_pos_out: list, new_recurrence=None):
        # Define input channels
        channel_pos_in = list(range(0, self.visible_size))
        for _, val in enumerate(channel_pos_out):
            channel_pos_in.remove(val)
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
        # Forward path
        for idx in range(X.shape[0]):
            i[:self.visible_size] = X[idx, :]
            u = torch.mul(self.Lambda, self.W(r)) + torch.mul((1 - self.Lambda), i)
            r = torch.tanh(u)
        return u[:self.visible_size]
