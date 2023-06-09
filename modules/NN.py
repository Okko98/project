from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


# The code was copied from https://github.com/b04901014/FT-w2v2-ser. All rights belong to the authors.

class LinearHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        #input: (B, L, C)
        super().__init__()
        self.l1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.act = nn.Tanh
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=n_classes)

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.l2(x)
        return x

    def reset_parameters(self):
        self.l1.reset_parameters()
        self.l2.reset_parameters()

class RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        #Input: (B, L, C)
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim//2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.relu(x.mean(1))
        return x

