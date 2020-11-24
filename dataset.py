import os
import time
import torch
import pandas as pd
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import text_classification
from torchtext import datasets
from torchtext.data import Field, Iterator, TabularDataset, Iterator



def make_x_y(batch):
    y, X = zip(*batch)
    device = torch.device('cpu')
    #Convert into sorted packed sequences.
    X_len, X_idx = torch.tensor([len(x) for x in X],
        dtype=torch.int16, device=device, requires_grad=False)\
        .sort(descending=True)
    X = nn.utils.rnn.pad_sequence(X, batch_first=True)[X_idx]
    X = nn.utils.rnn.pack_padded_sequence(X, X_len, batch_first=True)
    y = torch.tensor(y, dtype=torch.int64, device=device, requires_grad=False)
    return X, y
