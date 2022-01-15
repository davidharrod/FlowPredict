import os
from numpy.core.numeric import identity
import pandas as pd
import torch
from torch.nn.functional import layer_norm
from torch.nn.modules.activation import ReLU
from torch.utils import data
import data_utils
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torch.utils.data import Dataset


class Rk4Dataset(Dataset):
    def __init__(self, data) -> None:
        super(Dataset, self).__init__()
        self.data = data
        # Subtract lamd and nT.
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        df = self.data[idx].T
        input_tensor = torch.tensor([df["lamd"][0], df["nT"][0]])
        Wt0 = torch.tensor(df["Wt0"][0])
        # dwt0 = torch.tensor(df["dwt0"][0])
        return input_tensor, Wt0


class TestNN(nn.Module):
    """Pass test case."""

    def __init__(self):
        super(TestNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(2, 512), nn.ReLU(),
                                               nn.Linear(512, 4096), nn.ReLU(),
                                               nn.Linear(4096, 50001))

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


class FlowResNet(nn.Module):
    def __init__(self, norm_layer=None, dropout=0) -> None:
        super(FlowResNet, self).__init__()
        if norm_layer is None:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(512),
                 nn.BatchNorm1d(32768)])
        else:
            self.bn_layers = norm_layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = nn.ModuleList(
            [nn.Linear(2, 512), nn.Linear(512, 32768)])
        self.out_fc = nn.Linear(32768, 50001)

    def forward(self, x):
        for i in range(len(self.bn_layers)):
            out = self.fc_layers[i](x)
            out = self.dropout(out)
            out = self.bn_layers[i](out)
            x = self.relu(out)
        out = self.out_fc(out)
        return out


def use_default_transformer():
    return nn.Transformer(norm_first=True)

    # return nn.Transformer(kwargs)


def _test_create_dataset(file_dir):
    """Pass."""
    data = data_utils.read_files(file_dir)
    dataset = Rk4Dataset(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    input_tensor, Wt0 = next(iter(dataloader))
    print(input_tensor, Wt0)
    return None
