import os
import pandas as pd
import torch
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
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        df = self.data[idx].T
        input_tensor = torch.tensor([df["lamd"][0], df["nT"][0]])
        Wt0 = torch.tensor(df["Wt0"][0])
        dwt0 = torch.tensor(df["dwt0"][0])
        return input_tensor, Wt0, dwt0


class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(2, 512), nn.ReLU(),
                                               nn.Linear(512, 1024), nn.ReLU(),
                                               nn.Linear(1024, 50001))

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def _test_create_dataset(file_dir):
    """Pass."""
    data = data_utils.read_files(file_dir)
    dataset = Rk4Dataset(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    input_tensor, Wt0, dwt0 = next(iter(dataloader))
    print(input_tensor, Wt0, dwt0)
    return None


if __name__ == "__main__":
    file_dir = "/home/yqs/dave/pod/test"
    _test_create_dataset(file_dir)