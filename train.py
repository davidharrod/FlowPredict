import torch
from torch.nn.modules.container import ModuleList
from torch.utils.data.dataloader_experimental import DataLoader2
import modules
import pandas as pd
import numpy as np
import data_utils
from torch import nn
from torch.utils.data import DataLoader


def load_dataset(file_dir, batch_size, shuffle=True):
    data = data_utils.read_files(file_dir)
    dataset = modules.Rk4Dataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_model():
    # Get device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = modules.TestNN().to(device)
    return model


def _test_create_model():
    print(create_model())
    return None


if __name__ == "__main__":
    _test_create_model()
    print("done!")