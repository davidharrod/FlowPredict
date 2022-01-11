import torch
from torch import optim
from torch.nn.modules import loss
from torch.nn.modules.container import ModuleList
from torch.utils.data.dataloader_experimental import DataLoader2
import modules
import pandas as pd
import numpy as np
import data_utils
from torch import nn
from torch.utils.data import DataLoader, dataloader

TRAIN_FIRST_TIME = "TRAIN_FIRST_TIME"
CONTINUE_TRAINING = "CONTINUE_TRAINING"
TEST = "TEST"


def _restore_model(model_path):
    model = modules.TestNN()  # todo: Update to new model.
    model.load_state_dict(torch.load(model_path))
    return model


def load_dataset(file_dir, batch_size, shuffle=True):
    data = data_utils.read_files(file_dir)
    dataset = modules.Rk4Dataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def set_up_training(lr, mode=TRAIN_FIRST_TIME, model_path=None):
    # Set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == TRAIN_FIRST_TIME:
        model = modules.TestNN().to(device)
    elif mode == CONTINUE_TRAINING:
        model = _restore_model(model_path)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fn, optimizer, device


def train(data_loader, modle, loss_fn, optimizer, device, check_step, epoch):
    size = len(data_loader.dataset)
    modle.train()
    for epoch_count in range(epoch):
        print(f"Epoch {epoch_count+1}\n-------------------------------")
        for batch, (input_tensor, Wt0) in enumerate(data_loader):
            input_tensor, Wt0 = input_tensor.to(device), Wt0.to(device)
            # Compute prediction error.
            pred = modle(input_tensor)
            loss = loss_fn(pred, Wt0)
            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % check_step == 0:
                loss, current = loss.item(), batch * len(input_tensor)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Training done!")
    return None


def _test_create_model():
    print(set_up_training(lr=1e-3))
    return None


if __name__ == "__main__":
    file_dir = "/home/yqs/dave/pod/FlowTransformer/test"
    data_loader = load_dataset(file_dir, batch_size=2)
    model, loss_fn, optimizer, device = set_up_training(lr=1e-3)
    train(data_loader,
          model,
          loss_fn,
          optimizer,
          device,
          check_step=10,
          epoch=5)
    print("Training done!")
