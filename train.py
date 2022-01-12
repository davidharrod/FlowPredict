import torch
import os
from torch import optim
from torch.autograd.grad_mode import F
from torch.nn.modules import loss
from torch.nn.modules.container import ModuleList
from torch.utils.data.dataloader_experimental import DataLoader2
import modules
import data_utils
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter, writer

TRAIN_FIRST_TIME = "TRAIN_FIRST_TIME"
CONTINUE_TRAINING = "CONTINUE_TRAINING"
TEST = "TEST"
CHECK_POINT = 100


def _restore_model(model_path):
    model = modules.TestNN()  # todo: Update to new model.
    model.load_state_dict(torch.load(model_path))
    return model


def load_dataset(file_dir, batch_size, shuffle=True):
    data = data_utils.read_files(file_dir)
    dataset = modules.Rk4Dataset(data)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=True)
    print("=====Dataset loaded=====")
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
    print("=====Training set up=====")
    return model, loss_fn, optimizer, device


def train(data_loader, model, loss_fn, optimizer, device, check_step, epoch,
          log_dir):
    size = len(data_loader.dataset)
    model.train()
    tensorboard_dir = data_utils.make_dir_for_current_time(
        log_dir, "tensorboard")
    model_ckpt_dir = data_utils.make_dir_for_current_time(
        log_dir, "model_ckpt")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print("=====Start training=====")
    for epoch_count in range(epoch):
        print(f"=====Epoch {epoch_count+1} start training=====")
        for batch, (input_tensor, Wt0) in enumerate(data_loader):
            input_tensor, Wt0 = input_tensor.to(device), Wt0.to(device)
            # Compute prediction error.
            pred = model(input_tensor)
            loss = loss_fn(pred, Wt0)
            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % check_step == 0:
                writer.add_scalar(
                    "loss", loss,
                    batch + epoch_count * size / data_loader.batch_size)
        if (epoch_count + 1) % CHECK_POINT == 0:
            torch.save(model, os.path.join(model_ckpt_dir, "model_ckpt.pt"))
        print(f"+++++Epoch {epoch_count+1} done+++++")
    writer.close()
    print("Training done!")
    return None


def _test_create_dataset(file_dir):
    """Pass."""
    data = data_utils.read_files(file_dir)
    dataset = modules.Rk4Dataset(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    input_tensor, Wt0 = next(iter(dataloader))
    print(input_tensor, Wt0)
    return None


def _test_create_model():
    print(set_up_training(lr=1e-3))
    return None


def _test_train():
    file_dir = "/home/yqs/dave/pod/FlowTransformer/dataset/wt0_data"
    data_loader = load_dataset(file_dir, batch_size=10)
    model, loss_fn, optimizer, device = set_up_training(lr=1e-3)
    train(data_loader,
          model,
          loss_fn,
          optimizer,
          device,
          check_step=10,
          epoch=5)
    print("Training done!")


if __name__ == "__main__":
    test_dir = "/home/yqs/dave/pod/FlowTransformer/test"
    file_dir = "/home/yqs/dave/pod/FlowTransformer/dataset/wt0_data"
    log_dir = "./log"
    data_loader = load_dataset(file_dir, batch_size=50)
    model, loss_fn, optimizer, device = set_up_training(lr=1e-3)
    train(data_loader,
          model,
          loss_fn,
          optimizer,
          device,
          check_step=10,
          epoch=300,
          log_dir=log_dir)
