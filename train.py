from pickle import NONE
import torch
import os
from torch import optim
from torch.autograd.grad_mode import F
from torch.nn.modules.container import ModuleList
from torch.utils.data.dataloader_experimental import DataLoader2
import modules
import data_utils
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter, writer

TRAIN_FIRST_TIME = "TRAIN_FIRST_TIME"
CONTINUE_TRAINING = "CONTINUE_TRAINING"
EVALUATE = "EVALUATE"
VISUALIZE = "VISUALIZE"
TEST = "TEST"
GET_PREDICTION = "GET_PREDICTION"
CHECK_POINT = 1


def _restore_model(model_path):
    model = modules.TestNN()  # todo: Update to new model.
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
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


def _set_up_training(lr=1e-3, mode=TRAIN_FIRST_TIME, model_path=None):
    # Set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.MSELoss()
    if mode == TRAIN_FIRST_TIME:
        model = modules.TestNN().to(device)
    elif mode == CONTINUE_TRAINING or mode == EVALUATE:
        model = _restore_model(model_path).to(device)
        if mode == EVALUATE:
            model.eval()
            return model, loss_fn, device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
            model_loss = loss_fn(pred, Wt0)
            # Backpropagation.
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
            if batch % check_step == 0:
                writer.add_scalar(
                    "loss", model_loss,
                    batch + epoch_count * size / data_loader.batch_size)
        if (epoch_count + 1) % CHECK_POINT == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_ckpt_dir, "model_ckpt.pth"))
        print(f"+++++Epoch {epoch_count+1} done+++++")
    writer.close()
    print("Training done!")
    return None


def evaluate(model_path, data_loader, mode=None):
    size = len(data_loader.dataset)
    model, loss_fn, device = _set_up_training(model_path=model_path,
                                              mode=EVALUATE)
    pred_list = []
    with torch.no_grad():
        sum_loss = 0
        for batch, (input_tensor, Wt0) in enumerate(data_loader):
            input_tensor = input_tensor.to(device)
            Wt0 = Wt0.to(device)
            pred = model(input_tensor)
            model_loss = loss_fn(pred, Wt0)
            print(f"Loss for test batch {batch} is: {model_loss}")
            sum_loss += model_loss
            if mode == GET_PREDICTION:
                pred_list.append(pred)
            if mode == VISUALIZE:
                data_utils.visualize(pred)
    return sum_loss / size, pred_list if pred_list else None


def _test_create_dataset(file_dir):
    """Pass."""
    data = data_utils.read_files(file_dir)
    dataset = modules.Rk4Dataset(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    input_tensor, Wt0 = next(iter(dataloader))
    print(input_tensor, Wt0)
    return None


def _test_create_model():
    print(_set_up_training(lr=1e-3))
    return None


def _test_train():
    file_dir = "/home/yqs/dave/pod/FlowTransformer/dataset/wt0_data"
    data_loader = load_dataset(file_dir, batch_size=10)
    model, loss_fn, optimizer, device = _set_up_training(lr=1e-3)
    train(data_loader,
          model,
          loss_fn,
          optimizer,
          device,
          check_step=10,
          epoch=5)
    print("Training done!")


if __name__ == "__main__":
    test_dir = "./test/test1"
    file_dir = "./test"
    log_dir = "./log"
    model_path = "C:/Users/Harold/Desktop/FlowPredict/log/2022_01_15_15_14/model_ckpt/model_ckpt.pth"
    # Evaluate(Visualize) pipeline.
    data_loader = load_dataset(test_dir, batch_size=1)
    avg_loss, pred_list = evaluate(model_path,
                                   data_loader,
                                   mode=VISUALIZE)
    print(f"avg_loss: {avg_loss}\n")
    print(pred_list)

    # Train pipeline.
    # data_loader = load_dataset(file_dir, batch_size=1)
    # model, loss_fn, optimizer, device = _set_up_training(lr=1e-2,
    #                                                      mode=TRAIN_FIRST_TIME)
    # train(data_loader,
    #       model,
    #       loss_fn,
    #       optimizer,
    #       device,
    #       check_step=1,
    #       epoch=10,
    #       log_dir=log_dir)
