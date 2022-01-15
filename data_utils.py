import os
from posixpath import dirname
from sys import dont_write_bytecode
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _dict_2_df(dict):
    return pd.DataFrame.from_dict(dict, orient="index")


def _get_file_path(dir, file):
    return os.path.join(dir, file)


def _str_2_float(str):
    return float(str)


def _read_file(file):
    file_as_dict = {}
    Wt0 = []
    # dwt0 = []
    # flag = 1
    with open(file) as file:
        for i, line in enumerate(file.readlines()):
            line = line.split(" ")[0]
            if i == 0:
                file_as_dict["lamd"] = _str_2_float(line)
            elif i == 1:
                file_as_dict["nT"] = _str_2_float(line)
            else:
                if "=" in line:
                    if i > 3:
                        break
                    continue
                # if flag:
                Wt0.append(line)
                # else:
                #     dwt0.append(line)
    file_as_dict["Wt0"] = list(map(_str_2_float, Wt0))
    # file_as_dict["dwt0"] = list(map(_str_2_float, dwt0))
    return file_as_dict


def read_files(file_dir):
    dataset = []
    for _, _, files in os.walk(file_dir):
        for file in files:
            file_path = _get_file_path(file_dir, file)
            if _read_file(file_path):
                dataset.append(_read_file(file_path))
    return list(map(_dict_2_df, dataset))


def _try_2_create_directory(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        return dir
    else:
        return dir


def make_dir_for_current_time(target_dir, dir_name=None):
    current_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    dir = os.path.join(target_dir, f"{current_time}", dir_name)
    return _try_2_create_directory(dir)


def visualize(input_tensor):
    x = np.asarray([i/500 for i in range(50001)]).reshape(50001, 1)
    data = input_tensor.numpy().reshape(50001, 1)
    plt.plot(x, data)
    # Set axis.
    plt.title("Prediction")
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.axis([0, 100, -1.5e-1, 2e-1])
    plt.show()
    input()
    return None
