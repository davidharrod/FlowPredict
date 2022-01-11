import os
from sys import dont_write_bytecode
import pandas as pd


def _dict_2_df(dict):
    return pd.DataFrame.from_dict(dict, orient="index")


def _get_file_path(dir, file):
    return os.path.join(dir, file)


def _str_2_float(str):
    return float(str)


def _read_file(file):
    file_as_dict = {}
    Wt0 = []
    dwt0 = []
    flag = 1
    with open(file) as file:
        for i, line in enumerate(file.readlines()):
            line = line.split(" ")[0]
            if i == 0:
                file_as_dict["lamd"] = _str_2_float(line)
            elif i == 1:
                file_as_dict["nT"] = _str_2_float(line)
            if "=" in line:
                if i > 3:
                    flag = 0
                continue
            if flag:
                Wt0.append(line)
            else:
                dwt0.append(line)
    file_as_dict["Wt0"] = list(map(_str_2_float, Wt0))
    file_as_dict["dwt0"] = list(map(_str_2_float, dwt0))
    return file_as_dict


def read_files(file_dir):
    dataset = []
    for _, _, files in os.walk(file_dir):
        for file in files:
            file_path = _get_file_path(file_dir, file)
            if _read_file(file_path):
                dataset.append(_read_file(file_path))
    return list(map(_dict_2_df, dataset))


if __name__ == "__main__":
    file_dir = "/home/yqs/dave/pod/test"
    a = read_files(file_dir)
    print("done!")