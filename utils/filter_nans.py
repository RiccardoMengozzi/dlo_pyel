import os
import glob
import pickle
import numpy as np
import argparse


def check_and_remove_nan(dataset_path):

    data_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
    print(f"Found {len(data_files)} files")
    for file in data_files:

        try:
            data = pickle.load(open(file, "rb"))

            x0 = data["init_shape"].T
            x1 = data["final_shape"].T
        except Exception as e:
            print(f"Error: {e}")
            x0 = np.array([np.nan])
            x1 = np.array([np.nan])

        if np.isnan(x0).any() or np.isnan(x1).any():
            print(f"Removing {file}")
            os.remove(file)


def check_len(dataset_path):
    LEN_RANGE = [0.45, 0.6]
    data_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
    print(f"Found {len(data_files)} files")

    for file in data_files:
        data = pickle.load(open(file, "rb"))
        x0 = data["init_shape"].T
        x1 = data["final_shape"].T

        len_x0 = np.linalg.norm(np.diff(x0, axis=0), axis=-1).sum()
        len_x1 = np.linalg.norm(np.diff(x1, axis=0), axis=-1).sum()

        if len_x0 < LEN_RANGE[0] or len_x0 > LEN_RANGE[1] or len_x1 < LEN_RANGE[0] or len_x1 > LEN_RANGE[1]:
            print(f"Removing {file}")
            os.remove(file)


if __name__ == "__main__":

    files_path = "/home/lar/dev25/DLO_DIFFUSION/DATA/action_5cm_22deg/10"
    check_and_remove_nan(files_path)

    check_len(files_path)
