import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from cosserat_model.plot import plot_observation_2d


dataset_folder = "dataset_20250718_171730"

files = os.listdir(dataset_folder)
files = [f for f in files if f.endswith(".pkl")]
files = sorted(files)


for file in files:
    print(f"Processing {file}")

    data = pickle.load(open(os.path.join(dataset_folder, file), "rb"))

    plot_observation_2d(data)
