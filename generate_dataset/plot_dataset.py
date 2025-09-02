import matplotlib.pyplot as plt

from pyel_model.plot import plot_observation_2d

import os
import zarr
import numpy as np

def load_all_zarr(folder_path):
    simulations = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".zarr"):
            path = os.path.join(folder_path, fname)
            root = zarr.open_group(path, mode="r")

            sim_data = {}
            # carica i dataset come array numpy
            for key in root.array_keys():
                sim_data[key] = root[key][:]
            # carica gli attributi
            sim_data.update(root.attrs.asdict())

            simulations.append(sim_data)

    return np.array(simulations)

data = load_all_zarr("dataset_zarr_20250902_161338")
for d in data:
    plot_observation_2d(d)
