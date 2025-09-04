import os
import pickle
import zarr
import numpy as np
from tqdm import tqdm
import datetime
from multiprocessing import Pool, cpu_count


# -----------------------------
# CONFIGURATION
# -----------------------------
PKL_FOLDER = "/home/lar/Riccardo/dlo_pyel/dataset_20250901_132645"
SAVE_PATH = f"dataset_zarr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"

os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Converting PKL dataset from {PKL_FOLDER} to Zarr dataset at {SAVE_PATH}")


# -----------------------------
# CONVERSION FUNCTION
# -----------------------------
def convert_pkl_to_zarr(fname):
    pkl_path, zarr_path = fname
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    root = zarr.open_group(zarr_path, mode="w")

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            root.create_dataset(
                key,
                data=value,
                chunks=True,
                compressor=False  
            )
        elif isinstance(value, (int, float, str, np.int32, np.float32)):
            root.attrs[key] = value
        else:
            root.attrs[key] = str(value)


# -----------------------------
# MAIN LOOP (PARALLEL)
# -----------------------------
def main():
    pkl_files = sorted([f for f in os.listdir(PKL_FOLDER) if f.endswith(".pkl")])
    tasks = [
        (
            os.path.join(PKL_FOLDER, fname),
            os.path.join(SAVE_PATH, os.path.splitext(fname)[0] + ".zarr"),
        )
        for fname in pkl_files
    ]

    with Pool(processes=32) as pool:
        list(tqdm(pool.imap_unordered(convert_pkl_to_zarr, tasks), total=len(tasks)))

    print("Conversion completed.")


if __name__ == "__main__":
    main()
