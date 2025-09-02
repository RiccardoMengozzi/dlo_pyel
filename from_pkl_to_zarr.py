import os
import pickle
import zarr
import numpy as np
from tqdm import tqdm
import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
PKL_FOLDER = "/home/mengo/research/dlo_pyel/dataset_20250901_131816"   # cartella sorgente con i file .pkl
SAVE_PATH = f"dataset_zarr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"

os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Converting PKL dataset from {PKL_FOLDER} to Zarr dataset at {SAVE_PATH}")

# -----------------------------
# CONVERSION FUNCTION
# -----------------------------
def convert_pkl_to_zarr(pkl_file, zarr_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    root = zarr.open_group(zarr_file, mode="w")

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            root.create_dataset(
                key,
                data=value,
                chunks=True,
                compressor=zarr.get_codec({'id': 'zlib'})  # compressione
            )
        elif isinstance(value, (int, float, str, np.int32, np.float32)):
            root.attrs[key] = value
        else:
            # fallback: salvo come stringa se non è numpy/scalare
            root.attrs[key] = str(value)


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    pkl_files = sorted([f for f in os.listdir(PKL_FOLDER) if f.endswith(".pkl")])

    for fname in tqdm(pkl_files, desc="Converting"):
        pkl_path = os.path.join(PKL_FOLDER, fname)
        base_name = os.path.splitext(fname)[0] + ".zarr"
        zarr_path = os.path.join(SAVE_PATH, base_name)

        convert_pkl_to_zarr(pkl_path, zarr_path)

    print("Conversion completed.")


if __name__ == "__main__":
    main()
