import time
from multiprocessing import Pool, cpu_count
from pyel_model.dlo_model import DloModel, DloModelParams
import numpy as np
import datetime
import os
import pickle

# -----------------------------
# CONFIGURATION
# -----------------------------
ITERS = 100        # total number of simulations
MAX_DISP = 0.075
MAX_ROT = np.pi / 4
RESET_EVERY = 20
SAVE_PATH = f"dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"

CONFIG = {
    "dt": 1e-5,
    "nu": 1,
    "node_gap": 0.01,
    "length": 0.5,
    "radius": 0.005,
    "density": 1e3,
    "youngs_modulus": 1e7,
    "action_velocity": 0.1,
}

# -----------------------------
# SETUP
# -----------------------------
dlo_params = DloModelParams(
    dt=CONFIG["dt"],
    n_elem=int(CONFIG["length"] / CONFIG["node_gap"]),
    length=CONFIG["length"],
    radius=CONFIG["radius"],
    density=CONFIG["density"],
    youngs_modulus=CONFIG["youngs_modulus"],
    nu=CONFIG["nu"],
    action_velocity=CONFIG["action_velocity"],
)

os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Saving data to {SAVE_PATH}")


# -----------------------------
# WORKER FUNCTION
# -----------------------------
def run_block(start_idx):
    """
    Run a block of simulations sequentially.
    Keeps the reset/continuation logic intact within the block.
    """
    init_shape, init_directors = None, None

    for i in range(start_idx, start_idx + BLOCK_SIZE):
        if i >= ITERS:  # safety stop for last block
            break

        # random action
        action_1 = np.random.randint(0, dlo_params.n_elem - 1)
        action_2 = np.random.uniform(-MAX_DISP, MAX_DISP)
        action_3 = np.random.uniform(-MAX_DISP, MAX_DISP)
        action_4 = np.random.uniform(-MAX_ROT, MAX_ROT)

        # build and simulate
        dlo = DloModel(dlo_params, position=init_shape, directors=init_directors)
        dlo.build_model(action=[action_1, action_2, action_3, action_4])
        dict_out = dlo.run_simulation(progress_bar=False)

        # save result
        mod_value = i % RESET_EVERY
        save_name = str(i).zfill(5) + "_" + str(mod_value).zfill(2) + ".pkl"
        pickle.dump(dict_out, open(os.path.join(SAVE_PATH, save_name), "wb"))

        # update continuation/reset
        if mod_value == (RESET_EVERY - 1):
            init_shape, init_directors = None, None
        else:
            init_shape = dict_out["final_shape"]
            init_directors = dict_out["final_directors"]

    return start_idx


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    start = time.time()

    n_workers = cpu_count()
    # define BLOCK_SIZE as global for run_block
    BLOCK_SIZE = int(np.ceil(ITERS / n_workers))
    tasks = range(0, ITERS, BLOCK_SIZE)

    print(f"Using {n_workers} workers, {len(tasks)} blocks of size {BLOCK_SIZE}, total_size = {n_workers * BLOCK_SIZE}")

    with Pool(n_workers) as pool:
        for _ in pool.imap_unordered(run_block, tasks):
            pass

    end = time.time()
    print(f"Total runtime: {end - start:.2f} seconds")
