import time
from multiprocessing import Pool, cpu_count, Manager, Value
from pyel_model.dlo_model import DloModel, DloModelParams
import numpy as np
import datetime
import os
import zarr
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
ITERS = 20
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
# SOLUTION 2: SHARED COUNTER
# -----------------------------

def init_worker():
    """Initialize each worker with unique random seed"""
    worker_id = os.getpid()
    np.random.seed(worker_id + int(time.time() * 1000) % 1000000)


def run_block_with_counter(args):
    """Run a block of simulations with shared progress counter"""
    start_idx, counter, lock, block_size = args
    init_shape, init_directors = None, None

    for i in range(start_idx, start_idx + block_size):
        if i >= ITERS:
            break

        # random action
        action_1 = np.random.randint(0, dlo_params.n_elem - 1)
        action_2 = np.random.uniform(-MAX_DISP, MAX_DISP)
        action_3 = np.random.uniform(-MAX_DISP, MAX_DISP)
        action_4 = np.random.uniform(-MAX_ROT, MAX_ROT)

        dlo = DloModel(dlo_params, position=init_shape, directors=init_directors)
        dlo.build_model(action=[action_1, action_2, action_3, action_4])
        dict_out = dlo.run_simulation(progress_bar=False)

        # save result in Zarr
        mod_value = i % RESET_EVERY
        save_name = str(i).zfill(5) + "_" + str(mod_value).zfill(2) + ".zarr"
        save_path = os.path.join(SAVE_PATH, save_name)

        root = zarr.open_group(save_path, mode="w")

        for key, value in dict_out.items():
            if isinstance(value, np.ndarray):
                root.create_dataset(
                    key,
                    data=value,
                    chunks=True,          # chunking automatico
                    compressor=zarr.get_codec({'id': 'zlib'})  # compressione
                )
            elif isinstance(value, (int, float, str, np.int32, np.float32)):
                root.attrs[key] = value
            else:
                root.attrs[key] = str(value)

        # update counter
        with lock:
            counter.value += 1

        # reset or continue
        if mod_value == (RESET_EVERY - 1):
            init_shape, init_directors = None, None
        else:
            init_shape = dict_out["final_shape"]
            init_directors = dict_out["final_directors"]

    return start_idx


def shared_counter():
    """Solution 2: Use shared counter for real-time progress updates"""
    print("Solution 2: Shared counter approach")
    
    start = time.time()
    n_workers = cpu_count()
    BLOCK_SIZE = int(np.ceil(ITERS / n_workers))
    print(f"Generating {ITERS} data with {n_workers} processes ({BLOCK_SIZE} data each) ")
    
    with Manager() as manager:
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        
        tasks = [(i, counter, lock, BLOCK_SIZE) 
                for i in range(0, ITERS, BLOCK_SIZE)]
        
        # Use initializer to set unique random seeds
        with Pool(n_workers, initializer=init_worker) as pool:

            # Start the workers
            result = pool.map_async(run_block_with_counter, tasks)
            
            # Monitor progress
            with tqdm(total=ITERS, desc="Simulations") as pbar:
                last_count = 0
                while not result.ready():
                    current_count = counter.value
                    if current_count > last_count:
                        pbar.update(current_count - last_count)
                        last_count = current_count
                    time.sleep(0.1)  # Check every 100ms
                
                # Final update
                final_count = counter.value
                if final_count > last_count:
                    pbar.update(final_count - last_count)
            
            result.get()  # Wait for completion
    
    end = time.time()
    print(f"Total runtime: {end - start:.2f} seconds")



# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    shared_counter()
