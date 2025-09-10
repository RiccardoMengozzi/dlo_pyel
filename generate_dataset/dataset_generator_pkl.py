from pyel_model.dlo_model import DloModel, DloModelParams
from pyel_model.plot import plot_interactive_2d, plot_observation_2d

import numpy as np
import datetime
import os
import pickle
from tqdm import tqdm

ITERS = 1
MAX_DISP = 0.7
MAX_ROT = np.pi / 8
RESET_EVERY = 1
SAVE_PATH = f"dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"

CONFIG = {
    "dt": 1e-5,  # time step
    "nu": 1,  # damping coefficient for the simulation
    "node_gap": 0.01,  # number of elements
    "length": 0.5,  # length of the rod (m)
    "radius": 0.005,  # radius of the rod (m)
    "density": 1e3,  # density of the rod (kg/m^3)
    "youngs_modulus": 1e7,  # young's modulus of the rod (Pa)
    "action_velocity": 0.1,  # m/s (velocity of the action which influences the number of steps during simulation)
}

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

for i in tqdm(range(ITERS // RESET_EVERY)):

    data_to_save = {}

    # first iteration or every RESET_EVERY iterations we reset the shape
    init_shape = None
    init_directors = None

    for j in range(RESET_EVERY):

        action_1 = np.random.randint(0, dlo_params.n_elem - 1)
        action_2 = np.random.uniform(0.2, 0.21)
        action_3 = np.random.uniform(0.21, 0.22)
        action_4 = np.random.uniform(-MAX_ROT, MAX_ROT)

        dlo = DloModel(dlo_params, position=init_shape, directors=init_directors)
        dlo.build_model(action=[action_1, action_2, action_3, action_4])
        dict_out = dlo.run_simulation(progress_bar=False)

        init_shape = dict_out["final_shape"]
        init_directors = dict_out["final_directors"]

        data_to_save[j] = dict_out.copy()
        print(dict_out.keys())

    # save
    save_name = str(i).zfill(5) + ".pkl"
    pickle.dump(data_to_save, open(os.path.join(SAVE_PATH, save_name), "wb"))