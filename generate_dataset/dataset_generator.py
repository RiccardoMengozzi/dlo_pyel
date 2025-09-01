from pyel_model.dlo_model import DloModel, DloModelParams
from pyel_model.plot import plot_interactive_2d, plot_observation_2d

import numpy as np
import datetime
import os
import pickle
from tqdm import tqdm

ITERS = 1_000_000
MAX_DISP = 0.075
MAX_ROT = np.pi / 4
RESET_EVERY = 20
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


init_shape = None
init_directors = None


os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Saving data to {SAVE_PATH}")

for i in tqdm(range(ITERS)):

    action_1 = np.random.randint(0, dlo_params.n_elem - 1)
    action_2 = np.random.uniform(-MAX_DISP, MAX_DISP)
    action_3 = np.random.uniform(-MAX_DISP, MAX_DISP)
    action_4 = np.random.uniform(-MAX_ROT, MAX_ROT)

    dlo = DloModel(dlo_params, position=init_shape, directors=init_directors)
    dlo.build_model(action=[action_1, action_2, action_3, action_4])
    dict_out = dlo.run_simulation(progress_bar=False)

    if False:
        callback_data = dlo.get_callback_data()

        plot_interactive_2d(callback_data)
        plot_observation_2d(dict_out)

    mod_value = i % RESET_EVERY

    #####################################
    save_name = str(i).zfill(5) + "_" + str(mod_value).zfill(2) + ".pkl"
    pickle.dump(dict_out, open(os.path.join(SAVE_PATH, save_name), "wb"))

    if mod_value == 9:
        init_shape = None
        init_directors = None
    else:
        init_shape = dict_out["final_shape"]
        init_directors = dict_out["final_directors"]
