from pyel_model.dlo_model import DloModel, DloModelParams
from pyel_model.plot import plot_interactive_2d, plot_observation_2d

import numpy as np
import matplotlib.pyplot as plt


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

for i in range(10):

    action_1 = 0  # np.random.randint(0, dlo_params.n_elem - 1)
    action_2 = 0.0  # np.random.uniform(-.05, 0.05)
    action_3 = 0.01  # np.random.uniform(-0.05, 0.05)
    action_4 = 0.0

    dlo = DloModel(dlo_params, position=init_shape, directors=init_directors)
    dlo.build_model(action=[action_1, action_2, action_3, action_4])
    dict_out = dlo.run_simulation()

    callback_data = dlo.get_callback_data()

    if False:
        plot_interactive_2d(callback_data)

    else:

        plot_observation_2d(dict_out)

    # Save the data
    # with open("U_50cm.pkl", "wb") as f:
    #    pickle.dump(dict_out, f)

    init_directors = dict_out["final_directors"]
    init_shape = dict_out["final_shape"]
