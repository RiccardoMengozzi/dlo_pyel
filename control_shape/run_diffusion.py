import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
from tqdm import tqdm

# DLO_DIFFUSION
from inference import DiffusionInference
from normalize import convert_action_horizon_to_absolute

# Cosserat Model
from pyel_model.dlo_model import DloModel, DloModelParams

np.set_printoptions(precision=8, suppress=True, linewidth=100, threshold=1000)


DLO_0_N_TEST = np.array([[-0.24576193,  0.00006841],
        [-0.23842283, -0.00002251],
        [-0.22844487,  0.00012381],
        [-0.21831221, -0.00013607],
        [-0.20829912,  0.00004509],
        [-0.19852597,  0.00002071],
        [-0.18863376, -0.00013362],
        [-0.17862239,  0.00013134],
        [-0.16849192, -0.00012976],
        [-0.1584807 ,  0.00004854],
        [-0.1487078 ,  0.00002425],
        [-0.13881573, -0.00013011],
        [-0.1288045 ,  0.00013485],
        [-0.11915079,  0.00005128],
        [-0.10937787,  0.00002708],
        [-0.09948576, -0.00012734],
        [-0.08947451,  0.00013762],
        [-0.07934402, -0.00012354],
        [-0.06933276,  0.00005479],
        [-0.05955984,  0.0000305 ],
        [-0.04966774, -0.00012383],
        [-0.03965648,  0.00014113],
        [-0.02952601, -0.00012003],
        [-0.01951475,  0.00005839],
        [-0.00974183,  0.0000341 ],
        [ 0.00015027, -0.00012026],
        [ 0.009804  , -0.0001172 ],
        [ 0.01981526,  0.00006107],
        [ 0.02958817,  0.00003687],
        [ 0.03948028, -0.00011755],
        [ 0.04949154,  0.00014741],
        [ 0.05962201, -0.00011369],
        [ 0.06963327,  0.00006461],
        [ 0.07940618,  0.00004038],
        [ 0.08929829, -0.00011398],
        [ 0.09930954,  0.00015092],
        [ 0.10944003, -0.00011024],
        [ 0.1194513 ,  0.00006809],
        [ 0.1286283 , -0.00011127],
        [ 0.13863954,  0.00015369],
        [ 0.14876996, -0.00010747],
        [ 0.15878121,  0.00007086],
        [ 0.16855409,  0.0000466 ],
        [ 0.17844616, -0.00010776],
        [ 0.18845755,  0.0001572 ],
        [ 0.19858809, -0.00010396],
        [ 0.20859942,  0.0000744 ],
        [ 0.21837217,  0.00004773],
        [ 0.22824936, -0.00011361],
        [ 0.23792662,  0.0001189 ],
        [ 0.24565348, -0.00008682]])

DLO_1_N_TEST = np.array([
    [0.5000227,  0.23180239 ],
    [0.49989584, 0.22447005 ],
    [0.49945357, 0.21452278 ],
    [0.49948218, 0.20441478 ],
    [0.49924994, 0.1944266  ],
    [0.49953243, 0.1846698  ],
    [0.5003945,  0.17481326 ],
    [0.50135416, 0.16482817 ],
    [0.5033843,  0.15486442 ],
    [0.5055199,  0.14501947 ],
    [0.50829244, 0.13555898 ],
    [0.5116568,  0.12616804 ],
    [0.51515365, 0.11670154 ],
    [0.51924855, 0.10788424 ],
    [0.5237646,  0.09917503 ],
    [0.5289718,  0.09073476 ],
    [0.5343901,  0.08229896 ],
    [0.54078937, 0.074453   ],
    [0.5472114,  0.06677794 ],
    [0.55377823, 0.05953822 ],
    [0.5603046,  0.05206416 ],
    [0.56542146, 0.0433929  ],
    [0.56787646, 0.03348375 ],
    [0.5666649,  0.02346172 ],
    [0.56244767, 0.0144706  ],
    [0.5570772,  0.00599453 ],
    [0.55133843, -0.00190075],
    [0.5453437,  -0.01004345],
    [0.53994703, -0.01828689],
    [0.5348691,  -0.02686904],
    [0.52967834, -0.03552297],
    [0.52534956, -0.04476995],
    [0.521081,   -0.05391448],
    [0.5174823,  -0.06308798],
    [0.5143985,  -0.07256885],
    [0.51124674, -0.08214363],
    [0.5089485,  -0.09205871],
    [0.50657266, -0.10181558],
    [0.5049495,  -0.11086131],
    [0.50301147, -0.12068724],
    [0.5019348,  -0.13076119],
    [0.50081307, -0.14071679],
    [0.50025153, -0.15049027],
    [0.5000738,  -0.16041125],
    [0.4996664,  -0.17045552],
    [0.49989936, -0.18063611],
    [0.4997488,  -0.19070452],
    [0.499814,   -0.2005332,],
    [0.50002533, -0.21045949],
    [0.49985456, -0.22016968],
    [0.5001113,  -0.2279048,],
])


DLO_2_N_TEST = np.array([[-0.00383,  0.04268],
 [ 0.00555,  0.04613],
 [ 0.01494,  0.04959],
 [ 0.02432,  0.05304],
 [ 0.03371,  0.05648],
 [ 0.04312,  0.05988],
 [ 0.05254,  0.06324],
 [ 0.06198,  0.06653],
 [ 0.07146,  0.06972],
 [ 0.08098,  0.07277],
 [ 0.08985,  0.07646],
 [ 0.09949,  0.07912],
 [ 0.11006,  0.07999],
 [ 0.12002,  0.08081],
 [ 0.13002,  0.08089],
 [ 0.14001,  0.08034],
 [ 0.14995,  0.07927],
 [ 0.15984,  0.07776],
 [ 0.16966,  0.07589],
 [ 0.17943,  0.07372],
 [ 0.18913,  0.07132],
 [ 0.1988 ,  0.06874],
 [ 0.20842,  0.06602],
 [ 0.21801,  0.06319],
 [ 0.22758,  0.06029],
 [ 0.23714,  0.05735],
 [ 0.24669,  0.05438],
 [ 0.25624,  0.05141],
 [ 0.26579,  0.04845],
 [ 0.27535,  0.04552],
 [ 0.28492,  0.04262],
 [ 0.29451,  0.03978],
 [ 0.30411,  0.03698],
 [ 0.31373,  0.03425],
 [ 0.32337,  0.03158],
 [ 0.33302,  0.02897],
 [ 0.34269,  0.02643],
 [ 0.35239,  0.02396],
 [ 0.36209,  0.02155],
 [ 0.37181,  0.01921],
 [ 0.38155,  0.01693],
 [ 0.3913 ,  0.01471],
 [ 0.40106,  0.01254],
 [ 0.41084,  0.01043],
 [ 0.42062,  0.00836],
 [ 0.43041,  0.00633],
 [ 0.44021,  0.00433],
 [ 0.45002,  0.00236],
 [ 0.45982,  0.00041],
 [ 0.46963, -0.00153],
 [ 0.47945, -0.00346]])



def run_model_simulation(dlo_diff, dlo_params, init_shape, init_dir, target_shape, num_iterations=10, half_exec=True):
    """Run simulation for a single model and return all actions and shapes"""
    dlo_0 = init_shape.copy().T
    dir_0 = init_dir.copy()
    
    all_actions = []
    all_shapes = []

    for i in tqdm(range(num_iterations), desc="performing actions"):

        # run diffusion to get the predicted action
        dlo_0_n, dlo_1_n, pred_action = dlo_diff.run(dlo_0.T, target_shape)

        list_shapes, list_directors = [], []
        pred_action = pred_action[:int(pred_action.shape[0] // 2 + 1), :] if half_exec else pred_action
        for h in range(pred_action.shape[0]):
            dlo = DloModel(dlo_params, position=dlo_0, directors=dir_0)
            dlo.build_model(action=pred_action[h])
            dict_out = dlo.run_simulation(progress_bar=False)
            list_shapes.append(dict_out["final_shape"])
            list_directors.append(dict_out["final_directors"])

            # update dlo_0 and dir_0 for the next iteration
            dlo_0 = list_shapes[-1]
            dir_0 = list_directors[-1]

        list_shapes_ok = np.array([shape.T for shape in list_shapes])
        all_actions.append(pred_action)
        all_shapes.append(list_shapes_ok)

    all_actions = np.array(all_actions)
    all_shapes = np.array(all_shapes)
    
    return all_actions, all_shapes


def plot_single_model(ax, iteration, all_actions, all_shapes, init_shape_ok, target_shape_ok, model_name):
    """Plot results for a single model on given axis"""
    ax.clear()
    
    # Get current action and shapes
    action = all_actions[iteration]
    simulation_shapes = all_shapes[iteration]
    
    # Plot initial and target shapes
    ax.plot(init_shape_ok[:, 0], init_shape_ok[:, 1], label="Initial Shape", marker="o", zorder=5)
    ax.plot(target_shape_ok[:, 0], target_shape_ok[:, 1], label="Target Shape", marker="x", zorder=4)
    
    # Plot all shapes from the current simulation run
    for j, shape in enumerate(simulation_shapes):
        ax.plot(shape[:, 0], shape[:, 1], color="black", linestyle="--", alpha=0.5, zorder=1)
    
    # Highlight the first and final shapes of the current run
    ax.plot(simulation_shapes[0][:, 0], simulation_shapes[0][:, 1], label="First Shape", marker="o", color="blue", zorder=2)
    ax.plot(simulation_shapes[-1][:, 0], simulation_shapes[-1][:, 1], label="Final Shape", marker="o", color="red", zorder=3)

    # Get the predicted action for this iteration
    pred_action_iter = action
    act_global_pos, _ = convert_action_horizon_to_absolute(simulation_shapes[0][:, :2], pred_action_iter)

    color_vals = np.linspace(0, 1, act_global_pos.shape[0])
    scatter = ax.scatter(
        act_global_pos[:, 0], act_global_pos[:, 1], c=color_vals, cmap="inferno", s=50, alpha=0.8, zorder=6
    )
    
    ax.set_title(f"{model_name} - Iteration {iteration+1}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid()
    ax.axis("equal")


if __name__ == "__main__":

    CONFIG = {
        "dt": 1e-5,  # time step
        "nu": 1,  # damping coefficient for the simulation
        "node_gap": 0.01,  # number of elements
        "length": 0.5,  # length of the rod (m)
        "radius": 0.005,  # radius of the rod (m)
        "density": 1e3,  # density of the rod (kg/m^3)
        "youngs_modulus": 1e5,  # young's modulus of the rod (Pa)
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

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(MAIN_DIR, "dataset_evaluation")
    
    # Define paths for both models
    CHECKPOINT_PATH_1 = os.path.join(MAIN_DIR, "checkpoints/diffusion_super-brook-8_best.pt")
    CHECKPOINT_PATH_2 = os.path.join(MAIN_DIR, "checkpoints/diffusion_sage-pine-63_best.pt")  # Replace with your second model path
    
    # Load both models
    print("Loading Model 1...")
    model_1 = DiffusionInference(CHECKPOINT_PATH_1, device="cuda")
    
    print("Loading Model 2...")
    model_2 = DiffusionInference(CHECKPOINT_PATH_2, device="cuda")


    # Initialize DLO model
    import pickle, glob, random
    action = [0, 0.0, 0.0, 0.0]
    dlo = DloModel(dlo_params)
    dlo.build_model(action=action)

    # Load all samples (each action from each episode is a separate sample)
    data_files = glob.glob(os.path.join(DATA_PATH, "*.pkl"))
    print("Found {} episode files in dataset {}".format(len(data_files), DATA_PATH))
    data_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[0]))

    all_samples = []
    for data_file in data_files:
        episode = pickle.load(open(data_file, "rb"))
        # Each action in the episode is a separate sample
        for action_key, action_data in episode.items():
            all_samples.append(action_data)

    init_sample = random.choice(all_samples)
    final_sample = random.choice(all_samples)

    init_shape = init_sample["dlo_0"]
    init_dir = init_sample["dir_0"]
    target_shape = final_sample["dlo_1"]

    ########################################

    # Run simulations for both models
    print("Running simulation for Model 1...")
    all_actions_1, all_shapes_1 = run_model_simulation(model_1, dlo_params, init_shape, init_dir, target_shape, num_iterations=1, half_exec=False)
    
    print("Running simulation for Model 2...")
    all_actions_2, all_shapes_2 = run_model_simulation(model_2, dlo_params, init_shape, init_dir, target_shape, num_iterations=1, half_exec=False)

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(bottom=0.15)  # Make room for the slider

    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    max_iterations = max(len(all_actions_1), len(all_actions_2)) - 1
    slider = Slider(ax_slider, 'Iteration', 0, max_iterations, valinit=0, valstep=1, valfmt='%d')

    def update_plot(iteration):
        """Update both plots based on the slider value"""
        iteration = int(iteration)
        
        # Plot Model 1 (only if iteration is within range)
        if iteration < len(all_actions_1):
            plot_single_model(ax1, iteration, all_actions_1, all_shapes_1, init_shape, target_shape, "Model 1")
        else:
            ax1.clear()
            ax1.set_title(f"Model 1 - No data for iteration {iteration+1}")
        
        # Plot Model 2 (only if iteration is within range)
        if iteration < len(all_actions_2):
            plot_single_model(ax2, iteration, all_actions_2, all_shapes_2, init_shape, target_shape, "Model 2")
        else:
            ax2.clear()
            ax2.set_title(f"Model 2 - No data for iteration {iteration+1}")
        
        plt.tight_layout()
        fig.canvas.draw()

    # Connect the slider to the update function
    slider.on_changed(update_plot)

    # Initialize the plot with the first iteration
    update_plot(0)

    # Show the plot
    plt.show()

