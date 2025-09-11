import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
from tqdm import tqdm
import pickle, glob

# DLO_DIFFUSION
from diffusers.schedulers import DDPMScheduler
from conditional_1d_unet import ConditionalUnet1D
from normalize import compute_cs0_csR, normalize_dlo, check_rot_and_flip
from normalize import denormalize_action_horizon, convert_action_horizon_to_absolute
from compute_directors import create_directors_from_positions
from intermediate_targets_generator import get_intermediate_shapes

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


class DiffusionInference:
    def __init__(self, checkpoint_path, device="cpu"):

        state = torch.load(checkpoint_path)

        self.disp_scale = state["scale_disp"]
        self.angle_scale = state["scale_rot"]
        self.num_points = state["num_points"]

        self.pred_horizon = state["pred_h_dim"]
        self.obs_dim = state["obs_dim"]
        self.obs_h_dim = state["obs_h_dim"]
        self.action_dim = state["action_dim"]
        self.device = device

        self.noise_steps = 100

        # Build diffusion components
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.model = ConditionalUnet1D(
            input_dim=state["action_dim"], global_cond_dim=state["obs_dim"] * state["obs_h_dim"]
        )
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()

    def run_denoise_action(self, obs_horizon, progress_bar=True):

        norm_obs_tensor = torch.from_numpy(obs_horizon.copy()).float().unsqueeze(0).to(self.device)

        obs_cond = norm_obs_tensor.flatten(start_dim=1).to(self.device)

        # initialize action from Gaussian noise
        naction = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)

        # init scheduler
        self.noise_scheduler.set_timesteps(self.noise_steps)

        list_actions = []
        for k in tqdm(self.noise_scheduler.timesteps, disable=not progress_bar):
            # predict noise
            noise_pred = self.model(sample=naction, timestep=k, global_cond=obs_cond)

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            list_actions.append(naction.squeeze().detach().cpu().numpy())

        return np.stack(list_actions, axis=0)

    def normalize_observation(self, dlo_0, dlo_1):
        # compute normalization factors
        cs0, csR = compute_cs0_csR(dlo_0)

        # dlo shape
        dlo_0_n = normalize_dlo(dlo_0, cs0, csR)
        dlo_1_n = normalize_dlo(dlo_1, cs0, csR)

        # check rot
        dlo_0_n, dlo_1_n, rot_check_flag = check_rot_and_flip(dlo_0_n, dlo_1_n)

        return dlo_0_n, dlo_1_n, cs0, csR, rot_check_flag

    def prepare_obs_action(obs_n, dlo_1_n, action_n):

        action_steps = action_n[1:] / obs_n.shape[1]
        actions_idx = np.tile(action_n[0], (obs_n.shape[0], 1))
        actions = np.tile(action_steps, (obs_n.shape[0], 1))
        actions = np.concatenate([actions_idx, actions], axis=1)

        obs_target = np.tile(dlo_1_n, (obs_n.shape[0], 1, 1))
        norm_obs = np.concatenate([obs_n, obs_target], axis=1)

        # flatten the observation
        norm_obs = norm_obs.reshape(norm_obs.shape[0], -1)

        return norm_obs, actions

    def run(self, dlo_0, dlo_1):

        #
        dlo_0 = dlo_0.T
        dlo_1 = dlo_1.T
        dlo_0 = dlo_0[:, :2]  # take only x and y coordinates
        dlo_1 = dlo_1[:, :2]  # take only x and y coordinates

        # Normalize observation
        dlo_0_n, dlo_1_n, cs0, csR, rot_check_flag = self.normalize_observation(dlo_0, dlo_1)

        obs_0 = np.tile(dlo_0_n, (self.obs_h_dim, 1, 1))  # repeat for the obs horizon
        obs_target = np.tile(dlo_1_n, (self.obs_h_dim, 1, 1))
        norm_obs = np.concatenate([obs_0, obs_target], axis=1)

        ###################################################################
        pred_actions = self.run_denoise_action(norm_obs, progress_bar=False)
        pred_action = pred_actions[-1]  # take the last action from the denoised actions

        ############################

        act = denormalize_action_horizon(
            dlo_0,
            pred_action,
            cs0,
            csR,
            disp_scale=self.disp_scale,
            angle_scale=self.angle_scale,
            rot_check_flag=rot_check_flag,
        )

        return dlo_0, dlo_1, act


def run(dlo_diff, dlo_params, dlo_0, dir_0, dlo_1, num_iterations=10, half_exec=True):
    """Run simulation for a single model and return all actions and shapes"""
    dlo_0 = dlo_0.T
    dir_0 = dir_0
    target_shape = dlo_1.T
    
    all_actions = []
    all_shapes = []
    errors = []

    for i in range(num_iterations):

        # run diffusion to get the predicted action
        dlo_0_diff, dlo_1_diff, pred_action = dlo_diff.run(dlo_0, target_shape)

        # convert the predicted action to the absolute frame for plotting
        act_global_pos, _ = convert_action_horizon_to_absolute(dlo_0_diff, pred_action)

        # apply the predicted action to the DLO model
        action_idx = pred_action[0, 0]

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

        errors.append(np.linalg.norm((target_shape - list_shapes[-1]), axis=0))


        list_shapes_ok = np.array([shape.T for shape in list_shapes])
        all_actions.append(pred_action)
        all_shapes.append(list_shapes_ok)



    errors = np.array(errors)
    avg_errors = np.mean(errors, axis=1)
    max_errors = np.max(errors, axis=1)
    result = np.stack([avg_errors, max_errors])

    all_actions = np.array(all_actions)
    all_shapes = np.array(all_shapes)
    
    return result, all_actions, all_shapes


def run_intermediate(dlo_diff, 
                     dlo_params, 
                     dlo_0, 
                     dir_0, 
                     dlo_1, 
                     intermediate_targets, 
                     num_intermediate_targets=2, 
                     num_iterations=5, 
                     half_exec=True,
                     use_dataset=False):
    """Run simulation for a single model and return all actions and shapes"""
    dlo_0 = dlo_0.T
    dir_0 = dir_0
    target_shape = dlo_1.T
    
    if use_dataset:
        # evenly spaced indices
        step = intermediate_targets.shape[0] / num_intermediate_targets
        idxs = [int(step/2 + step*i) for i in range(num_intermediate_targets)]

        # shift to center if only one index
        if num_intermediate_targets == 1:
            idxs = [intermediate_targets.shape[0] // 2]  # middle element

        intermediate_targets = intermediate_targets[idxs]
    else:
        intermediate_targets = get_intermediate_shapes(dlo_0.T, target_shape.T, num_intermediate_targets)
        intermediate_targets = np.moveaxis(intermediate_targets, 1, 2)



    target_shape_expanded = target_shape[np.newaxis, :, :]  # shape (1, 3, 51)
    # Concatenate along axis 0
    intermediate_targets = np.concatenate((intermediate_targets, target_shape_expanded), axis=0)


    all_actions = []
    all_shapes = []
    errors = [] 
    for i in range(num_iterations):
        if i < intermediate_targets.shape[0]:
            target_shape = intermediate_targets[i]


        # run diffusion to get the predicted action
        dlo_0_diff, dlo_1_diff, pred_action = dlo_diff.run(dlo_0, target_shape)

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

        errors.append(np.linalg.norm((intermediate_targets[-1] - list_shapes[-1]), axis=0))

        list_shapes_ok = np.array([shape.T for shape in list_shapes])
        all_actions.append(pred_action)
        all_shapes.append(list_shapes_ok)

    errors = np.array(errors)
    avg_errors = np.mean(errors, axis=1)
    max_errors = np.max(errors, axis=1)
    result = np.stack([avg_errors, max_errors])

    all_actions = np.array(all_actions)
    all_shapes = np.array(all_shapes)
    
    return result, all_actions, all_shapes, intermediate_targets



def plot_single_model(ax, iteration, all_actions, all_shapes, init_shape_ok, target_shape_ok, model_name, intermediate_shapes=None):
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

    if intermediate_shapes is not None:
        intermediate_shape = intermediate_shapes[iteration].T
        ax.plot(intermediate_shape[:, 0], intermediate_shape[:, 1], color="green", linestyle="--", alpha=0.5, zorder=1)
    
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
    NUM_SAMPLES = 5
    NUM_ITERATIONS = 10  # Iterations for standard technique
    NUM_INTERMEDIATE_TARGETS = 3
    
    # Define path for the model
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "checkpoints/diffusion_super-brook-8_best.pt")
    model_name = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]
    model_name = model_name.replace("diffusion_", "").replace("_best", "")
    # Load the model
    print("Loading Model...")
    model = DiffusionInference(CHECKPOINT_PATH, device="cuda")

    ################################

    # Initialize DLO model
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


    print(f"Total samples (actions): {len(all_samples)}")

    import random
    all_samples = random.choices(all_samples, k=NUM_SAMPLES)

    # Run simulations for both models
    results_std = []
    results_int = []
    for sample in tqdm(all_samples, desc="Running evaluation"):
        result_std, all_actions_1, all_shapes_1 = run(model, 
                                        dlo_params, 
                                        sample["dlo_0"], 
                                        sample["dir_0"], 
                                        sample["dlo_1"], 
                                        num_iterations=NUM_ITERATIONS, 
                                        half_exec=False)
        
        result_int, all_actions_2, all_shapes_2, intermediate_targets = run_intermediate(model, 
                                                                            dlo_params, 
                                                                            sample["dlo_0"], 
                                                                            sample["dir_0"], 
                                                                            sample["dlo_1"], 
                                                                            sample["obs"],
                                                                            num_iterations=NUM_ITERATIONS,
                                                                            num_intermediate_targets=NUM_INTERMEDIATE_TARGETS,
                                                                            half_exec=False)

        results_std.append(result_std)
        results_int.append(result_int)

    results_std = np.array(results_std)
    results_int = np.array(results_int)

    avg_avg_max_std = np.mean(results_std, axis=0)
    avg_avg_max_max = np.mean(results_int, axis=0)



    # Extract average and max errors for both models
    avg_errors_model_std = avg_avg_max_std[0]  # Average errors for model 1
    max_errors_model_std = avg_avg_max_std[1]  # Max errors for model 1
    avg_errors_model_int = avg_avg_max_max[0]  # Average errors for model 2
    max_errors_model_int = avg_avg_max_max[1]  # Max errors for model 2

    # Create iteration numbers for x-axis
    iterations = np.arange(1, len(avg_errors_model_std) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Average Errors Comparison
    ax1.plot(iterations, avg_errors_model_std, 'b-o', label=f'Model: {model_name}, Standard', linewidth=2, markersize=6)
    ax1.plot(iterations, avg_errors_model_int, 'r-s', label=f'Model: {model_name}, Intermediate actions', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Error')
    ax1.set_title('Average Error Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Use log scale if errors vary significantly

    # Plot 2: Max Errors Comparison
    ax2.plot(iterations, max_errors_model_std, 'b-o', label=f'Model: {model_name}, Standard', linewidth=2, markersize=6)
    ax2.plot(iterations, max_errors_model_int, 'r-s', label=f'Model: {model_name}, Intermediate actions', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Maximum Error')
    ax2.set_title('Maximum Error Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Use log scale if errors vary significantly

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== Model Performance Summary ===")
    print(f"Model: {model_name}, Standard:")
    print(f"  Average of Average Errors after {NUM_ITERATIONS} actions: {avg_errors_model_std[-1]:.6f}")
    print(f"  Average of Max Errors after {NUM_ITERATIONS} actions: {max_errors_model_std[-1]:.6f}")

    print(f"\nModel: {model_name}, Intermediate Actions:")
    print(f"  Average of Average Errors after {NUM_ITERATIONS} actions: {avg_errors_model_int[-1]:.6f}")
    print(f"  Average of Max Errors after {NUM_ITERATIONS} actions: {max_errors_model_int[-1]:.6f}")

    # Determine which model performs better
    if avg_errors_model_std[-1] < avg_errors_model_int[-1]:
        value = ((avg_errors_model_int[-1] - avg_errors_model_std[-1]) / avg_errors_model_std[-1] * 100)
        print(f"\n✓ Standard performes {value:.2f}% better on average error after {NUM_ITERATIONS} actions")
    else:
        value = ((avg_errors_model_std[-1] - avg_errors_model_int[-1]) / avg_errors_model_int[-1] * 100)
        print(f"\n✓ Intermediate actions performes {value:.2f}% better on average error after {NUM_ITERATIONS} actions")

    if max_errors_model_std[-1] < max_errors_model_int[-1]:
        value = ((max_errors_model_int[-1] - max_errors_model_std[-1]) / max_errors_model_std[-1] * 100)
        print(f"\n✓ Standard performes {value:.2f}% better on max error after {NUM_ITERATIONS} actions")
    else:
        value = ((max_errors_model_std[-1] - max_errors_model_int[-1]) / max_errors_model_int[-1] * 100)
        print(f"\n✓ Intermediate actions performes {value:.2f}% better on max error after {NUM_ITERATIONS} actions")
