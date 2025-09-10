import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
from tqdm import tqdm
import random
import glob, pickle

# DLO_DIFFUSION
from diffusers.schedulers import DDPMScheduler
from conditional_1d_unet import ConditionalUnet1D
from normalize import compute_cs0_csR, normalize_dlo, check_rot_and_flip
from normalize import denormalize_action_horizon, convert_action_horizon_to_absolute
from compute_directors import create_directors_from_positions

# Cosserat Model
from pyel_model.dlo_model import DloModel, DloModelParams

np.set_printoptions(precision=8, suppress=True, linewidth=100, threshold=1000)


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


def run_model_simulation(dlo_diff, dlo_params, init_shape, init_dir, target_shape, num_iterations=10, half_exec=True):
    """Run simulation for a single model and return all actions and shapes"""
    dlo_0 = init_shape.copy()
    dir_0 = init_dir.copy()
    
    errors = []
    all_actions = []
    all_shapes = []

    for i in range(num_iterations):
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

        errors.append(np.linalg.norm((target_shape - list_shapes[-1]), axis=0))
    
        list_shapes_ok = np.array([shape.T for shape in list_shapes])
        all_actions.append(pred_action)
        all_shapes.append(list_shapes_ok)

    errors = np.array(errors)
    avg_errors = np.mean(errors, axis=1)
    max_errors = np.max(errors, axis=1)
    all_actions = np.array(all_actions)
    all_shapes = np.array(all_shapes)

    result = np.stack([avg_errors, max_errors])
    
    return result, all_actions, all_shapes


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

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(MAIN_DIR, "dataset_evaluation")
    NUM_SAMPLES = 1
    NUM_ITERATIONS = 10
    PLOT = True
    
    # Define paths for both models


    CHECKPOINT_PATH_1 = os.path.join(MAIN_DIR, "checkpoints/diffusion_super-brook-8_best.pt")
    CHECKPOINT_PATH_2 = os.path.join(MAIN_DIR, "checkpoints/diffusion_sage-pine-63_best.pt")  # Replace with your second model path
    # Extract readable model names from checkpoint filenames
    model_name_1 = os.path.splitext(os.path.basename(CHECKPOINT_PATH_1))[0]
    model_name_2 = os.path.splitext(os.path.basename(CHECKPOINT_PATH_2))[0]
    model_name_1 = model_name_1.replace("diffusion_", "").replace("_best", "")
    model_name_2 = model_name_2.replace("diffusion_", "").replace("_best", "")

    # Load both models
    print("Loading Model 1...")
    model_1 = DiffusionInference(CHECKPOINT_PATH_1, device="cuda")
    
    print("Loading Model 2...")
    model_2 = DiffusionInference(CHECKPOINT_PATH_1, device="cuda")

    ################################

    # Initialize DLO model and get initial conditions
    action = [0, 0.0, 0.0, 0.0]
    dlo = DloModel(dlo_params)
    dlo.build_model(action=action)

    data_files = glob.glob(os.path.join(DATA_PATH, "*.pkl"))
    print("Found {} files in dataset {}".format(len(data_files), DATA_PATH))
    data_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[0]))

    data_shape = []
    data_dir = []
    for data_file in data_files:
        episode = pickle.load(open(data_file, "rb"))
        for action in episode.values():
            obs = action["obs"]
            obs_dir = action["obs_dir"]
            data_shape.append([obs])
            data_dir.append([obs_dir])


    print(f"Total actions: {len(data_shape)}")
    data_shape = np.array(data_shape).squeeze().reshape(-1, 3, 51)
    data_dir = np.array(data_dir).squeeze().reshape(-1, 3, 3, 50)
    print(f"Total data: {len(data_shape)}")

    data = [{"shape":s, "dir":d} for s, d in zip(data_shape, data_dir)]

    # create the test set: each file has init_shape, init_dir, final_shape, final_dir
    init_samples = random.choices(data, k=NUM_SAMPLES)
    target_samples = random.choices(data, k=NUM_SAMPLES)
    test_set = []
    for (init, target) in zip(init_samples, target_samples):
        test_set.append({"init": init, "target": target})

    # Run simulations
    results1 = []
    results2 = []
    
    all_actions_1 = []
    all_actions_2 = []
    all_shapes_1 = []
    all_shapes_2 = []

    for sample in tqdm(test_set, desc="Evaluating models"):
        result1, all_sample_actions_1, all_sample_shapes_1 = run_model_simulation(model_1, 
                                        dlo_params, 
                                        sample["init"]["shape"], 
                                        sample["init"]["dir"], 
                                        sample["target"]["shape"], 
                                        num_iterations=NUM_ITERATIONS, 
                                        half_exec=False)
        
        result2, all_sample_actions_2, all_sample_shapes_2 = run_model_simulation(model_2, 
                                                                                dlo_params, 
                                                                                sample["init"]["shape"], 
                                                                                sample["init"]["dir"], 
                                                                                sample["target"]["shape"], 
                                                                                num_iterations=NUM_ITERATIONS, 
                                                                                half_exec=False)
        results1.append(result1)
        results2.append(result2)

        all_actions_1.append(all_sample_actions_1)
        all_actions_2.append(all_sample_actions_2)
        all_shapes_1.append(all_sample_shapes_1)
        all_shapes_2.append(all_sample_shapes_2)

    results1 = np.array(results1)
    results2 = np.array(results2)

    all_actions_1 = np.array(all_actions_1)
    all_actions_2 = np.array(all_actions_2)
    all_shapes_1 = np.array(all_shapes_1)
    all_shapes_2 = np.array(all_shapes_2)

    avg_avg_max1 = np.mean(results1, axis=0)
    avg_avg_max2 = np.mean(results2, axis=0)



    # Extract average and max errors for both models
    avg_errors_model1 = avg_avg_max1[0]  # Average errors for model 1
    max_errors_model1 = avg_avg_max1[1]  # Max errors for model 1
    avg_errors_model2 = avg_avg_max2[0]  # Average errors for model 2
    max_errors_model2 = avg_avg_max2[1]  # Max errors for model 2

    # Create iteration numbers for x-axis
    iterations = np.arange(1, len(avg_errors_model1) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Average Errors Comparison
    ax1.plot(iterations, avg_errors_model1, 'b-o', label=f'Model 1 {model_name_1}', linewidth=2, markersize=6)
    ax1.plot(iterations, avg_errors_model2, 'r-s', label=f'Model 2 {model_name_2}', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Error')
    ax1.set_title('Average Error Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Use log scale if errors vary significantly

    # Plot 2: Max Errors Comparison
    ax2.plot(iterations, max_errors_model1, 'b-o', label=f'Model 1 {model_name_1}', linewidth=2, markersize=6)
    ax2.plot(iterations, max_errors_model2, 'r-s', label=f'Model 2 {model_name_2}', linewidth=2, markersize=6)
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
    print(f"Model 1 {model_name_1}:")
    print(f"  Average of Average Errors after {NUM_ITERATIONS} actions: {avg_errors_model1[-1]:.6f}")
    print(f"  Average of Max Errors after {NUM_ITERATIONS} actions: {max_errors_model1[-1]:.6f}")

    print(f"\nModel 2 {model_name_2}:")
    print(f"  Average of Average Errors after {NUM_ITERATIONS} actions: {avg_errors_model2[-1]:.6f}")
    print(f"  Average of Max Errors after {NUM_ITERATIONS} actions: {max_errors_model2[-1]:.6f}")

    # Determine which model performs better
    if avg_errors_model1[-1] < avg_errors_model2[-1]:
        print(f"\n✓ Model 1 performes {(avg_errors_model2[-1] / avg_errors_model1[-1] - 1)}% better on average error after {NUM_ITERATIONS} actions")
    else:
        print(f"\n✓ Model 2 performes {(avg_errors_model1[-1] / avg_errors_model2[-1] - 1)}% better on average error after {NUM_ITERATIONS} actions")

    if max_errors_model1[-1] < max_errors_model2[-1]:
        print(f"\n✓ Model 1 performes {(max_errors_model2[-1] / max_errors_model1[-1] - 1)}% better on max error after {NUM_ITERATIONS} actions")
    else:
        print(f"\n✓ Model 1 performes {(max_errors_model1[-1] / max_errors_model2[-1] - 1)}% better on max error after {NUM_ITERATIONS} actions")





    if PLOT:
        idx = 0

        init_shape = test_set[idx]["init"]["shape"]
        target_shape = test_set[idx]["target"]["shape"]
        all_shapes_1 = all_shapes_1[idx]
        all_actions_1 = all_actions_1[idx]
        all_shapes_2 = all_shapes_2[idx]
        all_actions_2 = all_actions_2[idx]
        init_shape_ok = init_shape.T
        target_shape_ok = target_shape.T


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
                plot_single_model(ax1, iteration, all_actions_1, all_shapes_1, init_shape_ok, target_shape_ok, "Model 1")
            else:
                ax1.clear()
                ax1.set_title(f"Model 1 - No data for iteration {iteration+1}")
            
            # Plot Model 2 (only if iteration is within range)
            if iteration < len(all_actions_2):
                plot_single_model(ax2, iteration, all_actions_2, all_shapes_2, init_shape_ok, target_shape_ok, "Model 2")
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