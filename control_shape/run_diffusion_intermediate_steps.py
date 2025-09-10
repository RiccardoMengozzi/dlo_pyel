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


def run_standard_technique(dlo_diff, dlo_params, sample_data, num_iterations=10, half_exec=True):
    """Standard technique: Direct path from initial to target shape"""
    init_shape = sample_data["dlo_0"].copy().T
    init_dir = sample_data["dir_0"].copy()
    target_shape = sample_data["dlo_1"].T



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


def run_sequential_technique(dlo_diff, dlo_params, sample_data, num_intermediate_targets=3, num_iterations_per_target=10, half_exec=True):
    """Sequential technique: Using intermediate shapes as stepping stones"""
    
    init_shape = sample_data["dlo_0"].copy().T
    init_dir = sample_data["dir_0"].copy()
    target_shape = sample_data["dlo_1"].T
    intermediate_shapes = sample_data["obs"]  # Shape: (num_intermediate_steps, 3, 51)

    # evenly spaced indices
    step = intermediate_shapes.shape[0] / num_intermediate_targets
    idxs = [int(step/2 + step*i) for i in range(num_intermediate_targets)]


    # shift to center if only one index
    if num_intermediate_targets == 1:
        idxs = [intermediate_shapes.shape[0] // 2]  # middle element

    intermediate_shapes = intermediate_shapes[idxs]

    dlo_0 = init_shape.copy()
    dir_0 = init_dir.copy()
    
    all_errors = []
    all_actions_sequence = []
    all_shapes_sequence = []
    
    # Go through each intermediate shape as a target
    for step in range(intermediate_shapes.shape[0]):
        current_target = intermediate_shapes[step]  # Current intermediate target
        
        step_errors = []
        step_actions = []
        step_shapes = []
        
        for i in range(num_iterations_per_target):
            # run diffusion to get the predicted action towards current intermediate target
            dlo_0_diff, dlo_1_diff, pred_action = dlo_diff.run(dlo_0, current_target)

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

            # Calculate error with respect to the final target (not intermediate)
            error_to_final = np.linalg.norm((target_shape - list_shapes[-1]), axis=0)
            step_errors.append(error_to_final)
        
            list_shapes_ok = np.array([shape.T for shape in list_shapes])
            step_actions.append(pred_action)
            step_shapes.append(list_shapes_ok)

        all_errors.extend(step_errors)
        all_actions_sequence.extend(step_actions)
        all_shapes_sequence.extend(step_shapes)

    errors = np.array(all_errors)
    avg_errors = np.mean(errors.reshape(-1, errors.shape[-1]), axis=1) if len(errors.shape) > 1 else errors
    max_errors = np.max(errors.reshape(-1, errors.shape[-1]), axis=1) if len(errors.shape) > 1 else errors
    all_actions_sequence = np.array(all_actions_sequence)
    all_shapes_sequence = np.array(all_shapes_sequence)

    result = np.stack([avg_errors, max_errors])
    
    return result, all_actions_sequence, all_shapes_sequence


def plot_technique_comparison(ax, iteration, all_actions_std, all_shapes_std, all_actions_seq, all_shapes_seq, 
                            init_shape_ok, target_shape_ok, technique_names):
    """Plot results for both techniques on given axis"""
    ax.clear()
    
    # Plot initial and target shapes
    ax.plot(init_shape_ok[:, 0], init_shape_ok[:, 1], label="Initial Shape", marker="o", color="green", markersize=8, zorder=5)
    ax.plot(target_shape_ok[:, 0], target_shape_ok[:, 1], label="Target Shape", marker="x", color="red", markersize=10, zorder=4)
    
    # Plot standard technique (if iteration is within range)
    if iteration < len(all_actions_std):
        action_std = all_actions_std[iteration]
        simulation_shapes_std = all_shapes_std[iteration]
        
        # Plot all shapes from standard technique
        for j, shape in enumerate(simulation_shapes_std):
            ax.plot(shape[:, 0], shape[:, 1], color="blue", linestyle="--", alpha=0.3, zorder=1)
        
        # Highlight the final shape of standard technique
        ax.plot(simulation_shapes_std[-1][:, 0], simulation_shapes_std[-1][:, 1], 
               label=f"{technique_names[0]} Final", color="blue", linewidth=2, zorder=3)
        
        # Plot predicted actions for standard technique
        act_global_pos_std, _ = convert_action_horizon_to_absolute(simulation_shapes_std[0][:, :2], action_std)
        ax.scatter(act_global_pos_std[:, 0], act_global_pos_std[:, 1], 
                  c='blue', marker='o', s=30, alpha=0.6, label=f"{technique_names[0]} Actions", zorder=6)
    
    # Plot sequential technique (if iteration is within range)
    if iteration < len(all_actions_seq):
        action_seq = all_actions_seq[iteration]
        simulation_shapes_seq = all_shapes_seq[iteration]
        
        # Plot all shapes from sequential technique
        for j, shape in enumerate(simulation_shapes_seq):
            ax.plot(shape[:, 0], shape[:, 1], color="orange", linestyle=":", alpha=0.3, zorder=1)
        
        # Highlight the final shape of sequential technique
        ax.plot(simulation_shapes_seq[-1][:, 0], simulation_shapes_seq[-1][:, 1], 
               label=f"{technique_names[1]} Final", color="orange", linewidth=2, zorder=3)
        
        # Plot predicted actions for sequential technique
        act_global_pos_seq, _ = convert_action_horizon_to_absolute(simulation_shapes_seq[0][:, :2], action_seq)
        ax.scatter(act_global_pos_seq[:, 0], act_global_pos_seq[:, 1], 
                  c='orange', marker='^', s=30, alpha=0.6, label=f"{technique_names[1]} Actions", zorder=6)
    
    ax.set_title(f"Technique Comparison - Step {iteration+1}")
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
    NUM_SAMPLES = 10
    NUM_ITERATIONS_STANDARD = 10  # Iterations for standard technique
    NUM_ITERATIONS_PER_TARGET = 1  # Iterations per intermediate target in sequential technique
    NUM_INTERMEDIATE_TARGETS = 3
    PLOT = True
    
    # Define path for the model
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "checkpoints/diffusion_super-brook-8_best.pt")
    
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

    all_samples = all_samples[:NUM_SAMPLES]

    print(f"Total samples (actions): {len(all_samples)}")

    # Run comparisons on all samples
    results_standard = []
    results_sequential = []
    
    all_actions_standard = []
    all_shapes_standard = []
    all_actions_sequential = []
    all_shapes_sequential = []

    for sample_idx, sample in enumerate(tqdm(all_samples, desc="Evaluating techniques")):
        
        # Run standard technique: initial → target directly
        result_std, actions_std, shapes_std = run_standard_technique(
            model, dlo_params, sample, 
            num_iterations=NUM_ITERATIONS_STANDARD, half_exec=False
        )
        
        # Run sequential technique: initial → intermediate1 → intermediate2 → ... → target
        result_seq, actions_seq, shapes_seq = run_sequential_technique(
            model, dlo_params, sample, 
            num_intermediate_targets=NUM_INTERMEDIATE_TARGETS,
            num_iterations_per_target=NUM_ITERATIONS_PER_TARGET, half_exec=False
        )
        
        results_standard.append(result_std)
        results_sequential.append(result_seq)
        
        all_actions_standard.append(actions_std)
        all_shapes_standard.append(shapes_std)
        all_actions_sequential.append(actions_seq)
        all_shapes_sequential.append(shapes_seq)

    # Convert to numpy arrays
    results_standard = np.array(results_standard)
    results_sequential = np.array(results_sequential)

    # Calculate average performance across all samples
    avg_results_standard = np.mean(results_standard, axis=0)
    avg_results_sequential = np.mean(results_sequential, axis=0)

    # Extract errors
    avg_errors_standard = avg_results_standard[0]
    max_errors_standard = avg_results_standard[1]
    avg_errors_sequential = avg_results_sequential[0]  
    max_errors_sequential = avg_results_sequential[1]

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Ensure both arrays have the same length for comparison
    max_length = max(len(avg_errors_standard), len(avg_errors_sequential))
    iterations = np.arange(1, max_length + 1)

    # Pad shorter arrays with their last values if needed
    if len(avg_errors_standard) < max_length:
        avg_errors_standard = np.pad(avg_errors_standard, (0, max_length - len(avg_errors_standard)), 
                                   mode='edge')
        max_errors_standard = np.pad(max_errors_standard, (0, max_length - len(max_errors_standard)), 
                                   mode='edge')
    
    if len(avg_errors_sequential) < max_length:
        avg_errors_sequential = np.pad(avg_errors_sequential, (0, max_length - len(avg_errors_sequential)), 
                                     mode='edge')
        max_errors_sequential = np.pad(max_errors_sequential, (0, max_length - len(max_errors_sequential)), 
                                     mode='edge')

    # Plot 1: Average Errors Comparison
    ax1.plot(iterations[:len(avg_errors_standard)], avg_errors_standard, 'b-o', label='Standard Technique', 
             linewidth=2, markersize=6)
    ax1.plot(iterations[:len(avg_errors_sequential)], avg_errors_sequential, 'r-s', label='Sequential Technique', 
             linewidth=2, markersize=6)
    ax1.set_xlabel('Step/Iteration')
    ax1.set_ylabel('Average Error')
    ax1.set_title('Average Error Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Max Errors Comparison
    ax2.plot(iterations[:len(max_errors_standard)], max_errors_standard, 'b-o', label='Standard Technique', 
             linewidth=2, markersize=6)
    ax2.plot(iterations[:len(max_errors_sequential)], max_errors_sequential, 'r-s', label='Sequential Technique', 
             linewidth=2, markersize=6)
    ax2.set_xlabel('Step/Iteration')
    ax2.set_ylabel('Maximum Error')
    ax2.set_title('Maximum Error Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== Technique Performance Summary ===")
    print(f"Evaluated on {len(all_samples)} samples from {len(data_files)} episodes")
    
    print(f"\nStandard Technique (Direct Initial → Target):")
    print(f"  Final Average Error: {avg_errors_standard[-1]:.6f}")
    print(f"  Final Max Error: {max_errors_standard[-1]:.6f}")

    print(f"\nSequential Technique (Initial → Intermediates → Target):")
    print(f"  Final Average Error: {avg_errors_sequential[-1]:.6f}")
    print(f"  Final Max Error: {max_errors_sequential[-1]:.6f}")

    # Determine which technique performs better
    if avg_errors_standard[-1] < avg_errors_sequential[-1]:
        improvement = (avg_errors_sequential[-1] / avg_errors_standard[-1] - 1) * 100
        print(f"\n✓ Standard Technique performs {improvement:.1f}% better on average error")
    else:
        improvement = (avg_errors_standard[-1] / avg_errors_sequential[-1] - 1) * 100
        print(f"\n✓ Sequential Technique performs {improvement:.1f}% better on average error")

    if max_errors_standard[-1] < max_errors_sequential[-1]:
        improvement = (max_errors_sequential[-1] / max_errors_standard[-1] - 1) * 100
        print(f"✓ Standard Technique performs {improvement:.1f}% better on max error")
    else:
        improvement = (max_errors_standard[-1] / max_errors_sequential[-1] - 1) * 100
        print(f"✓ Sequential Technique performs {improvement:.1f}% better on max error")

    # Interactive visualization of a single sample
    if PLOT and len(all_samples) > 0:
        sample_idx = 0
        sample = all_samples[sample_idx]
        
        init_shape = sample["dlo_0"]
        target_shape = sample["dlo_1"]
        
        init_shape_ok = init_shape.T
        target_shape_ok = target_shape.T
        
        sample_actions_std = all_actions_standard[sample_idx]
        sample_shapes_std = all_shapes_standard[sample_idx]
        sample_actions_seq = all_actions_sequential[sample_idx]
        sample_shapes_seq = all_shapes_sequential[sample_idx]

        # Create the figure with subplot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15)

        # Create slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        max_iterations = max(len(sample_actions_std), len(sample_actions_seq)) - 1
        slider = Slider(ax_slider, 'Step', 0, max_iterations, valinit=0, valstep=1, valfmt='%d')

        def update_plot(iteration):
            """Update plot based on the slider value"""
            iteration = int(iteration)
            plot_technique_comparison(ax, iteration, sample_actions_std, sample_shapes_std,
                                    sample_actions_seq, sample_shapes_seq, init_shape_ok, target_shape_ok,
                                    ["Standard", "Sequential"])
            fig.canvas.draw()

        # Connect the slider to the update function
        slider.on_changed(update_plot)

        # Initialize the plot
        update_plot(0)

        print(f"\nShowing interactive visualization for sample {sample_idx}")
        print(f"Number of intermediate shapes in this sample: {sample['obs'].shape[0]}")

        # Show the plot
        plt.show()