import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
from tqdm import tqdm

# DLO_DIFFUSION
from diffusers.schedulers import DDPMScheduler
from conditional_1d_unet import ConditionalUnet1D
from normalize import compute_cs0_csR, normalize_dlo, check_rot_and_flip
from normalize import denormalize_action_horizon, convert_action_horizon_to_absolute
from compute_directors import create_directors_from_positions

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

DLO_1_N_TEST =  np.array([[-0.24070267,  0.00292686],
        [-0.23132242, -0.00052248],
        [-0.22193218, -0.00398182],
        [-0.21255194, -0.00743116],
        [-0.20316169, -0.0108705 ],
        [-0.19375145, -0.01426983],
        [-0.18433122, -0.01762917],
        [-0.17489099, -0.02091851],
        [-0.16541076, -0.02410784],
        [-0.15589055, -0.02715717],
        [-0.14702029, -0.03084654],
        [-0.1373801 , -0.03350586],
        [-0.12681004, -0.03437512],
        [-0.11684998, -0.03519442],
        [-0.10684997, -0.03527371],
        [-0.09686001, -0.03472301],
        [-0.08692009, -0.03365231],
        [-0.0770302 , -0.03214161],
        [-0.06721033, -0.03027092],
        [-0.05744048, -0.02810023],
        [-0.04774065, -0.02569955],
        [-0.03807083, -0.02311887],
        [-0.02845102, -0.02039819],
        [-0.01886122, -0.01756751],
        [-0.00929143, -0.01466684],
        [ 0.00026837, -0.01172617],
        [ 0.00981816, -0.00875549],
        [ 0.01936795, -0.00578482],
        [ 0.02891774, -0.00282415],
        [ 0.03847753,  0.00010653],
        [ 0.04804733,  0.0030072 ],
        [ 0.05763713,  0.00584788],
        [ 0.06723693,  0.00864855],
        [ 0.07685674,  0.01137923],
        [ 0.08649655,  0.01404991],
        [ 0.09614637,  0.01666059],
        [ 0.10581619,  0.01920127],
        [ 0.11551601,  0.02167195],
        [ 0.12521584,  0.02408264],
        [ 0.13493568,  0.02642332],
        [ 0.14467552,  0.02870401],
        [ 0.15442536,  0.03092469],
        [ 0.16418521,  0.03309538],
        [ 0.17396506,  0.03520607],
        [ 0.18374492,  0.03727676],
        [ 0.19353477,  0.03930745],
        [ 0.20333463,  0.04130814],
        [ 0.21314449,  0.04327883],
        [ 0.22294435,  0.04522952],
        [ 0.23275422,  0.04717021],
        [ 0.24257408,  0.0491009 ]])




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


if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(MAIN_DIR, "DATA/train")
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "checkpoints/diffusion_super-brook-8_best.pt")


    dlo_diff = DiffusionInference(CHECKPOINT_PATH, device="cuda")


    ######### TEST #########

    init_shape = DLO_0_N_TEST
    target_shape = DLO_1_N_TEST

    ########################################

    print(init_shape.shape)
    print(target_shape.shape)
    # run diffusion to get the predicted action
    dlo_0_diff, dlo_1_diff, pred_action = dlo_diff.run(init_shape.T, target_shape.T)
    # convert the predicted action to the absolute frame for plotting
    act_global_pos, _ = convert_action_horizon_to_absolute(dlo_0_diff, pred_action)

    # Simple single frame plot
    plt.figure(figsize=(10, 8))
    
    
    # Plot initial and target shapes
    plt.plot(init_shape[:, 0], init_shape[:, 1], label="Initial Shape", marker="o", zorder=5)
    plt.plot(target_shape[:, 0], target_shape[:, 1], label="Target Shape", marker="x", zorder=4)
    
    # Plot predicted actions
    color_vals = np.linspace(0, 1, act_global_pos.shape[0])
    plt.scatter(
        act_global_pos[:, 0], act_global_pos[:, 1], c=color_vals, cmap="inferno", s=50, alpha=0.8, zorder=6
    )
    
    plt.title("Diffusion DLO Simulation")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.tight_layout()
    
    plt.show()