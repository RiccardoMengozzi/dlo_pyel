import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from conditional_1d_unet import ConditionalUnet1D
from diffusers.schedulers import DDPMScheduler

from normalize import (
    normalize,
    denormalize_dlo,
    denormalize_action_horizon,
    convert_action_horizon_to_absolute,
)
from dataset import load_sample

import pickle

from matplotlib.widgets import Slider


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

    def run_denoise_action(self, obs_horizon):

        norm_obs_tensor = torch.from_numpy(obs_horizon.copy()).float().unsqueeze(0).to(self.device)

        obs_cond = norm_obs_tensor.flatten(start_dim=1).to(self.device)

        # initialize action from Gaussian noise
        naction = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)

        # init scheduler
        self.noise_scheduler.set_timesteps(self.noise_steps)

        list_actions = []
        for k in tqdm(self.noise_scheduler.timesteps):
            # predict noise
            noise_pred = self.model(sample=naction, timestep=k, global_cond=obs_cond)

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            list_actions.append(naction.squeeze().detach().cpu().numpy())

        return np.stack(list_actions, axis=0)

    def denormalize_action_absolute(self, dlo, action_horizon, cs0, csR, rot_check_flag=False):
        act = denormalize_action_horizon(
            dlo,
            action_horizon,
            cs0,
            csR,
            disp_scale=self.disp_scale,
            angle_scale=self.angle_scale,
            rot_check_flag=rot_check_flag,
        )

        act_pos, act_rot = convert_action_horizon_to_absolute(dlo, act)

        return act_pos, act_rot

    def from_observation_to_dlos(self, obs, cs0, csR, rot_check_flag=False):
        obs_last = obs[-1, :].reshape(-1)
        dlo_0_n = obs_last[: self.num_points * 2].reshape(self.num_points, 2)
        dlo_1_n = obs_last[self.num_points * 2 :].reshape(self.num_points, 2)

        dlo_0_ok = denormalize_dlo(dlo_0_n, cs0, csR, rot_check_flag)
        dlo_1_ok = denormalize_dlo(dlo_1_n, cs0, csR, rot_check_flag)

        return dlo_0_ok, dlo_1_ok

    def run(self, dlo_0, dlo_1, act):
        dlo_0_n, dlo_1_n, action_n, cs0, csR, rot_check_flag = normalize(
            dlo_0, dlo_1, act, disp_scale=self.disp_scale, angle_scale=self.angle_scale
        )

        # prepare action given pred_h_dim
        idx = np.linspace(0, action_n.shape[0] - 1, self.pred_horizon, dtype=int)
        action_horizon = action_n[idx, :]

        gt_action_disp = action_horizon[:, 1:3] * self.disp_scale
        gt_action_rot = action_horizon[:, 3] * self.angle_scale

        # prepare obs given pred_h_dim
        norm_obs = np.concatenate([dlo_0_n, dlo_1_n], axis=0)
        obs_horizon = norm_obs.reshape(1, norm_obs.shape[0] * 2)

        print("before sampling: ", obs_horizon.shape, action_horizon.shape)

        ###################################################################
        pred_actions = self.run_denoise_action(obs_horizon)

        pred_action = pred_actions[-1]  # take the last action from the denoised actions

        pred_action_disp = pred_action[:, 1:3] * self.disp_scale
        pred_action_rot = pred_action[:, 3] * self.angle_scale

        list_pred_action_pos = []
        list_pred_action_idx = []
        for i in range(pred_actions.shape[0]):
            pred_idx = int(pred_actions[i][0, 0] * (self.num_points - 2))
            pred_idx = np.clip(pred_idx, 0, self.num_points - 2)
            list_pred_action_idx.append(pred_idx)

            pred_action_pos = pred_actions[i][:, 1:3] * self.disp_scale + dlo_0_n[pred_idx]
            list_pred_action_pos.append(pred_action_pos)

        if True:
            from matplotlib.widgets import Slider

            # Plot initialization
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.subplots_adjust(bottom=0.25)

            # Plot the static DLO curves
            (dlo0_line,) = ax.plot(dlo_0_n[:, 0], dlo_0_n[:, 1], label="dlo_0", color="blue")
            (dlo1_line,) = ax.plot(dlo_1_n[:, 0], dlo_1_n[:, 1], label="dlo_1", color="orange")

            # Initial scatter plot
            initial_i = 0
            try:
                action_pos = list_pred_action_pos[initial_i]
                color_vals = np.linspace(0, 1, action_pos.shape[0])
                scatter = ax.scatter(
                    action_pos[:, 0], action_pos[:, 1],
                    c=color_vals, cmap="inferno", s=50, alpha=0.8
                )
            except Exception as e:
                print(f"Error denormalizing action {initial_i}: {e}")
                scatter = ax.scatter([], [], c=[], cmap="inferno", s=50, alpha=0.8)

            # Add slider
            ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
            slider = Slider(ax_slider, "Action Index", 0, len(pred_actions) - 1,
                            valinit=initial_i, valstep=1)

            # Update function
            def update(val):
                i = int(slider.val)
                try:
                    action_pos = list_pred_action_pos[i]
                    scatter.set_offsets(action_pos[:, :2])  # x, y positions
                except Exception as e:
                    print(f"Error denormalizing action {i}: {e}")
                    scatter.set_offsets([])

                fig.canvas.draw_idle()

            slider.on_changed(update)

            # Final touches
            ax.axis("equal")
            ax.set_xlim(0, 0.6)     # <-- Fix x range
            ax.set_ylim(-0.3, 0.3)  # <-- Fix y range
            ax.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(MAIN_DIR, "dataset_spline/val")
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "checkpoints/diffusion_proud-eon-67_best.pt")

    dlo_diff = DiffusionInference(CHECKPOINT_PATH, device="cuda")

    ################################

    files = os.listdir(DATA_PATH)
    files = [os.path.join(DATA_PATH, f) for f in files if f.endswith(".pkl")]
    print(f"Found {len(files)} files in {DATA_PATH}")

    # shuffle files
    np.random.shuffle(files)

    for file in files:

        file_path = os.path.join(DATA_PATH, file)

        data = pickle.load(open(file_path, "rb"))
        for step, sample in data.items():

            # LOAD
            dlo_0, dlo_1, act = load_sample(sample)
            dlo_diff.run(dlo_0, dlo_1, act)