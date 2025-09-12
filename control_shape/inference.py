import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from conditional_1d_unet import ConditionalUnet1D
from diffusers.schedulers import DDPMScheduler

from normalize import (
    normalize_observation,
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
        for k in self.noise_scheduler.timesteps:
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

    def run(self, dlo_0, dlo_1):
        dlo_0 = dlo_0[:, :2]
        dlo_1 = dlo_1[:, :2]

        dlo_0_n, dlo_1_n, cs0, csR, rot_check_flag = normalize_observation(dlo_0, dlo_1)

        # prepare obs given pred_h_dim
        norm_obs = np.concatenate([dlo_0_n, dlo_1_n], axis=0)
        obs_horizon = norm_obs.reshape(1, norm_obs.shape[0] * 2)

        ###################################################################
        pred_actions = self.run_denoise_action(obs_horizon)

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
    DATA_PATH = os.path.join(MAIN_DIR, "dataset_linear/train")
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "checkpoints/diffusion_super-brook-8_best.pt")

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