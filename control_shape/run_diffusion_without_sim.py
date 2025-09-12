import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
from tqdm import tqdm
import pickle, glob, random

# DLO_DIFFUSION
from inference import DiffusionInference
from normalize import compute_cs0_csR, normalize_dlo, check_rot_and_flip
from normalize import denormalize_action_horizon, convert_action_horizon_to_absolute
from compute_directors import create_directors_from_positions


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



if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(MAIN_DIR, "dataset_linear/train")
    CHECKPOINT_PATH = os.path.join(MAIN_DIR, "checkpoints/diffusion_sage-pine-63_best.pt")


    dlo_diff = DiffusionInference(CHECKPOINT_PATH, device="cuda")


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


    while True:
        sample = random.choice(all_samples)
        init_shape = sample["dlo_0"]
        target_shape = sample["dlo_1"]
        

        # run diffusion to get the predicted action
        dlo_0, dlo_1, pred_action = dlo_diff.run(init_shape, target_shape)
        # convert the predicted action to the absolute frame for plotting

        act_global_pos, _ = convert_action_horizon_to_absolute(dlo_0, pred_action)

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