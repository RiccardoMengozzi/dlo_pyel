import os, glob, pickle, random
from matplotlib import pyplot as plt
import numpy as np


def plot(init_shape, target_shape, intermediate_shapes, cmap_type="viridis", figsize=(10, 10)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cmap = plt.get_cmap(cmap_type, len(intermediate_shapes) + 2)

    ax.plot(init_shape[:, 0], init_shape[:, 1], '-o', label=f'Initial Shape', color=cmap(0), linewidth=2, markersize=6)
    for i, shape in enumerate(intermediate_shapes):
        ax.plot(shape[:, 0], shape[:, 1], '-o', label=f'Intermediate Shape {i+1}', color=cmap(i+1), linewidth=1, markersize=5)
    ax.plot(target_shape[:, 0], target_shape[:, 1], '-o', label=f'Target Shape', color=cmap(len(intermediate_shapes) + 1), linewidth=2, markersize=6)
        
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()
    plt.show()




def compute_lengths(intermediate_shapes, verbose=False):
    intermediate_lengths = []
    for shape in intermediate_shapes:
        length = np.sum([np.linalg.norm(shape[i, :2] - shape[i + 1, :2]) for i in range(shape.shape[0] - 1)])
        intermediate_lengths.append(length)
    intermediate_lengths = np.array(intermediate_lengths)

    if verbose:
        for i, length in enumerate(intermediate_lengths):
            print(f"Intermediate shape number {i+1} length: {length}")

    return intermediate_lengths

def get_intermediate_shapes(init_shape, target_shape, num_shapes, verbose=False):

    intermediate_shapes = []

    # Compute per-particle distance:
    distances = target_shape[:, :2] - init_shape[:, :2]
    for i in range(1, num_shapes + 1):
        intermediate_shapes.append(distances * i / (num_shapes + 1) + init_shape[:, :2])
    intermediate_shapes = np.array(intermediate_shapes)

    # Compute lengths
    dlo_0_length = np.sum([np.linalg.norm(init_shape[i, :2] - init_shape[i+1, :2]) for i in range(init_shape.shape[0] - 1)])
    dlo_1_length = np.sum([np.linalg.norm(target_shape[i, :2] - target_shape[i+1, :2]) for i in range(target_shape.shape[0] - 1)])

    # Lenght print
    intermediate_lengths = compute_lengths(intermediate_shapes, verbose=verbose)
    if verbose:
        print("dlo_0 total length = ", dlo_0_length)
        print("dlo_1 total length = ", dlo_1_length)


    ### Adjust Lengths ###
    adjusted_shapes = []

    for shape, current_length in zip(intermediate_shapes, intermediate_lengths):
        scale = dlo_0_length / current_length

        # Start from the first point
        adjusted_shape = [shape[0]]

        for i in range(1, shape.shape[0]):
            prev_point = adjusted_shape[-1]
            direction = shape[i] - shape[i - 1]
            scaled_segment = direction * scale
            new_point = prev_point + scaled_segment
            adjusted_shape.append(new_point)

        adjusted_shapes.append(np.array(adjusted_shape))

    adjusted_shapes = np.array(adjusted_shapes)

    # Lenght print
    intermediate_lengths = compute_lengths(adjusted_shapes, verbose=verbose)
    if verbose:
        print("dlo_0 total length = ", dlo_0_length)
        print("dlo_1 total length = ", dlo_1_length)

    return adjusted_shapes



def main():
    MAIN_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(MAIN_DIR, "dataset_evaluation")

    # Extract files
    data_files = glob.glob(os.path.join(DATA_PATH, "*.pkl"))
    data_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[0]))
    print("Found {} episode files in dataset {}".format(len(data_files), DATA_PATH))

    # Extracd data
    all_samples = []
    for data_file in data_files:
        episode = pickle.load(open(data_file, "rb"))
        for action_key, action_data in episode.items():
            all_samples.append(action_data)
    print("Number of samples: ", len(all_samples))

    while True:
        samples = random.choices(all_samples, k=2)
        dlo_0 = samples[0]["dlo_0"]
        dlo_1 = samples[1]["dlo_1"]
        NUM = 2

        intermediate_shapes = get_intermediate_shapes(dlo_0, dlo_1, NUM, verbose=True)
        plot(dlo_0, dlo_1, intermediate_shapes, cmap_type='viridis', figsize=(10, 10))        




if __name__ == "__main__":
    main()