import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given positions (X, Y coordinates, assuming constant Z spacing)
xy_positions = np.array([
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



def create_directors_from_positions(positions):
    """Create director matrices from positions along the rod"""
    directors = []
    # Create 3D positions - all frames at same Z level (cable in XY plane)
    z_constant = 0.0  # All frames at same Z level
    xyz_positions = np.zeros((len(positions), 3))
    xyz_positions[:, :2] = positions  # X, Y from input
    xyz_positions[:, 2] = z_constant     # Constant Z for all frames

    for i in range(len(xyz_positions)):
        # Compute tangent vector (Z-direction) using finite differences
        if i == 0:
            # Forward difference for first point
            tangent = xyz_positions[i+1] - xyz_positions[i]
        elif i == len(xyz_positions) - 1:
            # Backward difference for last point
            tangent = xyz_positions[i] - xyz_positions[i-1]
        else:
            # Central difference for middle points
            tangent = (xyz_positions[i+1] - xyz_positions[i-1]) / 2.0
        
        # Normalize tangent to get Z-direction (along the curve)
        z_dir = tangent / np.linalg.norm(tangent)
        
        # For a planar curve, we can use the out-of-plane direction as reference
        # Since the curve is in XY plane, use [0, 0, 1] as reference for creating orthogonal frame
        out_of_plane = np.array([0, 0, 1])
        
        # Compute Y-direction as cross product of out-of-plane with Z-direction
        # This gives the normal direction in the plane of the curve
        y_dir = np.cross(out_of_plane, z_dir)
        y_dir = y_dir / np.linalg.norm(y_dir)
        
        # Compute X-direction as cross product of Y and Z
        x_dir = np.cross(y_dir, z_dir)
        
        # Create director matrix [X, Y, Z] as columns
        director = np.column_stack([x_dir, y_dir, z_dir])
        directors.append(director)
    
    return directors, xyz_positions


def main():

    # Create directors from positions
    directors, xyz_positions = create_directors_from_positions(xy_positions)


    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scale factor for the frame vectors
    scale = 0.1

    # Colors for the axes
    colors = ['red', 'green', 'blue']
    labels = ['X-axis', 'Y-axis', 'Z-axis']

    # Plot each frame
    for i, (pos, director) in enumerate(zip(xyz_positions, directors)):
        # Extract the three basis vectors (columns of the director matrix)
        x_vec = director[:, 0]  # X-direction
        y_vec = director[:, 1]  # Y-direction  
        z_vec = director[:, 2]  # Z-direction
        
        # Plot the frame origin
        ax.scatter(pos[0], pos[1], pos[2], color='black', s=50, alpha=0.7)
        
        # Plot the three basis vectors
        for j, (vec, color, label) in enumerate(zip([x_vec, y_vec, z_vec], colors, labels)):
            ax.quiver(pos[0], pos[1], pos[2], 
                    vec[0], vec[1], vec[2], 
                    color=color, arrow_length_ratio=0.1, 
                    length=scale, linewidth=2,
                    label=label if i == 0 else "")  # Only label once
        
        # Add frame number annotation
        ax.text(pos[0] + 0.05, pos[1] + 0.05, pos[2] + 0.05, f'{i+1}', fontsize=8)

    # Plot the rod centerline
    ax.plot(xyz_positions[:, 0], xyz_positions[:, 1], xyz_positions[:, 2], 
            'k--', alpha=0.5, linewidth=2, label='Rod centerline')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cosserat Rod Frame Directors\n(Showing twist along the rod)')

    # Add legend
    ax.legend()

    # Set equal aspect ratio and same range for all axes
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    ax.view_init(elev=20, azim=45)  # Better viewing angle

    # Get the range of all coordinates to set equal axis limits
    all_coords = np.concatenate([xyz_positions.flatten(), 
                            (xyz_positions + scale * np.array([d[:, 0] for d in directors])).flatten(),
                            (xyz_positions + scale * np.array([d[:, 1] for d in directors])).flatten(),
                            (xyz_positions + scale * np.array([d[:, 2] for d in directors])).flatten()])

    coord_min, coord_max = all_coords.min(), all_coords.max()
    margin = (coord_max - coord_min) * 0.1  # Add 10% margin

    ax.set_xlim(coord_min - margin, coord_max + margin)
    ax.set_ylim(coord_min - margin, coord_max + margin)
    ax.set_zlim(coord_min - margin, coord_max + margin)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Additional analysis: show some director information
    print("Director Analysis:")
    print("-" * 40)
    print(f"Number of frames: {len(directors)}")
    print(f"Rod length (arc length): {np.sum([np.linalg.norm(xyz_positions[i+1, :2] - xyz_positions[i, :2]) for i in range(len(xyz_positions)-1)]):.3f}")
    print(f"X position range: [{xyz_positions[:, 0].min():.3f}, {xyz_positions[:, 0].max():.3f}]")
    print(f"Y position range: [{xyz_positions[:, 1].min():.3f}, {xyz_positions[:, 1].max():.3f}]")

    # Show first few director matrices
    print(f"\nFirst 3 director matrices:")
    for i in range(min(3, len(directors))):
        print(f"\nFrame {i+1}:")
        print(directors[i])
        # Check orthogonality
        det = np.linalg.det(directors[i])
        print(f"Determinant: {det:.6f} (should be ~1.0)")
        
        # Check if columns are orthonormal
        dot_xy = np.dot(directors[i][:, 0], directors[i][:, 1])
        dot_xz = np.dot(directors[i][:, 0], directors[i][:, 2])
        dot_yz = np.dot(directors[i][:, 1], directors[i][:, 2])
        print(f"Orthogonality check - X·Y: {dot_xy:.6f}, X·Z: {dot_xz:.6f}, Y·Z: {dot_yz:.6f}")
        print(f"Unit vector check - |X|: {np.linalg.norm(directors[i][:, 0]):.6f}, |Y|: {np.linalg.norm(directors[i][:, 1]):.6f}, |Z|: {np.linalg.norm(directors[i][:, 2]):.6f}")



if __name__ == "__main__":
    main()