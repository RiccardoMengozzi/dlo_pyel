import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Director matrices (your data)
directors = [
    np.array([[-0.23572307,  0.97182026, -0.0000849 ],
              [ 0.00002779,  0.0000941,   1.        ],
              [ 0.97182027,  0.23572307, -0.00004919]]),
    
    np.array([[-0.23754207,  0.97137725, -0.00008568],
              [ 0.00002804,  0.00009507,  1.        ],
              [ 0.97137725,  0.23754207, -0.00004982]]),
    
    np.array([[-0.24080805,  0.97057276, -0.00008727],
              [ 0.00002932,  0.00009719,  0.99999999],
              [ 0.97057276,  0.24080804, -0.00005186]]),
    
    np.array([[-0.24511929,  0.96949292, -0.00008967],
              [ 0.00003139,  0.00010043,  0.99999999],
              [ 0.96949293,  0.24511929, -0.00005505]]),
    
    np.array([[-0.25004871,  0.96823325, -0.00009291],
              [ 0.00003302,  0.00010448,  0.99999999],
              [ 0.96823326,  0.25004871, -0.0000581 ]])
]

# Generate positions along the rod using the Z-direction vectors
# Start at origin and integrate along the Z-directions
positions = [np.array([0.0, 0.0, 0.0])]

# Step length between frames
step_length = 0.5

for i in range(len(directors) - 1):
    # Get the Z-direction (tangent) from current frame
    z_direction = directors[i][:, 2]  # Third column is Z-direction
    # Move along this direction to get next position
    next_pos = positions[-1] + step_length * z_direction
    positions.append(next_pos)

positions = np.array(positions)

# Create the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scale factor for the frame vectors
scale = 0.1

# Colors for the axes
colors = ['red', 'green', 'blue']
labels = ['X-axis', 'Y-axis', 'Z-axis']

# Plot each frame
for i, (pos, director) in enumerate(zip(positions, directors)):
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
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
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
all_coords = np.concatenate([positions.flatten(), 
                           (positions + scale * np.array([d[:, 0] for d in directors])).flatten(),
                           (positions + scale * np.array([d[:, 1] for d in directors])).flatten(),
                           (positions + scale * np.array([d[:, 2] for d in directors])).flatten()])

coord_min, coord_max = all_coords.min(), all_coords.max()
margin = (coord_max - coord_min) * 0.1  # Add 10% margin

ax.set_xlim(coord_min - margin, coord_max + margin)
ax.set_ylim(coord_min - margin, coord_max + margin)
ax.set_zlim(coord_min - margin, coord_max + margin)

# Show the plot
plt.tight_layout()
plt.show()

# Additional analysis: compute twist angles
print("Twist Analysis:")
print("-" * 40)

# Compute the twist angle between consecutive frames
for i in range(1, len(directors)):
    # Get the rotation matrix from frame i-1 to frame i
    R_prev = directors[i-1]
    R_curr = directors[i]
    
    # Relative rotation
    R_rel = R_curr @ R_prev.T
    
    # Extract twist angle around z-axis
    # For small rotations, the twist is approximately the (0,1) element of the relative rotation
    twist_angle = np.arctan2(R_rel[0, 1], R_rel[0, 0])
    twist_degrees = np.degrees(twist_angle)
    
    print(f"Frame {i} to {i+1}: Twist angle = {twist_degrees:.2f}°")

# Compute cumulative twist
cumulative_twist = 0
print(f"\nCumulative twist:")
print(f"Frame 1: 0.00°")

for i in range(1, len(directors)):
    R_prev = directors[i-1]
    R_curr = directors[i]
    R_rel = R_curr @ R_prev.T
    twist_angle = np.arctan2(R_rel[0, 1], R_rel[0, 0])
    cumulative_twist += np.degrees(twist_angle)
    print(f"Frame {i+1}: {cumulative_twist:.2f}°")