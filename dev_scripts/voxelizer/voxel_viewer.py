import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image

# Load PNG slices and convert to binary (0 and 1)
slices = [np.array(Image.open(f'./dev_scripts/voxelizer/sliced_squirtle/s_{i:02d}.png').convert('1')) for i in range(50)]

# Create a 3D numpy array (voxel grid)
voxel_grid = np.stack(slices, axis=2)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get indices of non-zero elements (where the voxel is present)
x, y, z = np.where(voxel_grid)

# Plot the voxels
scatter = ax.scatter(x, y, z, marker='o', s=10, c='#5a9ca4') #stanford red = '#8C1515'

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Function to update the scatter plot for each frame of the animation
def update(frame):
    ax.view_init(elev=10, azim=frame)
    return scatter,

# Create the animation
animation = FuncAnimation(fig, update, frames=np.arange(0, 360, 3), interval=30)

# Uncomment the following line if you want to save the animation as a gif
animation.save('./dev_scripts/voxelizer/voxelized_squirtle_animation.gif', writer='imagemagick', fps=30)

plt.show()
