import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import tripy

def read_stl(file_path):
    triangles = []
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(50)  # Read the header
            if len(data) == 0:
                break  # Reached the end of the file

            # Extract normal and vertices from the binary data
            normal = np.frombuffer(data[0:12], dtype=np.float32)
            vertex1 = np.frombuffer(data[12:24], dtype=np.float32)
            vertex2 = np.frombuffer(data[24:36], dtype=np.float32)
            vertex3 = np.frombuffer(data[36:48], dtype=np.float32)

            # Append the triangle vertices to the list
            triangles.extend([normal, vertex1, vertex2, vertex3])

            # Skip the attribute byte count (2 bytes) at the end of each triangle
            f.read(2)

    return np.array(triangles)

# Path to the STL file
stl_path = 'dev_scripts/voxelizer/squirtle.stl'

# Read the STL file
mesh_data = read_stl(stl_path)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to update the mesh plot for each frame of the animation
def update(frame):
    ax.view_init(elev=10, azim=frame)
    return mesh,

# Plot the STL mesh
mesh = Poly3DCollection([mesh_data.reshape(-1, 3)], alpha=0.1, edgecolor='k', facecolors='cyan')
ax.add_collection3d(mesh)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Create the animation
animation = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

# Uncomment the following line if you want to save the animation as a gif
# animation.save('stl_turntable_animation.gif', writer='imagemagick', fps=30)

plt.show()
