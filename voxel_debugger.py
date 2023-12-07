import numpy as np
import matplotlib.pyplot as plt

def plot_3d_voxel(voxel_grid):
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
    plt.show()

generated_voxel = np.load('generated_sphere.npy')
generated_voxel = np.squeeze(generated_voxel)
generated_voxel = generated_voxel >= -0.5
generated_voxel = generated_voxel * 1
# print(generated_voxel)
plot_3d_voxel(generated_voxel)

# print(generated_voxel)
# print(np.amin(generated_voxel))
# print(np.amax(generated_voxel))
print(np.count_nonzero(generated_voxel))
# print(generated_voxel.shape)