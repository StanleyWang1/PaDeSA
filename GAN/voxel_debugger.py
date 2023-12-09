import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def histogram_3d_array(input_array):
    plt.hist(input_array.flatten(), bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Values in a 3D Numpy Array')
    plt.show()

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
    plt.show(block=True)

def plot_vox(voxel_grid, threshold):
    voxel_grid = voxel_grid.squeeze()  # Remove axes of length one

    x, y, z = np.indices(np.array(voxel_grid.shape) + 1)
    voxels = voxel_grid > threshold  # Adjust threshold as needed

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(x, y, z, voxels, edgecolor='k')
    plt.show()

generated_voxel = np.load('./LOGS/ap01_epoch1000.npy')
generated_voxel = np.squeeze(generated_voxel) * 1e5
histogram_3d_array(generated_voxel)
# generated_voxel = generated_voxel < 0.001
plot_vox(generated_voxel, threshold=0.5)

# input()

# generated_voxel = generated_voxel < 0.001
# print(np.count_nonzero(generated_voxel))
# print(np.count_nonzero(~generated_voxel))

# generated_voxel = np.load('./DATA/cylinder_samples/cylinder_sample_33.npy')
# plot_vox(generated_voxel, threshold=0)

# print(generated_voxel)
# print(np.amin(generated_voxel))
# print(np.amax(generated_voxel))
# print(np.count_nonzero(generated_voxel))
# print(generated_voxel.shape)