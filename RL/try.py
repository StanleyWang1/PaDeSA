import numpy as np
from scipy.ndimage import label

# Create a 2D or 3D occupancy grid (binary array)
occupancy_grid = np.load("thumb_screw.npy")
# Get the indices of occupied cells
occupied_indices = np.transpose(np.nonzero(occupancy_grid))

# Check if there are occupied cells
if len(occupied_indices) == 0:
    print("No occupied cells found.")
else:
    # Calculate the centroid as the mean of the occupied indices
    centroid = np.mean(occupied_indices, axis=0)

    print("Occupancy Grid:")
    print(occupancy_grid)
    print("\nCentroid of Occupied Cells:")
    print(centroid)

    labels_array, num_clusters = label(occupancy_grid)
    print("\n num Clusters: ")
    print(num_clusters)
