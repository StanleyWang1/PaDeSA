import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from matplotlib import pyplot as plt
import sys

def main():
    filename = sys.argv[1]
    point_cloud = np.loadtxt(filename)
    new_df = pd.DataFrame(data=point_cloud, columns=['x','y','z'])
    cloud = PyntCloud(new_df)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=10, n_y=10, n_z=10)
    voxelgrid = cloud.structures[voxelgrid_id]
    Binary_voxel_array = voxelgrid.get_feature_vector(mode="binary")
    print(Binary_voxel_array)
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = Binary_voxel_array.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()

if __name__ == '__main__': 
    main()
