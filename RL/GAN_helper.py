import numpy as np
import random
from matplotlib import pyplot as plt

def cylinder(dim, xind, yind, zind, r, h, axis, centroid):
        """
        create cylinder action

        r: radius (int)
        h: height (int)
        axis: cartesian axis parallel to centerline of cylinder (int)
            1: x
            2: y
            3: z
        centroid(x, y, z)

        returns: 3D numpy array of current state + new cylinder. 
        DOES NOT UPDATE self.STATE!
        """
        x_o, y_o, z_o = centroid
        cylinder = np.zeros((dim, dim, dim))
        if axis == 1:
            distance_to_axis = np.sqrt((yind - y_o)**2 + (zind - z_o)**2)
            cylinder_mask = (distance_to_axis <= r) & (xind >= x_o - h/2) & (xind <= x_o + h/2)
            cylinder[cylinder_mask] = 1
        if axis == 2:
            distance_to_axis = np.sqrt((xind - x_o)**2 + (zind - z_o)**2)
            cylinder_mask = (distance_to_axis <= r) & (yind >= y_o - h/2) & (yind <= y_o + h/2)
            cylinder[cylinder_mask] = 1
        if axis == 3:
            distance_to_axis = np.sqrt((xind - x_o)**2 + (yind - y_o)**2)
            cylinder_mask = (distance_to_axis <= r) & (zind >= z_o - h/2) & (zind <= z_o + h/2)
            cylinder[cylinder_mask] = 1
        return cylinder

def sphere(dim, xind, yind, zind, r, centroid):
    """
    create sphere action

    r: radius (int)
    centroid: tuple (x, y, z)
    """
    x_o, y_o, z_o = centroid
    sphere = np.zeros((dim, dim, dim))
    sphere_mask = ((xind - x_o)**2 + (yind - y_o)**2 + (zind - z_o)**2) <= r**2
    sphere[sphere_mask] = 1
    return sphere

if __name__ == "__main__":
    dim = 10
    state = np.zeros((dim, dim, dim))
    xind, yind, zind = np.indices(state.shape)
    
    for i in np.arange(50):
        r = int(random.uniform(1.9, 3.1))
        h = int(random.uniform(1, 9))
        ax = int(random.uniform(0.9, 3.1))
        centroid = (5, 5, 5)
        sample = cylinder(dim, xind, yind, zind, r, h, ax, centroid)
        np.save("cylinder_sample_" + str(i) + ".npy", sample)
        print("donezo")

    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = sample.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()
    
