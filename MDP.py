import numpy as np
import cadquery as cq
import sys 

class MDP: 
    """
    Class MDP contains the MDP problem at hand

    self.TARGET: 3D numpy array of target part voxel
    
    """
    def __init__(self, s_target, discount_rate):
        """
        class initializier

        s_target: 3D numpy array of target part voxel
        discount_rate: learning discount rate
        """
        self.TARGET = s_target
        self.STATE = np.zeros((s_target.shape))
        self.GAMMA = discount_rate
        
    def reward(self):
        """
        calculates the reward associated with being the current state R(s)
        
        all elements where state == target == 1 are scored +1
        all elements where state == target == 0 are scored 0
        all elements where state != target are scored -1

        returns: int score
        """
        matches = np.inner(self.STATE, self.TARGET)
        mismatches = np.count_nonzero(self.STATE - self.TARGET)
        return matches - mismatches

    def box(self, x, y, z, centroid):
        """
        create box action.  Updates self.STATE

        x: x-dimension (int)
        y: y-dimension (int)
        z: z-dimension (int)
        centroid: tuple of ints (x,y,z)
        """
        x_o, y_o, z_o = centroid
        for i in np.arange(x):
            for j in np.arange(y):
                for k in np.arange(z): 
                    self.STATE[x_o + i, y_o + j, z_o + k] = 1

    def sphere(self, r, centroid):
        """
        create sphere action

        r: radius (int)
        centroid: tuple (x, y, z)
        """
        x_o, y_o, z_o = centroid
        return
        
    def cylinder(self, r, h, axis, centroid):
        """
        create cylinder action

        r: radius (int)
        h: height (int)
        axis: cartesian axis parallel to centerline of cylinder (int)
            1: x
            2: y
            3: z
        centroid(x, y, z)
        """
        x_o, y_o, z_o = centroid
        return
        
    