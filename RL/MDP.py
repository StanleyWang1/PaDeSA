import numpy as np

class MDP: 
    """
    Class MDP contains the MDP problem at hand
        contains state space (3D-numpy array)
        contains action space
        contains full rollout
    """
    def __init__(self, s_target, stop_criteria):
        """
        class initializier

        s_target: 3D binary numpy array of target part voxel
        discount_rate: learning discount rate

        self.TARGET: 3D numpy array containing target geometry
        self.STATE: 3D numpy array containing current geometry
        self.XLIM, YLIM, ZLIM: int dimensions of 3D numpy array
        self.XIND, YIND, ZIND: 3D numpy arrays containing indexes
        self.GAMMA: int discount rate of next action
        """
        self.TARGET = s_target
        self.STATE = np.zeros((s_target.shape)) # initialize empty occupancy grid
        self.XLIM, self.YLIM, self.ZLIM = s_target.shape
        self.XIND, self.YIND, self.ZIND = np.indices(s_target.shape)
        self.ALPHA = stop_criteria
        
    def reward(self, state):
        """
        calculates the reward associated with being the current state R(s)
        
        all elements where state == target == 1 are scored +1
        all elements where state == target == 0 are scored 0
        all elements where state != target are scored -1

        state: 3-D numpy array to be scored against self.TARGET

        returns: int score
        """
        matches = np.count_nonzero(np.logical_and(state, self.TARGET))
        mismatches = np.count_nonzero(state - self.TARGET)
        return matches - 10*mismatches

    def box(self, x, y, z, centroid):
        """
        create box action

        x: x-dimension (int)
        y: y-dimension (int)
        z: z-dimension (int)
        centroid: tuple of ints (x,y,z)

        returns: 3D binary numpy array of current state + new box
        DOES NOT UPDATE self.STATE!
        """
        x_o, y_o, z_o = centroid
        box = np.zeros((self.XLIM, self.YLIM, self.ZLIM))
        x_mask = (self.XIND >= x_o - x/2) & (self.XIND <= x_o + x/2)
        y_mask = (self.YIND >= y_o - y/2) & (self.YIND <= y_o + y/2)
        z_mask = (self.ZIND >= z_o - z/2) & (self.ZIND <= z_o + z/2)
        box_mask = x_mask & y_mask & z_mask
        box[box_mask] = 1
        return np.logical_or(self.STATE, box).astype(int)

    def sphere(self, r, centroid):
        """
        create sphere action

        r: radius (int)
        centroid: tuple (x, y, z)

        returns: 3D binary numpy array of current state + new sphere
        DOES NOT UPDATE self.STATE!
        """
        x_o, y_o, z_o = centroid
        sphere = np.zeros((self.XLIM, self.YLIM, self.ZLIM))
        sphere_mask = ((self.XIND - x_o)**2 + (self.YIND - y_o)**2 + (self.ZIND - z_o)**2) <= r**2
        sphere[sphere_mask] = 1
        return np.logical_or(self.STATE, sphere).astype(int)
        
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

        returns: 3D numpy array of current state + new cylinder. 
        DOES NOT UPDATE self.STATE!
        """
        x_o, y_o, z_o = centroid
        cylinder = np.zeros((self.XLIM, self.YLIM, self.ZLIM))
        if axis == 1:
            distance_to_axis = np.sqrt((self.YIND - y_o)**2 + (self.ZIND - z_o)**2)
            cylinder_mask = (distance_to_axis <= r) & (self.XIND >= x_o - h/2) & (self.XIND <= x_o + h/2)
            cylinder[cylinder_mask] = 1
        if axis == 2:
            distance_to_axis = np.sqrt((self.XIND - x_o)**2 + (self.ZIND - z_o)**2)
            cylinder_mask = (distance_to_axis <= r) & (self.YIND >= y_o - h/2) & (self.YIND <= y_o + h/2)
            cylinder[cylinder_mask] = 1
        if axis == 3:
            distance_to_axis = np.sqrt((self.XIND - x_o)**2 + (self.YIND - y_o)**2)
            cylinder_mask = (distance_to_axis <= r) & (self.ZIND >= z_o - h/2) & (self.ZIND <= z_o + h/2)
            cylinder[cylinder_mask] = 1
        return np.logical_or(self.STATE, cylinder).astype(int)
    
    def get_dim_lims_box(self, centroid):
        """
        calculates largest box that can be created from centroid that stays within voxel bounds
        
        centroid: 3-tuple xyz centroid

        returns: 3-tuple of x-limit, y-limit, z-limit
        """
        x_o, y_o, z_o = centroid
        xlim = min([self.XLIM - x_o, x_o])
        ylim = min([self.YLIM - y_o, y_o])
        zlim = min([self.ZLIM - z_o, z_o])
        return xlim * 2, ylim * 2, zlim * 2
        
    def get_dim_lims_cylinder(self, centroid, axis):
        """
        calculates the largest cylinder that can be created from centroid and with axis 
        that stays within voxel bounds

        centroid: 3-tuple xyz centroid

        returns: 2-tuple of r-limit, h-limit
        """
        x_o, y_o, z_o = centroid

        if axis == 1:
            rlim = min([self.ZLIM - z_o, z_o, self.YLIM - y_o, y_o])
            hlim = min([x_o, self.XLIM - x_o])
        if axis == 2: 
            rlim = min([self.XLIM - x_o, x_o, self.ZLIM - z_o, z_o])
            hlim = min([y_o, self.YLIM - y_o])
        if axis == 3:
            rlim = min([self.XLIM - x_o, x_o, self.YLIM - y_o, y_o])
            hlim = min([z_o, self.ZLIM - z_o])
        return rlim, hlim * 2
    
    def get_dim_lims_sphere(self, centroid):
        """
        calculates the largest sphere that can be created from centroid that stays within voxel bounds

        centroid: 3-tuple xyz centroid

        returns: int of r-limit
        """
        x_o, y_o, z_o = centroid
        return min([x_o, y_o, z_o, self.XLIM - x_o, self.YLIM - y_o, self.ZLIM - z_o])
    
    def optimized_rollout_box(self, centroid):
        xlim, ylim, zlim = self.get_dim_lims_box(centroid)
        xo, yo, zo = centroid
        x, y, z = (1, 1, 1)
        nextx, nexty, nextz = (1, 1, 1)
        converged = False
        max_r = float('-inf')
        while not converged:
            for i in np.arange(1, xlim): 
                next_state = self.box(i, y, z, centroid)
                r = self.reward(next_state)
                if r > max_r: 
                    nextx = i
                    max_r = r
            for j in np.arange(1, ylim):
                next_state = self.box(nextx, j, z, centroid)
                r = self.reward(next_state)
                if r > max_r:
                    nexty = j
                    max_r = r
            for k in np.arange(1, zlim):
                next_state = self.box(nextx, nexty, k, centroid)
                r = self.reward(next_state)
                if r > max_r:
                    nextz = k
                    max_r = r
            if [nextx, nexty, nextz] == [x, y, z]:
                converged = True
            x, y, z = (nextx, nexty, nextz)
        
        return {max_r: [1, xo, yo, zo, x, y, z]}
                
    def optimized_rollout_cylinder(self, centroid):
        best_outcome_by_axis = {}
        for ax in np.arange(3) + 1:
            radlim, hlim = self.get_dim_lims_cylinder(centroid, ax)
            xo, yo, zo = centroid
            rad, h = (1, 1)
            nextrad, nexth = (1, 1)
            max_r = float('-inf')
            converged = False
            while not converged:
                for radius in np.arange(1, radlim): 
                    next_state = self.cylinder(radius, h, ax, centroid)
                    r = self.reward(next_state)
                    if r > max_r: 
                        nextrad = radius
                        max_r = r
                for height in np.arange(1, hlim):
                    next_state = self.cylinder(nextrad, height, ax, centroid)
                    r = self.reward(next_state)
                    if r > max_r:
                        nexth = height
                        max_r = r
                if [nextrad, nexth] == [rad, h]:
                    converged = True
                rad, h = (nextrad, nexth)
            best_outcome_by_axis[max_r] = {max_r: [2, xo, yo, zo, rad, h, ax]}
        max_reward = max(best_outcome_by_axis.keys())
        return best_outcome_by_axis[max_reward]
            
        
    def optimized_rollout_sphere(self, centroid):
        rlim = self.get_dim_lims_sphere(centroid)
        xo, yo, zo  = centroid
        max_r = float('-inf')
        best_radius = 1
        for radius in np.arange(1, rlim):
            next_state = self.sphere(radius, centroid)
            r = self.reward(next_state)
            if r > max_r:
                best_radius = radius
                max_r = r
        return {max_r: [3, xo, yo, zo, best_radius]}
    
    def optimized_rollouts(self, centroid):
        rollout_action_rewards = {}
        rollout_action_rewards.update(self.optimized_rollout_box(centroid))
        rollout_action_rewards.update(self.optimized_rollout_cylinder(centroid))
        rollout_action_rewards.update(self.optimized_rollout_sphere(centroid))
        return rollout_action_rewards

    
    def stop_case(self):
        """
        determines whether the current state occupies at least a certain proportion (self.ALPHA)...
        of the target state. 
        when stop_case == True, one epoch of this MDP Learning problem is over

        TO-MODIFY IF INCLUDING SUBTRACTIVE ACTIONS

        returns: boolean
        """
        if (np.count_nonzero(np.logical_and(self.STATE, self.TARGET)) / np.count_nonzero(self.TARGET)) > self.ALPHA:
            return True
        else:
            return False
        
    """
    def full_rollout_box(self, centroid):
        xlim, ylim, zlim = self.get_dim_lims_box(centroid)
        xo, yo, zo  = centroid
        rewards = {}
        for i in np.arange(xlim):
            for j in np.arange(ylim):
                for k in np.arange(zlim):
                    next_state = self.box(i, j, k, centroid)
                    r = self.reward(next_state)
                    rewards[r] = [1, xo, yo, zo, i, j, k]
        return rewards
    """

    """
    def full_rollout(self, centroid):
        rollout_rewards = {}
        rollout_rewards.update(self.full_rollout_box(centroid))
        rollout_rewards.update(self.full_rollout_cylinder(centroid))
        rollout_rewards.update(self.full_rollout_sphere(centroid))
        return rollout_rewards
    """

    """
    def full_rollout_cylinder(self, centroid):
        rewards = {}
        xo, yo, zo  = centroid
        for axis in np.arange(3) + 1:
            rlim, hlim = self.get_dim_lims_cylinder(centroid, axis)
            for radius in np.arange(rlim):
                for h in np.arange(hlim):
                    next_state = self.cylinder(radius, h, axis, centroid)
                    r = self.reward(next_state)
                    rewards[r] = [2, xo, yo, zo, radius, h, axis]
        return rewards
    """