import cadquery as cq
import numpy as np

class cq_executor:
    """
    class cq_executor executes the policy of pseudo-actions from class MDP
    """
    def __init__ (self):
        return

    def convert_to_axis_tuple(self, indicator):
        """
        converts an indicator for the center axis of a cylinder to a tuple that cadquery can read

        indicator: int (1, 2, or 3)

        returns: axis tuple
        """
        if indicator == 1:
            return (1, 0, 0)
        if indicator == 2: 
            return (0, 1, 0)
        if indicator == 3:
            return (0, 0, 1)

    def unpack_pseudoaction(self, action):
        """
        generates a cadquery action from a pseudoaction

        action: 1-D numpy array representing pseudoaction
        pseudoaction contains all the necessary information for a single cq action
        eg: [1 (int indicator for box), xo, yo, zo, x, y ,z]
        eg: [2 (int indicator for cylinder), xo, yo, zo, r, h, axis indictor (int)]
        eg: [3 (int indicator for sphere), xo, yo, zo, r]

        returns: cadquery action (saveable as a STEP file)
        """
        action_type = action[0] #read the type of action

        # read the centroid coordinates
        xo = action[1]
        yo = action[2]
        zo = action[3]

        # create a box
        if action_type == 1:
            x = action[4]
            y = action[5]
            z = action[6]
            cqaction = cq.Workplane(origin=(xo, yo, zo)).box(x, y, z, centered = (True, True, True))
        
        # create a cylinder
        if action_type == 2:
            r = action[4]
            h = action[5]
            axis_indc = action[6]
            if axis_indc == 1:
                xo = xo - h/2
            if axis_indc == 2: 
                yo = yo - h/2
            if axis_indc == 3:
                zo = zo - h/2
            axis = self.convert_to_axis_tuple(axis_indc)
            cqaction = cq.Workplane(origin = (xo, yo, zo)).cylinder(h, r, direct=axis, centered = (True, True, True))
        
        # create a sphere
        if action_type == 3:
            r = action[4]
            cqaction = cq.Workplane(origin= (xo, yo, zo)).sphere(r, centered= (True, True, True))
        return cqaction # return the cadquery action
        
    def execute_policy(self, policy, outfile):
        """
        Executes a policy of pseudo-actions by converting them to cadquery actions
        Saves resulting model to model.step file

        policy: a dictionary of pseudo-actions
            keys: int values indicating the order of the action
            values: 1-D numpy pseudo-actions (see documentation of get_cq_action for examples) 
        """
        MODEL = cq.Workplane()
        for key, pseudoaction in policy.items():
            next_cq_action = self.unpack_pseudoaction(pseudoaction)
            MODEL = MODEL | next_cq_action # update model with model produced by next action
        cq.exporters.export(MODEL, outfile + '_out.step') # save step file
