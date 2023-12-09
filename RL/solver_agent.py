import numpy as np
import random
from scipy.ndimage import label
from MDP import MDP
from cq_executor import cq_executor

TARGET = np.zeros(1) # target voxel (to be set by user input)

"""
TRAINING HYPERPARAMETERS: do with it as you please :)
    GAMMA:  discount factor of future action rewards
        Large values of GAMMA encourage more actions in policy
        Small values of GAMMA encourage less actions in policy
        Recommend keep small.
    ALPHA: stop condition for a single RL epoch.  
        Part is considered complete when ALPHA proportion of mass is filled.  
        Recommend keep close to 1. 
    BETA: convergence criteria for a policy 
        If < BETA percent change in consecutive policy utilities, optimal policy is assumed reached
        Recommend keep close to 0 
    N: number of new random centroids to explore via random exploration strategy. 
"""
GAMMA = 0.99 # (0 < GAMMA < 1)
ALPHA = 0.95 # (0 < ALPHA < 1)
BETA = 0.01 # (0 < BETA < 1)

def update_MDP_state(action, mdp):
    action_type = action[0]
    centroid = (action[1], action[2], action[3])
    if action_type == 1:
        x, y, z = action[4], action[5], action[6]
        next_state = mdp.box(x,y,z, centroid)
        mdp.STATE = next_state
    if action_type == 2:
        r, h, ax = action[4], action[5], action[6]
        next_state = mdp.cylinder(r,h,ax,centroid)
        mdp.STATE = next_state
    if action_type == 3:
        r = action[4]
        next_state = mdp.sphere(r, centroid)
        mdp.STATE = next_state

def utility(policy):
    mirror_dim_mdp = MDP(TARGET, ALPHA)
    sum_discounted_rewards = 0
    action_number_tracker = 0
    for key, action in policy.items():
        update_MDP_state(action, mirror_dim_mdp)
        sum_discounted_rewards += GAMMA**action_number_tracker * key
        action_number_tracker += 1
    return sum_discounted_rewards

def converged(old_policy, new_policy):
    new_policy_utility = utility(new_policy) 
    old_policy_utility = utility(old_policy)
    if old_policy_utility == 0: return False # this indicates an old policy is a blank policy
    print("percent change in policy utilities = " + str((new_policy_utility - old_policy_utility) / old_policy_utility))
    if abs((new_policy_utility - old_policy_utility) / old_policy_utility) < BETA:
        return True
    else: 
        return False
    
def hooke_jeeves(action, reward, voxel):
    step = 8 # HYPERPARAMETER
    X, Y, Z = voxel.shape
    curr_best_reward = reward
    curr_best_action = action
    better_neighbor = False
    xo, yo, zo = (action[1], action[2], action[3])
    while step >= 1 and (better_neighbor == False): 
        list_of_neighbors = np.array([[xo+step,yo,zo],[xo-step,yo,zo],[xo,yo+step,zo],[xo,yo-step,zo],[xo,yo,zo+step],[xo,yo,zo-step]])
        for neighbor in list_of_neighbors:
            xn, yn, zn = (int(neighbor[0]), int(neighbor[1]), int(neighbor[2]))

            if (xn < 0 or yn < 0 or zn < 0):
                is_valid_neighbor = False
            elif(xn > X-1 or yn > Y-1 or zn > Z-1): 
                is_valid_neighbor = False
            else:
                is_valid_neighbor = voxel[xn, yn, zn] == 1

            if is_valid_neighbor:
                rollouts = mdp.optimized_rollouts((xn, yn , zn))
                max_reward = max(rollouts.keys())
                best_action = rollouts[max_reward]
                if max_reward > curr_best_reward:
                    curr_best_reward = max_reward
                    curr_best_action = best_action
                    better_neighbor = True
        step = step/2        
    return curr_best_action, curr_best_reward
   

def clustered_explore(voxel):
    labels, num_clusters = label(voxel)
    largest_cluster_size = 0
    best_cluster_label = 0
    for i in np.arange(num_clusters):
        size = np.count_nonzero(labels == i + 1)
        if size > largest_cluster_size:
            largest_cluster_size = size
            best_cluster_label = i + 1

    cluster = labels == best_cluster_label
    occ_idx = np.transpose(np.nonzero(cluster))
    centroid = np.round(np.mean(occ_idx, axis = 0))
    x = int(centroid[0])
    y = int(centroid[1])
    z = int(centroid[2])
    if voxel[x, y, z] == 0:
        return random_explore(voxel)
    return tuple(centroid)

def random_explore(voxel):
    occ_x, occ_y, occ_z = np.nonzero(voxel)
    num_occ_indx = len(occ_x)

    idx = random.randint(0, num_occ_indx - 1)
    centroid = np.array([occ_x[idx], occ_y[idx], occ_z[idx]])
    return tuple(centroid)

def learn(mdp):
    num_epochs = 0
    policy_converged = False
    policy = {} # initialize optimal policy
    need_explore = True
    cqe = cq_executor() 
    
    # GREEDY POLICY GRADIENT ASCENT LOOP
    while policy_converged == False:
        mdp.STATE = mdp.STATE * 0 # mdp reset to empty voxel
        updated_policy = {} # initialize next policy to be learned
        # LEARN NEW POLICY
        while mdp.stop_case() == False:
            # check if first epoch of learning
            if need_explore == True: 
                # find next centroid to explore and rollout from there
                next_centroid = clustered_explore(np.logical_or(TARGET, mdp.STATE) - mdp.STATE)
                rollouts = mdp.optimized_rollouts(next_centroid)

                # determine best of rollouts and add to updated_policy
                max_reward = max(rollouts.keys())
                best_action = rollouts[max_reward]
                update_MDP_state(best_action, mdp)
                updated_policy.update({max_reward: best_action})

                print("currently exploring...... percent target reached = " + str(np.count_nonzero(np.logical_and(TARGET, mdp.STATE))/np.count_nonzero(TARGET)))
                if mdp.stop_case() == True:
                    need_explore = False

            else: 
                count_actions_pruned = 0
                for reward, action in policy.items():
                    xo, yo, zo = (action[1], action[2], action[3])
                    remaining_unoccupied_voxel = np.logical_or(TARGET, mdp.STATE) - mdp.STATE
                    if remaining_unoccupied_voxel[int(xo), int(yo), int(zo)] == 1:
                        updated_action, new_reward = hooke_jeeves(action, reward, remaining_unoccupied_voxel)
                        update_MDP_state(updated_action, mdp)
                        updated_policy.update({new_reward: updated_action})
                    else: 
                        count_actions_pruned +=1 
                        print(str(count_actions_pruned) + " ACTIONS PRUNED!")
                if mdp.stop_case() == False:
                    need_explore = True

        policy_converged = converged(policy, updated_policy)
        policy = updated_policy
        num_epochs += 1
        cqe.execute_policy(policy, filename + "_epoch" + str(num_epochs))
        print("---" + str(num_epochs) + " LEARNING EPOCHS COMPLETE---")
        print("---" + str(len(policy.keys())) + " ACTIONS IN POLICY---")
        
 
    return policy

if __name__ == '__main__':
    filename = input("What is the name of your voxel file (omit '.npy')?: ")
    TARGET = np.load(filename + ".npy")
    mdp = MDP(TARGET, ALPHA)
    policy = learn(mdp)
    