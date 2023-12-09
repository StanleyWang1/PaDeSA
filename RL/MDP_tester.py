import numpy as np
import random
from matplotlib import pyplot as plt
from MDP import MDP

def visualize(voxel):
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = voxel.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()

def test_sphere_generator():
    target = np.random.choice([0, 1], size = (50, 50, 50), p = [0.995, 0.005])
    mdp = MDP(target, 1)
    mdp.STATE = target
    r = int(random.uniform(5, 25))
    sphere_voxel = mdp.sphere(r, (25, 25, 25))
    visualize(sphere_voxel)

def test_cylinder_generator():
    target = np.random.choice([0, 1], size = (50, 50, 50), p = [0.995, 0.005])
    mdp = MDP(target, 1)
    mdp.STATE = target
    r = int(random.uniform(5, 25))
    h = int(random.uniform(20, 40))
    ax = np.random.choice([1, 2, 3])
    cylinder_voxel = mdp.cylinder(r, h, ax, (25, 25, 25))
    visualize(cylinder_voxel)

def test_box_generator():
    target = np.random.choice([0, 1], size = (50, 50, 50), p = [0.995, 0.005])
    mdp = MDP(target, 1)
    mdp.STATE = target
    x = int(random.uniform(5, 45))
    y = int(random.uniform(5, 45))
    z = int(random.uniform(5, 45))
    boxel = mdp.box(x, y, z, (25, 25, 25))
    visualize(boxel)

def test_reward_convergence_function():
    target = np.zeros((50, 50, 50))
    mdp = MDP(target, 0.95)

    sphere_target = mdp.sphere(15, (25, 25, 25))
    mdp.TARGET = sphere_target

    rewards = []
    for r in np.arange(50):
        new_sphere = mdp.sphere(r, (25, 25, 25))
        mdp.STATE = new_sphere
        rewards.append(mdp.reward(new_sphere))
        print(mdp.stop_case())

    plt.figure()
    plt.plot(np.arange(50), rewards)
    plt.title("reward between growing sphere and sphere of constant radius 15")
    plt.xlabel('radius of growing sphere')
    plt.ylabel('reward')
    plt.show()

def test_full_rollout():
    target = np.random.choice([0, 1], size = (50, 50, 50), p = [0.995, 0.005])
    mdp = MDP(target, 1)
    centroid = (25, 25, 25)
    rollout_rewards = mdp.full_rollout(centroid)

    rewards = rollout_rewards.keys()
    print(max(rewards))

def test_optimized_rollout():
    target = np.zeros([50, 50, 50])
    mdp = MDP(target, 1)
    centroid = (25, 25, 25)
    mdp.TARGET = mdp.box(45, 30, 15, centroid)
    new_centroid = (20,20,20)
    best_action = mdp.optimized_rollouts(new_centroid)
    print(best_action)
    

if __name__ == "__main__":
    #test_sphere_generator()
    #test_cylinder_generator()
    #test_box_generator()
    #test_reward_convergence_function()
    #test_full_rollout()
    test_optimized_rollout()