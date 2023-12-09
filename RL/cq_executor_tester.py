import numpy as np
from cq_executor import cq_executor

if __name__ == "__main__":
    """
    Test script for cq_executor.py
    """

    # generate test policy
    action1 = np.array([1, 5, 5, 5, 10, 20, 30]) # box 1
    action2 = np.array([1, 0, 0, 0, 15, 15, 15]) # box 2
    action3 = np.array([3, 5, -5, -5, 10]) # sphere
    action4 = np.array([2, 10, 10, 10, 3, 40, 3]) # cylinder 1
    action5 = np.array([2, 10, 10, 10, 3, 40, 2]) # cylinder 2
    action6 = np.array([2, 10, 10, 10, 3, 40, 1]) # cylinder 3
    policy = {1:action1, 2:action2, 3:action3, 4:action4, 5:action5, 6:action6} # construct policy

    cqe = cq_executor()
    cqe.execute_policy(policy, filename= "cq_executor_test")