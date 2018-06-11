"""
Functions used to sample a reward function using
a "slow" method
"""

import numpy as np
import math
import copy

from utils import get_reward_function_log_likelihood

class SlowRewardSampler(object):

    def __init__(self, planner):
        self.planner = planner

    def sample_reward_functions(self, demonstrations, num_samples=50):
        """
        Randomly sample num_samples possible reward matrices
        
        Args:
            num_samples (int): Number of sample to take before terminating

        Returns:
            rewards (heigh x width array): Matrix representing the reward values
                                           at each state

        """
        best_likelihood = float("-inf")
        reward_matrix = np.zeros((self.planner.mdp.height, self.planner.mdp.width))

        for i in range(num_samples):
            current_reward_matrix = self._generate_reward_function()
            current_likelihood    = get_reward_function_log_likelihood(
                    self.planner, current_reward_matrix, demonstrations) 

            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                reward_matrix = current_reward_matrix

            print(current_likelihood)

        return reward_matrix

    def _generate_reward_function(self, lambda_parameter=1):
        """
        Randomly samples a reward function using a poisson prior

        Args:
            mdp (GridWorldMDP): MDP representing the world and all its states

        Returns:
            rewards (height x width array): Matrix representing the reward values at
                                            each state
        
        """
        mdp = self.planner.mdp
        reward_height = mdp.height
        reward_width = mdp.width

        reward_matrix = np.random.poisson(lambda_parameter, (reward_height, reward_width))
        max_reward = np.amax(reward_matrix)

        normalized_reward_matrix = reward_matrix / max_reward
        return normalized_reward_matrix
