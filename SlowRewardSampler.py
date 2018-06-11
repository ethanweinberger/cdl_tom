"""
Functions used to sample a reward function using
a "slow" method
"""

import numpy as np
import math

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
            current_likelihood    = self._get_reward_function_log_likelihood(
                    current_reward_matrix, demonstrations) 

            #print(current_likelihood)
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                reward_matrix = current_reward_matrix
        
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

    def _get_reward_function_log_likelihood(self, reward_matrix, demonstrations):
        """
        Given a matrix representing a potential reward function, calculates
        the likelihood given a list of expert demonstrations

        Args:
            reward_matrix (Height * Weight array): Matrix representing the rewards
                                                   contained in each state
            demonstrations (List of step lists): List of demonstrations composed
                                                 of lists of step tuples
        
        Returns:
            total_log_likelihood (float): Log likelihood of demonstrations given reward matrix
        
        """

        if self.planner.has_planned == False:
            self.planner.run_vi()

        total_log_likelihood = 0
        for demonstration in demonstrations:
            demonstration_log_likelihood = 0
            for step in demonstration:
               demonstration_log_likelihood += self._get_step_log_likelihood(step)
            total_log_likelihood += demonstration_log_likelihood
        
        return total_log_likelihood
    
    def _get_step_log_likelihood(self, step):
        """
        Based on a given step in an expert demonstration, calculates the
        likelihood of that step given that our agent has run value iteration
        on the world

        Args:
            step (named tuple): Container for all relevant information (state, action, etc)
                                about the current step in an expert demonstration

        Returns:
            step_log_likelihood (float): Log-likelihood of a (state, action) pair

        """

        softmax_total = 0
        state = step.cur_state
        action = step.action
        for action in self.planner.actions:
            q_s_a         = self.planner.get_q_value(state, action)
            softmax_val   = math.exp(q_s_a / self.planner.tau)
            softmax_total += softmax_val
        q_s_a = self.planner.get_q_value(state, action)
        softmax_val = math.exp(q_s_a / self.planner.tau)
        step_likelihood = softmax_val / softmax_total
        step_log_likelihood = np.log(step_likelihood)

        return step_log_likelihood
