import numpy as np
import random
from utils import generate_demonstrations
from utils import make_grid_world_from_file
from utils import get_potential_rewards
from utils import create_reward_matrix_from_location
from utils import get_partial_trajectory_log_likelihood
from utils import get_reward_function_log_likelihood
from utils import Step
from Planner import Planner

class PriorRewardSampler(object):
    def __init__(self, planner, prior_file="reward_prior.npy"):
        self.planner = planner
        self.likelihood_vals = []
        self.reward_prior = np.load(prior_file)

    def sample_reward_functions_softmax(self, demonstrations, softmax_param=0.25, num_samples=50):
        """
        Samples possible reward function, where the probability of picking a particular
        location for our reward is a function of the square's relative weighting based
        on our reward prior

        Args:
            demonstrations (list of step lists): Expert demonstrations
            softmax_param (float): Softmax temperature for our probability weightings
            num_samples (int): Number of reward functions to sample before terminating
        
        Returns:
            rewards (height x width array): Matrix representing the reward values at each state
        """
 
        reward_distribution = self.construct_reward_distribution(self.reward_prior, softmax_param)
        num_map_indices = self.planner.mdp.height * self.planner.mdp.width

        self.likelihood_vals = []
        best_likelihood = float("-inf")
        reward_matrix = np.zeros((self.planner.mdp.height, self.planner.mdp.width))
    
        for i in range(num_samples):
            reward_index = np.random.choice(num_map_indices, p=reward_distribution.flatten())
            reward_coordinates = self.integer_to_coordinate(reward_index)
            print(reward_coordinates)
            current_reward_matrix = create_reward_matrix_from_location(reward_coordinates, self.planner)
            current_likelihood = get_reward_function_log_likelihood(self.planner, current_reward_matrix, demonstrations)

            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                reward_matrix = current_reward_matrix
            
            self.likelihood_vals.append(best_likelihood)
        
        return reward_matrix

    def integer_to_coordinate(self, index):
        """
        Converts a single integer into the cooresponding coordinate in a map
        with dimensions defined by map_shape

        Args:
            index (int): 1D integer representing a coordinate, where 1 cooresponds to (0,0), 2
                     corresponds to (0, 1), etc.
            map_shape (int tuple): (height, width) tuple representing the dimensions of our map
        Returns:
            coordinate (int tuple): (row, column) coordinates corresponding to our 1D number
        """ 
        map_height = self.planner.mdp.height
        map_width = self.planner.mdp.width
    
        index_height = 0
        index_width = 0
    
        while index >= map_width:
            index -= map_width
            index_height += 1

        index_width = index
        return (index_height, index_width)
     
    def construct_reward_distribution(self, reward_matrix, softmax_param):
        """
        Given a reward matrix, normalizes it into a probability distribution
        (i.e., the sum of the values in the new matrix = 1).  Before normalizing
        we apply a softmax function to control how much low-reward squares
        influence the resulting distribution.
    
        Args:
            reward_matrix (Height * Width float matrix): matrix of reward values
            softmax_param (float): Parameter for our softmax function
        Returns:
            reward_distribution (Height * Width float matrix): Matrix of softmaxed rewards
        """
        max_val = np.amax(reward_matrix)
        reward_matrix = reward_matrix - max_val
        reward_distribution = np.exp(reward_matrix / softmax_param)
        reward_distribution = reward_distribution / np.sum(reward_distribution)
        return reward_distribution 

    def threshold_greedy(self, map_file = "L_wall.mp", prior_file = "reward_prior.npy"):
        reward_prior = np.load(prior_file)
        reward_location_list = get_potential_rewards(reward_prior)
        print(reward_location_list)
 
        mdp = make_grid_world_from_file("L_wall.mp")
        mdp.visualize_initial_map()
        planner = Planner(mdp, tau=0.005, sample_rate=5) 

        reward_matrix_list = [create_reward_matrix_from_location(location, planner)
            for location in reward_location_list]

        num_demonstrations = 1
        expert_demonstration = generate_demonstrations(planner, num_demonstrations)[0]
        partial_demonstration = expert_demonstration[:len(expert_demonstration)//2]
    
        for (reward_matrix, reward_location) in zip(reward_matrix_list, reward_location_list):
            likelihood = get_partial_trajectory_log_likelihood(planner, reward_matrix, partial_demonstration) 
            print(reward_location, likelihood)
