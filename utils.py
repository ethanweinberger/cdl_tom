"""
Miscellanous utility functions for demonstration scripts
"""

import os
import random
import math
import copy
import numpy as np
from Planner import Planner 
from GridWorldMDP import GridWorldMDP
from collections import namedtuple

#TODO: Refactor this into a class
Step = namedtuple('Step', 'cur_state action reward')

def make_grid_world_from_file(file_name):
    """
    Builds a GridWorldMDP from a file:
    	'w' --> wall
        'a' --> agent
        'g' --> goal
        '-' --> empty
    
    Args:
        file_name (str): Name of map file

    Returns:
        (GridWorldMDP)

    """

    name = file_name.split(".")[0]

    grid_path = os.path.dirname(os.path.realpath(__file__))
    wall_file = open(os.path.join(grid_path, "gridworld_maps", file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    empty_cells = []
    agent_x, agent_y = 1, 1
    walls = []
    goal_locs = []
    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_x, agent_y = j + 1, num_rows - i
            elif ch == "-":
                empty_cells.append((j + 1, num_rows - i))

    return GridWorldMDP(width=num_cols, height=num_rows, init_loc=(agent_x, agent_y), goal_locs=goal_locs, walls=walls, name=name)

def get_reward_function_log_likelihood(planner, reward_matrix, demonstrations):
    """
    Given a matrix representing a potential reward function, calculates
    the likelihood given a list of expert demonstrations

    Args:
        planner (Planner): Planner object representing the true mdp
        reward_matrix (Height * Width array): Matrix representing the rewards
                                               contained in each state
        demonstrations (List of step lists): List of demonstrations composed
                                             of lists of step tuples

    Returns:
        total_log_likelihood (float): Log likelihood of demonstrations given reward matrix

    """

    #TODO: Refactor so we don't need to make a copy
    planner = copy.deepcopy(planner)

    planner.set_reward_function(reward_matrix)
    planner.run_vi() 

    total_log_likelihood = 0
    for demonstration in demonstrations:
         demonstration_log_likelihood = 0
         for step in demonstration:
             demonstration_log_likelihood += _get_step_log_likelihood(planner, step)
         total_log_likelihood += demonstration_log_likelihood

    return total_log_likelihood

def get_partial_trajectory_log_likelihood(planner, reward_matrix, partial_trajectory):
    """
    Given a list of steps representing a partial trajectory, determines the likelihood of that
    trajectory.  Different reward matrices can be used to compare the validity of different reward
    functions.

    Args:
        planner (Planner): Planner object representing our mdp
        reward_matrix: (Height * Width array): Matrix representing the rewards contained
                                               in each state
        partial_trajectory (step list): Our list of moves so far
    Returns:
        total_log_likelihood (float): Log likelihood of demonstrations given reward matrix

    """
    planner = copy.deepcopy(planner)
    planner.set_reward_function(reward_matrix)
    planner.run_vi()
    
    total_log_likelihood = 0
    for step in partial_trajectory:
        total_log_likelihood += _get_step_log_likelihood(planner, step)
    return total_log_likelihood


def get_potential_rewards(reward_matrix, reward_cutoff=0.5):
    """
    Given a matrix representing a reward prior, determines the locations of all
    potential reward squares in the matrix

    Args:
	reward_matrix (Height * Width array): Matrix representing the rewards contained
                                              in each state
        reward_cutoff (float): Cutoff for what reward value coutns as a "reward square"
    Returns:
        reward_locations (tuple list): List of tuples with the location of each reward
    
    """
    reward_locations = []
    height, width = reward_matrix.shape

    for row in range(height):
       for column in range(width):
           if reward_matrix[row, column] > reward_cutoff:
               reward_locations.append((row,column))
    
    return reward_locations
     

def create_reward_matrix_from_location(location, planner):
    """
    Give a location tuple, creates a reward matrix with the location represented
    by that tuple as the reward

    Args:
        location (int tuple): Location of our reward
        planner (Planner): planner object containng our MDP
    Returns:
        new_reward_matrix
    """

    height = planner.mdp.height
    width = planner.mdp.width
    
    new_reward_matrix = np.zeros((height, width))
    new_reward_matrix[location[0], location[1]] = 1.0

    return new_reward_matrix

    

def _get_step_log_likelihood(planner, step):
    """
    Based on a given step in an expert demonstration, calculates the
    likelihood of that step given that our agent has run value iteration
    on the world

    Args:
        planner (PLanner): PLanner object representing our MDP
        step (named tuple): Container for all relevant information (state, action, etc)
                            about the current step in an expert demonstration

    Returns:
        step_log_likelihood (float): Log-likelihood of a (state, action) pair

    """


    softmax_total = 0
    cur_state = step.cur_state
    cur_action = step.action
    max_val = max(planner.get_q_value(cur_state, action) for action in planner.actions)

    for action in planner.actions:
        q_s_a         = planner.get_q_value(cur_state, action)
        q_s_a         -= max_val
        softmax_val   = math.exp(q_s_a / planner.tau)
        softmax_total += softmax_val

    q_s_a = planner.get_q_value(cur_state, cur_action)
    q_s_a -= max_val

    softmax_val = math.exp(q_s_a / planner.tau)
    step_likelihood = softmax_val / softmax_total
    step_log_likelihood = np.log(step_likelihood)

    return step_log_likelihood

def generate_demonstrations(planner, num_demonstrations):
    """
    Function to generate expert demonstrations given a particular mdp.

    Args:
        mdp (GridWorldMDP): An MDP representing our environment and the agent within it
        num_demonstrations (int): Number of expert demonstrations to gather

    Returns:
        demonstrations (list): List of action sequences

    """

    planner.run_vi()
    trajectories = []

    for i in range(num_demonstrations):
        planner.mdp.reset()

        episode = []
        action_seq, state_seq, reward_seq = planner.plan(planner.mdp.get_init_state())

        for state, action, reward in zip(state_seq, action_seq, reward_seq):
            episode.append(Step(cur_state = state, action = action, reward = reward))

        trajectories.append(episode)

    return trajectories
