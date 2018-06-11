"""
Miscellanous utility functions for demonstration scripts
"""

import os
import random
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
