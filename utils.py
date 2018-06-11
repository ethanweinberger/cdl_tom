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

def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None):
    """
    Builds a GridWorldMDP from a file:
    	'w' --> wall
        'a' --> agent
        'g' --> goal
        '-' --> empty
    
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)

    Returns:
        (GridWorldMDP)

    """

    if name is None:
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

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]]

    if randomize:
        agent_x, agent_y = random.choice(empty_cells)
        if len(goal_locs) == 0:
            # Sample @num_goals random goal locations.
            goal_locs = random.sample(empty_cells, num_goals)
        else:
            goal_locs = random.sample(goal_locs, num_goals)

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

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

        #Added so the network recognizes end states as "reward" states
        for i in range(10):
            episode.append(episode[-1])

        trajectories.append(episode)

    return trajectories
