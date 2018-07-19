import sys
sys.path.append("../")

from GridWorldMDP import GridWorldMDP
from Planner import Planner
from utils import get_agent_positions_in_grid
from utils import get_steps_from_position_list
from utils import read_annotations
from utils import create_video_grid
from deep_maxent_irl import deep_maxent_irl

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_demonstrations", type=int, default=25,
        help="Number of training demonstrations")
parser.add_argument("--map", type=str, default="empty_map.mp", help="Map file name")
parser.add_argument("--tau", type=float, default=0.005, help="Softmax parameter")
parser.add_argument("--learning_rate", type=float, default=0.05, help="Network learning rate")
parser.add_argument("--num_iterations", type=int, default=15,
        help="Number of network training iterations")

args = parser.parse_args()
VIDEO_WIDTH = 1400
VIDEO_HEIGHT = 1904
BLOCK_SIZE = 60

def main():

    demonstration_list = []
    agents = read_annotations("annotations.txt")
    grid = create_video_grid(VIDEO_WIDTH, VIDEO_HEIGHT, BLOCK_SIZE)
    
    for agent in agents:
        positions = get_agent_positions_in_grid(agent, grid)
        step_list = get_steps_from_position_list(positions)
        demonstration_list.append(step_list)

    grid_width = len(grid[0])
    grid_height = len(grid)
    num_states = grid_width * grid_height
    placeholder_mdp = GridWorldMDP(width = grid_width, height=grid_height, init_loc=(0,0), 
        goal_locs = [], walls = [], name = "placeholder")    
    planner = Planner(placeholder_mdp, sample_rate=5)
    planner._compute_matrix_from_trans_func()

    feature_map = np.eye(num_states)
    reward_array = deep_maxent_irl(
        feature_map, planner, demonstration_list, args.learning_rate, args.num_iterations)  
    reward_matrix = np.reshape(reward_array, (grid_height, grid_width))
    np.save("reward_prior", reward_matrix) 

if __name__ == "__main__":
    main()
