"""
Script to demonstrate the "slow" sampling method for testing reward functions
"""

from utils import make_grid_world_from_file
from utils import generate_demonstrations
from vis_utils import heatmap_2d
from Planner import Planner
from SlowRewardSampler import SlowRewardSampler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=0.005, help="Softmax parameter")
parser.add_argument("--map", type=str, default="empty_map.mp", help="Map file name")
parser.add_argument("--num_demonstrations", type=int, default=10,
        help="Number of expert demonstrations")
args = parser.parse_args()

def main():

    mdp = make_grid_world_from_file(args.map) 
    mdp.visualize_initial_map()
    planner = Planner(mdp, tau=args.tau, sample_rate=5)    

    demonstrations = generate_demonstrations(planner, args.num_demonstrations)

    sampler = SlowRewardSampler(planner) 
    reward_matrix = sampler.sample_reward_functions(demonstrations)
    heatmap_2d(reward_matrix)

if __name__ == "__main__":
    main()
