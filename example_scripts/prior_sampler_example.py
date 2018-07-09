"""
Script to demonstrate a reward function sampler that is provided
with a reward prior file
"""
import sys
sys.path.append("../")

from utils import make_grid_world_from_file
from utils import generate_demonstrations
from vis_utils import heatmap_2d
from Planner import Planner
from Samplers.PriorRewardSampler import PriorRewardSampler

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=0.005, help="Softmax parameter")
parser.add_argument("--map", type=str, default="L_wall.mp", help="Map file name")
parser.add_argument("--num_demonstrations", type=int, default=25,
        help="Number of expert demonstrations")
args = parser.parse_args()

def prior_sampler_example():

    mdp = make_grid_world_from_file(args.map) 
    mdp.visualize_initial_map()
    planner = Planner(mdp, tau=args.tau, sample_rate=5)    

    demonstrations = generate_demonstrations(planner, args.num_demonstrations)

    sampler = PriorRewardSampler(planner) 
    reward_matrix = sampler.sample_reward_functions_softmax(demonstrations)

    plt.plot(sampler.likelihood_vals)
    plt.ylabel("Log-likelihood")
    plt.xlabel("Iteration")
    plt.show()
    
    heatmap_2d(reward_matrix)

if __name__ == "__main__":
    prior_sampler_example()
