"""
Script to demonstrate the "slow" sampling method for testing reward functions
"""

from utils import make_grid_world_from_file
from utils import generate_demonstrations
from vis_utils import heatmap_2d
from Planner import Planner
from SlowRewardSampler import SlowRewardSampler

def main():

    mdp = make_grid_world_from_file("empty_map.mp") 
    mdp.visualize_initial_map()
    planner = Planner(mdp, sample_rate=5)    

    demonstrations = generate_demonstrations(planner, 10)

    sampler = SlowRewardSampler(planner) 
    reward_matrix = sampler.sample_reward_functions(demonstrations)
    heatmap_2d(reward_matrix)

if __name__ == "__main__":
    main()
