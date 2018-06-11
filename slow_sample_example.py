"""
Script to demonstrate the "slow" sampling method for testing reward functions
"""

from utils import make_grid_world_from_file
from utils import generate_demonstrations
from Planner import Planner
from SlowRewardSampler import SlowRewardSampler

def main():

    mdp = make_grid_world_from_file("empty_map.mp") 
    mdp.visualize_initial_map()
    planner = Planner(mdp, sample_rate=5)    

    demonstrations = generate_demonstrations(planner, 10)

    sampler = SlowRewardSampler(planner) 
    sampler.sample_reward_functions(demonstrations)

if __name__ == "__main__":
    main()
