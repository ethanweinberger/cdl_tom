import numpy as np
from utils import generate_demonstrations
from utils import make_grid_world_from_file
from utils import get_potential_rewards
from utils import create_reward_matrix_from_location
from utils import get_partial_trajectory_log_likelihood
from utils import Step
from Planner import Planner

if __name__ == "__main__":
    reward_prior = np.load("reward_prior.npy")
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
    
    for reward_matrix in reward_matrix_list:
        likelihood = get_partial_trajectory_log_likelihood(planner, reward_matrix, partial_demonstration) 
        print(likelihood)
    
    
     
    
     
