import csv
from GridWorldMDP import GridWorldMDP
from Planner import Planner
import numpy as np
from deep_maxent_irl import deep_maxent_irl
from vis_utils import heatmap_2d
from utils import generate_demonstrations
from utils import make_grid_world_from_file
from utils import Step
from utils import get_reward_function_log_likelihood 
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--num_demonstrations", type=int, default=25,
        help="Number of training demonstrations")
parser.add_argument("--map", type=str, default="empty_map.mp", help="Map file name")
parser.add_argument("--tau", type=float, default=0.005, help="Softmax parameter")
parser.add_argument("--learning_rate", type=float, default=0.05, help="Network learning rate")
parser.add_argument("--num_iterations", type=int, default=20, 
        help="Number of network training iterations")


args = parser.parse_args()


def inverse_model_example():
    # Setup MDP, Agents.
   
    mdp = make_grid_world_from_file(args.map) 
    mdp.visualize_initial_map()
    planner = Planner(mdp, sample_rate=5)

    expert_demonstrations = generate_demonstrations(planner, args.num_demonstrations) 
       
    num_states = mdp.height * mdp.width
    feature_map = np.eye(num_states)
    reward_array = deep_maxent_irl(
            feature_map, planner, expert_demonstrations, args.learning_rate, args.num_iterations) 

    reward_matrix = np.reshape(reward_array, (mdp.height, mdp.width))
    reward_likelihood = get_reward_function_log_likelihood(planner, 
            reward_matrix, expert_demonstrations)

    heatmap_2d(reward_matrix, "Recovered Reward Values")
    
	
def generate_reward_prior(map_list = ["top_right.mp", "top_left.mp", "bottom_right.mp", "bottom_left.mp",
    "center.mp"], save_prior=False):

    expert_demonstrations = []
    for map_name in map_list:
        mdp = make_grid_world_from_file(map_name)
        mdp.visualize_initial_map()
        planner = Planner(mdp, sample_rate=5)

        expert_demonstrations.extend(generate_demonstrations(planner, args.num_demonstrations))

    random.shuffle(expert_demonstrations)

    num_states = mdp.height * mdp.width
    feature_map = np.eye(num_states)
    reward_array = deep_maxent_irl(
            feature_map, planner, expert_demonstrations, args.learning_rate, args.num_iterations)

    reward_matrix = np.reshape(reward_array, (mdp.height, mdp.width))

    if save_prior:
        np.save("reward_prior", reward_matrix)
    
    heatmap_2d(reward_matrix, "Recovered Reward Values")

def save_plan(output_name, map_name, action_seq, state_seq):
    """
    Saves a plan (combination of action sequence and state sequence)
    to a CSV file.  Our CSV is organized such that its columns are

    Action    State.x    State.y

    Args:
        output_name (str): Name for our output CSV file
        map_name (str): Name of the map that our planner navigated in
        action_seq (str list): List of actions taken by our agent
        state_seq (State list): List of states visited by our agent

    Returns:
        Nothing (outputs to CSV file)
    """
    data = []
    column_names = ["Action", "State.x", "State.y", "Map_Name"]
    data.append(column_names)

    for action, state in zip(action_seq, state_seq):
        new_row = [action, state.x, state.y, map_name]
        data.append(new_row)

    output_file  = open(output_name + ".csv", 'w')
    with output_file:
        writer = csv.writer(output_file)
        writer.writerows(data)
