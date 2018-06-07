import csv
from GridWorldMDP import GridWorldMDP
from GridWorldMDP import make_grid_world_from_file
from Planner import Planner
import numpy as np
from deep_maxent_irl import deep_maxent_irl
from vis_utils import heatmap_2d

#TODO: Refactor this out!
from collections import namedtuple

Step = namedtuple('Step', 'cur_state action reward')

def main():
    # Setup MDP, Agents.
    
    mdp = make_grid_world_from_file("empty_map2.mp") 
    planner = Planner(mdp, sample_rate=5)
    """
    planner.run_vi() 

    mdp.visualize_initial_map()
    action_seq, state_seq, reward_seq = planner.plan(planner.mdp.get_init_state())
    for state, action in zip(state_seq, action_seq):
        print(state, action)
    """
    num_demonstrations = 50
    expert_demonstrations = generate_demonstrations(planner, num_demonstrations) 
       
    num_states = mdp.height * mdp.width
    feature_map = np.eye(num_states)
    learning_rate = 0.02
    num_iterations = 20
    reward_array = deep_maxent_irl(
            feature_map, planner, expert_demonstrations, learning_rate, num_iterations) 

    print(reward_array)

    reward_matrix = np.reshape(reward_array, (mdp.height, mdp.width), order='C')
    heatmap_2d(reward_matrix, "Recovered Reward Values")
    

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

if __name__ == "__main__":
    main()
