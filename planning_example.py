import csv
from GridWorldMDP import GridWorldMDP
from GridWorldMDP import make_grid_world_from_file
from Planner import Planner

def main():
    # Setup MDP, Agents.
    #mdp = GridWorldMDP(width=6, height=6, walls = [(1,2), (2,2), (3,2)],goal_locs=[(6, 6)])
    
    mdp = make_grid_world_from_file("multi_goal.mp") 
    value_iter = Planner(mdp, sample_rate=5)
    value_iter.run_vi()

    # Visualize the map
    mdp.visualize_initial_map()

    # Value Iteration.
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    print("Plan for", mdp)
    for i in range(len(action_seq)):
        print("\t", action_seq[i], state_seq[i])
    
    save_plan("test", "multi_goal", action_seq, state_seq)

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

