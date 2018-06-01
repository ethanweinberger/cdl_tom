from GridWorldMDP import GridWorldMDP
from Planner import Planner

def main():
    # Setup MDP, Agents.
    mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6, 6)])
    value_iter = Planner(mdp, sample_rate=5)
    value_iter.run_vi()

    # Visualize the map
    mdp.visualize_initial_map()

    # Value Iteration.
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    print("Plan for", mdp)
    for i in range(len(action_seq)):
        print("\t", action_seq[i], state_seq[i])
    

if __name__ == "__main__":
    main()

