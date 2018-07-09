"""
Test script for forward model.
"""
import sys
sys.path.append("../")

from GridWorldMDP import GridWorldMDP
from utils import make_grid_world_from_file
from Planner import Planner
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=0.005, help="Softmax parameter")
parser.add_argument("--map", type=str, default="big_map.mp", help="Map file name")
args = parser.parse_args()

def main():
    
    mdp = make_grid_world_from_file(args.map)
    planner = Planner(mdp, sample_rate=5, tau=args.tau)
    
    planner.run_vi() 

    mdp.visualize_initial_map()
    action_seq, state_seq, reward_seq = planner.plan(planner.mdp.get_init_state())
    for state, action in zip(state_seq, action_seq):
        print(state, action)
    
if __name__ == "__main__":
    main()
