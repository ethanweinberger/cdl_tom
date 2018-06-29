from slow_sampler_example import slow_sampler_example
from utils import make_grid_world_from_file
from utils import generate_demonstrations
from vis_utils import heatmap_2d
from Planner import Planner
from SlowRewardSampler import SlowRewardSampler
from deep_maxent_irl import deep_maxent_irl
from utils import get_reward_function_log_likelihood
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=0.005, help="Softmax parameter")
parser.add_argument("--map", type=str, default="empty_map.mp", help="Map file name")
parser.add_argument("--num_demonstrations", type=int, default=25,
        help="Number of expert demonstrations")
parser.add_argument("--learning_rate", type=float, default=0.05, help="Network learning rate")
parser.add_argument("--num_iterations", type=int, default=20,
        help="Number of network training iterations")
args = parser.parse_args()

def main():
    print("Sampling reward functions using slow sampler...")
    mdp = make_grid_world_from_file(args.map)
    mdp.visualize_initial_map()
    planner = Planner(mdp, tau=args.tau, sample_rate=5)

    demonstrations = generate_demonstrations(planner, args.num_demonstrations)

    sampler = SlowRewardSampler(planner)
    reward_matrix = sampler.sample_reward_functions(demonstrations)

    plt.plot(sampler.likelihood_vals)
    print(sampler.likelihood_vals[-1])
    plt.ylabel("Log-likelihood")
    plt.xlabel("Iteration")

    print("Sampling reward function using deep network...")

    num_states = mdp.height * mdp.width
    feature_map = np.eye(num_states)
    reward_array = deep_maxent_irl(
            feature_map, planner, demonstrations, args.learning_rate, args.num_iterations)

    reward_matrix = np.reshape(reward_array, (mdp.height, mdp.width))
    reward_likelihood = get_reward_function_log_likelihood(planner,
            reward_matrix, demonstrations)
    plt.plot([0], [reward_likelihood], marker='o')
    print(reward_likelihood)
    plt.savefig("likelihoods.png")

if __name__ == "__main__":
    main()
