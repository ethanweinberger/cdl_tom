import sys
sys.path.append("../")

from slow_sampler_example import slow_sampler_example
from utils import make_grid_world_from_file
from utils import generate_demonstrations
from vis_utils import heatmap_2d
from Planner import Planner
from Samplers.SlowRewardSampler import SlowRewardSampler
from Samplers.PriorRewardSampler import PriorRewardSampler
#from deep_maxent_irl import deep_maxent_irl
from utils import get_reward_function_log_likelihood
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=0.005, help="Softmax parameter")
parser.add_argument("--map", type=str, default="L_wall.mp", help="Map file name")
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

    #demonstrations = generate_demonstrations(planner, args.num_demonstrations)
    num_demonstrations = 1
    expert_demonstration = generate_demonstrations(planner, num_demonstrations)[0]
    partial_demonstration = []
    partial_demonstration.append(expert_demonstration[:len(expert_demonstration)//2])
    slow_sampler = SlowRewardSampler(planner)
    slow_reward_matrix = slow_sampler.sample_reward_functions(partial_demonstration)

    prior_sampler = PriorRewardSampler(planner)
    prior_sampler_reward_matrix = prior_sampler.sample_reward_functions_softmax(partial_demonstration)

    plt.plot(slow_sampler.likelihood_vals, label="Naive Sampler")
    plt.plot(prior_sampler.likelihood_vals, label="Prior Sampler")
    plt.legend(loc="best")
    print(slow_sampler.likelihood_vals[-1])
    print(prior_sampler.likelihood_vals[-1])
    plt.ylabel("Log-likelihood")
    plt.xlabel("Iteration")
    plt.show()

    #print("Sampling reward function using deep network...")

    #num_states = mdp.height * mdp.width
    #feature_map = np.eye(num_states)
    #reward_array = deep_maxent_irl(
    #        feature_map, planner, demonstrations, args.learning_rate, args.num_iterations)

    #reward_matrix = np.reshape(reward_array, (mdp.height, mdp.width))
    #reward_likelihood = get_reward_function_log_likelihood(planner,
    #        reward_matrix, demonstrations)
    #plt.plot([0], [reward_likelihood], marker='o')
    #print(reward_likelihood)
    heatmap_2d(slow_reward_matrix)

if __name__ == "__main__":
    main()
