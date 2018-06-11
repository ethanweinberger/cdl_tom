import numpy as np
import tensorflow as tf

#TODO: Add documentation comments to all the functions

class DeepIRLFC(object):

    def __init__(self, n_input, learning_rate, n_h1=400, n_h2=300, l2=10, name="deep_irl_fc"):
        #TODO: Understand what's happening here.  I really have no idea what's going on

        self.n_input = n_input
        self.learning_rate = learning_rate
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.name = name

        self.sess = tf.Session()
        self.input_s, self.reward, self.theta = self._build_network(self.name)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        self.grad_r = tf.placeholder(tf.float32, [None, 1])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
        self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]

        self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)
        self.grad_norms = tf.global_norm(self.grad_theta)

        self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self, name):
        input_s = tf.placeholder(tf.float32, [None, self.n_input])

        with tf.variable_scope(name):
            fully_connected_layer1 = tf.contrib.layers.fully_connected(
                                        inputs = input_s,
                                        num_outputs = self.n_h1,
                                        activation_fn = tf.nn.relu,
                                        weights_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"),
                                        scope = "fully_connected_layer1"
                                     )
            fully_connected_layer2 = tf.contrib.layers.fully_connected(
                                        inputs = fully_connected_layer1,
                                        num_outputs = self.n_h2,
                                        activation_fn = tf.nn.relu,
                                        weights_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"),
                                        scope = "fully_connected_layer2"
                                     )
            reward = tf.contrib.layers.fully_connected(
                        inputs = fully_connected_layer2,
                        num_outputs = 1,
                        scope = "reward"
                     )
            
            theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            
            return input_s, reward, theta 

    def get_theta(self):
        return self.sess.run(self.theta)

    def get_rewards(self, states):
        rewards = self.sess.run(self.reward, feed_dict = {self.input_s: states})
        return rewards

    def apply_grads(self, feat_map, grad_r):
        grad_r = np.reshape(grad_r, [-1, 1])
        feat_map = np.reshape(feat_map, [-1, self.n_input])
        _, grad_theta, l2_loss, grad_norms = self.sess.run(
                [self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
                feed_dict = {self.grad_r: grad_r, self.input_s: feat_map})
        return grad_theta, l2_loss, grad_norms
    
    def approx_value_iteration(self, planner, rewards, epsilon):
        """
        Implementation of approximate value iteration as defined in
        Algorithm 2 of Maximum Entropy Deep Reinforcement Learning
        (Wulfmeier et. al, 2015)

        Args:
            planner (Planner): Planner object with MDP representing our grid world
            rewards (Nx1 array): Mapping from state ID's to rewards
            epsilon (float): Used in stopping condition for VI
        Returns:
            policy (NxN_ACTIONS array): Policy matrix
        
        """

        transition_matrix = planner.trans_dict
        gamma = planner.mdp.gamma
        num_states = planner.mdp.height * planner.mdp.width
        num_actions = len(planner.actions)
        values = np.zeros([num_states])
        states = planner.get_states()
        actions = planner.actions

        while True:
            values_tmp = values.copy()
            
            for s in planner.get_states():
                s_idx = position_to_number(planner, s) 
                max_val = float("-inf") 
                for a in actions:
                    current_val = 0
                    for s1 in states:
                        s1_idx = position_to_number(planner, s1)
                        current_val += transition_matrix[s][a][s1] * (rewards[s_idx] + gamma * values_tmp[s1_idx])
                    max_val = max(current_val, max_val)
                values[s_idx] = max_val
                
            if max([abs(values[position_to_number(planner, s)] - values_tmp[position_to_number(planner, s)]) for s in planner.states]) < epsilon:
                break
        
        policy = np.zeros([num_states, num_actions])
        for s in states:
            s_idx = position_to_number(planner, s) 
            v_s = np.array([sum([transition_matrix[s][a][s1]*(rewards[s_idx] + gamma * values[position_to_number(planner, s1)])
                for s1 in planner.get_states()]) for a in planner.actions])
            
            if np.sum(v_s) > 0: 
                policy[s_idx, :] = np.transpose(v_s/np.sum(v_s))
            else:
                policy[s_idx, :] = np.transpose([1.0 / len(actions)])
                
        return policy


def position_to_number(planner, position):
    """
    Converts the coordinates of a position (x,y) into a single
    number that acts as an ID for the position. Note that our first
    position is (1,1) rather than (0,0).

    Args:
        planner (Planner): Planner object containing our MDP/Gridworld
        position (int tuple): Position in a grid
    
    Returns:
        idx (int): Integer representation of the position

    """
    idx = (position.x - 1) + (position.y-1)*planner.mdp.width
    return idx


def demonstration_svf(planner, demonstrations):
    """
    Calculates the state visitation frequencies (svf's) from
    our demonstrations

    Args:
        planner (Planner): Planner object containing MDP
        demonstrations: list of step lists
        num_states: number of possible states in our MDP

    Returns:
        p (Nx1 float array): Array of state visitation frequencies

    """
    num_states = planner.mdp.width * planner.mdp.height
    p = np.zeros(num_states)

    for demonstration in demonstrations:
        for step in demonstration:
            position_idx = position_to_number(planner, step.cur_state)
            p[position_idx] += 1

    p = p/len(demonstrations)
    return p

def compute_state_visitation_frequency(planner, demonstrations, policy):
    """
    Calculate the expected state visitation frequency p(s | theta, T)
    using the dynamic programming algorithm described in 
    Maximum Entropy Inverse Reinforcement Learning (Ziebart et. al, 2008)

    Args:
        planner (Planner): Planern object containing MDP
        demonstrations (list of step lists): List of expert demonstrations composed of steps
        policy (NxN numpy array): Policy matrix

    Returns:
        p (Nx1 array): Array of state visitation frequencies

    """

    num_states = planner.mdp.width * planner.mdp.height
    trans_dict = planner.trans_dict
    states = planner.get_states()
    actions = planner.actions

    T = max(len(demonstration) for demonstration in demonstrations)
    
    #mu[s, t] is probability of visiting state s at time t
    mu = np.zeros([num_states, T])

    for demonstration in demonstrations:
        start_state = demonstration[0].cur_state
        start_state_idx = position_to_number(planner, start_state)

        mu[start_state_idx, 0] += 1
    mu[:, 0] = mu[:, 0] / len(demonstrations)

    #TODO: Possible point of failure
    for s in states:
        for t in range(T-1):
            s_idx = position_to_number(planner, s)

            state_sum = 0
            for pre_s in states:
                pre_s_idx = position_to_number(planner, pre_s)
                a1_idx = 0
                action_sum = 0
                for a1 in actions:
                     action_sum += mu[pre_s_idx][t] * trans_dict[pre_s][a1][s] * policy[pre_s_idx][a1_idx]
                     a1_idx += 1
                state_sum += action_sum

            mu[s_idx][t+1] = state_sum 

    p = np.sum(mu, 1)
    return p

def normalize_rewards(rewards):
    """
    Normalize reward vector to lie between (0, max_val)

    Args:
        rewards (Nx1 array): reward values for each state

    Returns:
        normalized_rewards (Nx1 array): normalized reward values for each state
    """

    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
    return normalized_rewards


def deep_maxent_irl(feature_map, planner, demonstrations, learning_rate, num_iterations=20):
    """
    Maximum Entropy Inverse Reinforcement Learning

    Args:
        feature_map (NxD numpy array): Matrix containing features for each state
        planner (Planner): Planner object
        demonstrations (list of Step lists): List of expert demonstrations 
        learning_rate (float): Learning rate for Tensorflow
        num_iterations (int): Number of training iterations

    Returns:
        rewards (Nx1 numpy array): Recovered state rewards
    """
    num_actions = len(planner.actions)
    
    nn_r = DeepIRLFC(feature_map.shape[1], learning_rate)
    mu_D = demonstration_svf(planner, demonstrations) 

    for iteration in range(num_iterations):
        print("Iteration: {}".format(iteration))

        rewards = nn_r.get_rewards(feature_map)

        #Compute policy
        policy = nn_r.approx_value_iteration(planner, rewards, epsilon=0.01)

        #Compute expected svf
        mu_exp = compute_state_visitation_frequency(planner, demonstrations, policy)
    
        #Compute gradients on rewards
        grad_r = mu_D - mu_exp

        #Apply gradients to the neural network
        nn_r.apply_grads(feature_map, grad_r)

    rewards = nn_r.get_rewards(feature_map)
    return normalize_rewards(rewards)
