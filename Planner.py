from collections import defaultdict
import random
import math
import queue


class Planner(object):

    def __init__(self, mdp, name="value_iter", delta=0.0001, tau=0.005, max_iterations=500, sample_rate=3):
        '''
        Args:
            mdp (MDP)
            delta (float): After an iteration if VI, if no change more than @\delta has occurred, terminates.
            max_iterations (int): Hard limit for number of iterations.
            sample_rate (int): Determines how many samples from @mdp to take to estimate T(s' | s, a).
            tau (float): Softmax parameter
        '''

        self.delta = delta
        self.max_iterations = max_iterations
        self.sample_rate = sample_rate
        self.value_func = defaultdict(float)
        self.reachability_done = False
        self.has_computed_matrix = False
        
        #My additions
        self.mdp        = mdp
        self.states     = set([])
        self.actions    = self.mdp.get_actions()
        self.init_state = self.mdp.get_init_state()
        self.transition_func = self.mdp.get_transition_func()
        self.reward_func     = self.mdp.get_reward_func()
        self.gamma           = self.mdp.get_gamma()
        self.tau = tau
    
    def _compute_matrix_from_trans_func(self):
        if self.has_computed_matrix:
            self._compute_reachable_state_space()
            # We've already run this, just return.
            return

        self.trans_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
            # K: state
                # K: a
                    # K: s_prime
                    # V: prob

        for s in self.get_states():
            for a in self.actions:
                for sample in range(self.sample_rate):
                    s_prime = self.transition_func(s, a)
                    self.trans_dict[s][a][s_prime] += 1.0 / self.sample_rate

        self.has_computed_matrix = True
    
    def get_gamma(self):
        return self.mdp.get_gamma()

    def get_num_states(self):
        if not self.reachability_done:
            self._compute_reachable_state_space()
        return len(self.states)      

    def get_states(self):
        if self.reachability_done:
            return list(self.states)
        else:
            self._compute_reachable_state_space()
            return list(self.states)

    def get_value(self, s):
        '''
        Args:
            s (State)

        Returns:
            (float)
        '''
        return self._compute_max_qval_action_pair(s)[0]

    def get_q_value(self, s, a):
        '''
        Args:
            s (State)
            a (str): action

        Returns:
            (float): The Q estimate given the current value function @self.value_func.
        '''
        # Compute expected value.
        expected_future_val = 0
        for s_prime in self.trans_dict[s][a].keys():
            expected_future_val += self.trans_dict[s][a][s_prime] * self.value_func[s_prime]

        return self.reward_func(s,a) + self.gamma*expected_future_val

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''

        if self.reachability_done:
            return

        state_queue = queue.Queue()
        state_queue.put(self.init_state)
        self.states.add(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in range(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    next_state = self.transition_func(s,a)

                    if next_state not in self.states:
                        self.states.add(next_state)
                        state_queue.put(next_state)

        self.reachability_done = True

    def run_vi(self):
        '''
        Summary:
            Runs ValueIteration and fills in the self.value_func.           
        '''
        # Algorithm bookkeeping params.
        iterations = 0
        max_diff = float("inf")
        self._compute_matrix_from_trans_func()
        state_space = self.get_states()

        # Main loop.
        while max_diff > self.delta and iterations < self.max_iterations:
            max_diff = 0
            for s in state_space:
                # print("s", s)
                if s.is_terminal():
                    continue

                max_q = float("-inf")
                for a in self.actions:
                    q_s_a = self.get_q_value(s, a)
                    max_q = q_s_a if q_s_a > max_q else max_q

                # Check terminating condition.
                max_diff = max(abs(self.value_func[s] - max_q), max_diff)

                # Update value.
                self.value_func[s] = max_q
            iterations += 1

        value_of_init_state = self._compute_max_qval_action_pair(self.init_state)[0]
        self.has_planned = True

        return iterations, value_of_init_state

    def print_value_func(self):
        for key in self.value_func.keys():
            print(key, ":", self.value_func[key])

    def plan(self, state=None, horizon=20):
        '''
        Args:
            state (State): Starting state (defaults to MDP's initial state if None)
            horizon (int): Maximum number of steps

        Returns:
            action_seq (str list): List of actions
            state_seq (GridWorldState list): List of states visited
            reward_seq (float list): List of rewards corresponding to state
        '''

        state = self.mdp.get_init_state() if state is None else state

        if self.has_planned is False:
            print("Warning: VI has not been run. Plan will be random.")

        action_seq = []
        state_seq  = []
        reward_seq = [] 
        steps = 0

        while (not state.is_terminal()) and steps < horizon:
            next_action = self._get_softmax_q_action(state)
            action_seq.append(next_action)

            current_reward = self.mdp.state_to_reward(state)
            reward_seq.append(current_reward)

            state_seq.append(state)
            state = self.transition_func(state, next_action)
            steps += 1
        
        if state.is_terminal():
            next_action = "stay"
            action_seq.append(next_action)
            
            current_reward = self.mdp.state_to_reward(state)
            reward_seq.append(current_reward)

            state_seq.append(state)
             

        return action_seq, state_seq, reward_seq

    def _get_softmax_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): Action chosen by sampling from a probability distribution
                   constructed by softmaxing all possible actions
        '''

        return self._compute_softmax_action_pair(state)

    def _compute_softmax_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            str: action
        '''

        # Take random action in case we can't choose one
        best_action = self.actions[0]
        action_softmax_pairs = []
        softmax_total = 0

        #To prevent softmax from overflowing
        max_val = max(self.get_q_value(state, action) for action in self.actions)

        for action in self.actions:
            q_s_a         = self.get_q_value(state, action)
            q_s_a         -= max_val
            softmax_val   = math.exp(q_s_a / self.tau)
            softmax_total += softmax_val
            
            action_softmax_pairs.append((action, softmax_val))
        
        action_prob_pairs = []
        for (action, softmax_val) in action_softmax_pairs:
            prob = softmax_val / softmax_total

            action_prob_pairs.append((action, prob))

        sample = random.uniform(0,1)
        for (action, prob) in action_prob_pairs:
            if sample < prob:
                return action
            else:
                sample -= prob
    
    def _get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): The action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        max_q_val = float("-inf")
        best_action = self.actions[0]

        # Find best action (action w/ current max predicted Q value)
        for action in self.actions:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

