"""
Class for MDP objects
"""

class MDP(object):
    """
    Class for a Markov Decision Process
    """

    def __init__(self, S=[], A=[], transition_func=[], reward_func=[], init_state, gamma=0.95):
        """

        Markov Decision Process (MDP) class.

        Args:
            S (list): List of states
            A (list): List of actions
            T (matrix): Transition matrix where T[SO,A,SF] 
			is the probability of moving from So to SF 
			after taking action A
            R (matrix): Reward function where R[A,S] is the reward 
			for taking action A in state S
            init_state (State): Starting state of the MDP 
            gamma (float): Future discount

        Returns:
            MDP object
        """
        
        self.S = S
        self.A = A
        self.T = T
        self.R = R

        self.init_state = init_state
        self.gamma = gamma

        def execute_action(self, action):
            """
            Executes a single time step fo the MDP

            Args:
                action (str)
            Returns:
                (tuple: <float, State>): reward, State

            """
            reward = R[self.cur_state, action]

            #TODO: Figure this out
            next_state = self.transition_func(self.cur_state, action)
            self.cur_state = next_state

            return reward, next_state

        def transition_func(self, state, action):
            """
            Produces a new state given a state and an action


            """


        def reset(self):
            self.cur_state = self.init_state
