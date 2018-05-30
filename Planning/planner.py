class Planner(object):
    """

    Planner class

    """

    def __init__(self, mdp):

        self.name = name

        # MDP components.
        self.mdp = mdp
        self.init_state = self.mdp.get_init_state()
        self.states = set([])
        self.actions = mdp.get_actions()
        self.reward_func = mdp.get_reward_func()
        self.transition_func = mdp.get_transition_func()
        self.gamma = mdp.gamma
        self.has_planned = False

        def plan(self, state):
            pass

        def policy(self, state):
            pass
