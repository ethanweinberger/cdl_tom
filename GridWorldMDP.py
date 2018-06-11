'''GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
import random
import sys
import os
import numpy as np
import copy

from GridWorldState import GridWorldState

# Other imports.

class GridWorldMDP(object):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["stay", "up", "down", "left", "right"]

    def __init__(self,
                width=5,
                height=3,
                init_loc=(1,1),
                rand_init=False,
                goal_locs=[(5,3)],
                walls=[],
                is_goal_terminal=True,
                gamma=0.99,
                init_state=None,
                step_cost=0.0,
                name="gridworld"):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''

        # Setup init location.
        self.rand_init = rand_init
        if rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            while init_loc in walls:
                init_loc = random.randint(1, width), random.randint(1, height)
        self.init_loc = init_loc
        self.init_state = GridWorldState(init_loc[0], init_loc[1]) if init_state is None or rand_init else init_state
        self.actions = GridWorldMDP.ACTIONS
        self.transition_func = self._transition_func
        self.reward_func = self._reward_func
        self.gamma = gamma

        #MDP.__init__(self, GridWorldMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

        if type(goal_locs) is not list:
            print("(simple_rl) GridWorld Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)].")
            quit()
        self.step_cost = step_cost
        self.walls = walls
        self.width = width
        self.height = height
        self.goal_locs = goal_locs
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        self.is_goal_terminal = is_goal_terminal
        self.name = name

    def get_init_state(self):
        return self.init_state

    def get_actions(self):
        return self.actions
    
    def get_transition_func(self):
        return self.transition_func

    def get_reward_func(self):
        return self.reward_func
    def get_gamma(self):
        return self.gamma


    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if self._is_goal_state_action(state, action):
            return 1.0 - self.step_cost
        else:
            return 0 - self.step_cost

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if (state.x, state.y) in self.goal_locs and self.is_goal_terminal:
            # Already at terminal.
            return False

        goals = self.goal_locs

        if action == "left" and (state.x - 1, state.y) in goals:
            return True
        elif action == "right" and (state.x + 1, state.y) in goals:
            return True
        elif action == "down" and (state.x, state.y - 1) in goals:
            return True
        elif action == "up" and (state.x, state.y + 1) in goals:
            return True
        elif action == "stay" and (state.x, state.y) in goals:
            return True
        else:
            return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state

        if action == "up" and state.y < self.height and not self.is_wall(state.x, state.y + 1):
            next_state = GridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1 and not self.is_wall(state.x, state.y - 1):
            next_state = GridWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width and not self.is_wall(state.x + 1, state.y):
            next_state = GridWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1 and not self.is_wall(state.x - 1, state.y):
            next_state = GridWorldState(state.x - 1, state.y)
        elif action == "stay":
            next_state = GridWorldState(state.x, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) in self.goal_locs and self.is_goal_terminal:
            next_state.set_terminal(True)

        return next_state

    def is_wall(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''
        
        return (x, y) in self.walls

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def get_goal_locs(self):
        return self.goal_locs

    def state_to_reward(self, state):
        """
        Returns the reward value for a given state

        Args:
            state (GridWorldState): State we want to translate into reward

        Returns:
            reward (float): Reward value at the given state
        """
        if (state.x, state.y) in self.goal_locs:
            return 1.0
        else:
            return 0.0


    def visualize_initial_map(self):
        """
        Args:
            None

        Returns:
            None

        Summary:

            Prints out the initial GridWorld map to the console

            Key:
                'w' --> wall
                'a' --> agent
                'g' --> goal
                '-' --> empty
        """
        for y in reversed(range(1, self.height+1)):
            for x in range(1, self.width+1):
                if (x, y) in self.goal_locs:
                    print(' g ', end='')
                elif (x, y) in self.walls:
                    print(' w ', end='')
                elif (x, y) == self.init_loc:
                    print(' a ', end='')
                else:
                    print(' - ', end='')
            print('\n')

    def reset(self):
        if self.rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        else:
            self.cur_state = copy.deepcopy(self.init_state)

