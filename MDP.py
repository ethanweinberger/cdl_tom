#Much of this code adapted from https://github.com/julianje/Bishop

import numpy as np
import math
import random

class MDP(object):

    def __init__(self, S=[], A=[], T=[], R=[], gamma=0.95, tau=0.01)
        """
        Class for representing a Markov Decision Process

        Args:
            S (list): List of states
            A (list): List of actions
            T (matrix): Transition matrix where T[S0, A, SF] is
                        the probability of moving from S0 to SF
                        after taking action A
            R (matrix): Reward function where R[A,S] is the reward
                        for taking action A in state S
            gamma (float): Future discount
            tau (float): Softmax parameter

        Returns:
            MDP object
        """

        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma
        self.tau = tau

        #Placeholder to be filled in by Value Iteration algorithm
        self.values = np.zeros((1, len(S)))

        #Store softmaxed probabilities here
        self.policy = np.zeros((len(A), len(S)))

    def ValueIteration(self, epsilon=0.001):
        """
        Update each state's expected value and store them in MDP values field

        Args:
            epsilon (float): Convergence parameter

        Returns:
            None (values updated in place)
        """
        self.values = np.zeros(self.values.shape)
        while True:
            V2 = self.values.copy()
            for i in range(0, len(self.S)):
                prod = self.gamma * \
                        (np.mat(self.T[i, :, :]) * np.mat(V2.transpose()))
                self.values[0, i] = max(prod[j] + self.R[j, i]
                                        for j in range(len(self.A)))
            if (self.values - V2).max() <= epsilon:
                break

    def BuildPolicy(self, Softmax = True):
        """
        Builds optimal policy

        Args:
            Softmax (bool): Indicates if actions are softmaxed

        Returns:
            None (modifies self.policy in place)
        """

        for i in range(0, len(self.S)):
            options = np.mat(self.T[i, :, :]) * np.mat(self.values.transpose())
            options = options.tolist()

            maxval = abs(max(options)[0])
            options = [options[j][0] - maxval for j in range(len(options))]

            if Softmax:
                try:
                    options = [math.exp(options[j] / self.tau)
                                for j in range(len(options))]
                except OverflowError:
                    print("ERROR: Failed to softmax policy.")
                    raise

                #Set a uniform distribution if all actions have no value
                if sum(options) == 0:
                    self.policy[:, i] = [
                        1.0 / len(options) for j in range(len(options))]
                else:
                    self.policy[:, i] = [
                        options[j] / sum(options) for j in range(len(options))]
            else:
                totalchoices = sum([options[optloop] == max(options)
                                    for optloop in range(len(options))])
                self.policy[:, i] = [(1.0 / totalchoices if options[optloop] == max(options) else 0)
                                        for optloop in range(len(options))]

    def GetStates(self, StartingPoint, ActionSequence):
        """
        TODO: Understand what this is used for

        Given a starting point and sequence of actions,
        produce the set of states with highest likelihood

        Args:
            StartingPoint (int): State number where agent begins
            ActionSequence (list): List of indices of actions

        Returns:
            List of State Numbers
        """

        StateSequence = [0] * (len(ActionSequence) + 1)
        StateSequence[0] = StartingPoint
        for i in range(len(ActionSequence)):
            StateSequence[i + 1] = (
                self.T[StateSequence[i], ActionSequence[i], :].argmax())
        return StateSequence
    
    def Run(self, State, Softmax=False, Simple=False):
        """
        TODO: Understand how this works

        Sample an action from optimal policy given the state
        If softmax is set to true then Simple is ignored

        Args:
            State (int): State number where agent begins
            Softmax (bool): Simulate with softmaxed policy?
            Simple (bool): Some states have various actions all with an equally high value.
                           when this happens, Run() randomly selects one of these actions.
                           if Simple is set to True, it selects the first highest-value action.
	
	Returns:
            List of state numbers
        """
        if Softmax:

            ActSample    = random.uniform(0,1)
            ActionProbs  = self.policy[:, State]
            ActionChoice = -1
            for j in range(len(ActionProbs)):
                if ActSample < ActionProbs[j]:
                    ActionChoice = j
                    break
                else:
                    ActSample -= ActionProbs[j]
        else:
            maxval     = max(self.policy[:, State])
            maxindices = [
                i for i, j in enumerate(self.policy[:, State]) if j == maxval]
            if Simple:
                ActionChoice = maxindices[0]
            else:
                ActionChoice = random.choice(maxindices)

        EndStates = self.T[State, ActionChoice, :]
        StateSample = random.uniform(0, 1)

        for j in range(len(Endstates)):
            if StateSample < EndStates[j]:
                EndState = j
                break
            else:
                StateSample -= EndStates[j]
        return [EndState, ActionChoice]
