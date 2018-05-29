import MDP
import numpy
import copy
import math
import random

#TODO: How to fit agents in?
#TODO: How to fit maps in?
#TODO: What is the "method" parameter?
#TODO: What are critical states?
#TODO: What is planningreward?

class Planner(object):
    
    def __init__(self, Agent, Map, Method="Linear", Validate=True):
        """
        Build a planner object.

        """

        self.Method = Method
        self.Agent = Agent
        self.Map = Map
        self.MDP = []

        self.Policies = []
        self.CriticalStates = [] 
        self.CostMatrix = []
        self.planningreward = 500
    
        self.gamma = 0.95
        self.Prepare(Validate)

    def Prepare(self, Validate=True):
        """
        Runs the planner and builds the utility function.
        Function is mostly for readability of other code blocks.

        Args:
            Validate (bool): Run validation?

        Returns:
            None
        """

        try:
            self.BuildPlanner(Validate)
            self.ComputeUtilities()
        except Exception as error:
            print(error)
    


