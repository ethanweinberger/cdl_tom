
import numpy as np
import Planner
import sys
import math
import PosteriorContainer
import AgentSimulation
import AuxiliaryFunctions
import scipy.misc
from scipy.stats.stats import pearsonr


class Observer(object):

    def __init__(self, A, M, Method="Linear", Validate=False):
        """
        Build an observed object

        Args:
            A (Agent): Agent object
            M (Map): Map objects
            Method (str): What type of planner? "Rate" or "Linear"
            Validate (bool): Should objects be validated?
        """

        self.Plr = Planner.Planner(A, M, Method, Validate)
        self.Validate = Validate
        # hidden variables for progress bar
        self.begincolor = '\033[91m'
        self.endcolor = '\033[0m'
        self.block = u'\u2588'

    def TestModel(self, Simulations, Samples, Return=False, Verbose=True):
        """
        Simulate N agents, infer their parameters, and then correlate the inferred values with the true values.

        Simulations (int): Number of agents to Simulate
        Samples (int): Number of samples to use in each simulation
        Return (bool): When set to true the function returns the data
        Verbose (bool): Print progress bar?
        """
        if Verbose is False and Return is False:
            sys.stdout.write(
                "ERROR: The function is set on silent and return no input.")
            return None
        if Verbose:
            sys.stdout.write("Simulating agents...\n")
            sys.stdout.flush()
        Agents = self.SimulateAgents(Simulations, False, True, False, True)
        if Verbose:
            sys.stdout.write("\n\nRunning inference...\n")
            sys.stdout.flush()
        InferredCosts = [0] * Simulations
        InferredRewards = [0] * Simulations
        for i in range(Simulations):
            if Verbose:
                Percentage = round(i * 100.0 / Simulations, 2)
                sys.stdout.write("\rInferring agent " + str(i + 1) + " |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            Results = self.InferAgent(Agents.Actions[i], Samples)
            InferredCosts[i] = Results.GetExpectedCosts()
            InferredRewards[i] = Results.GetExpectedRewards()
        if Verbose:
            # Print complete progress bar
            sys.stdout.write("\rInferring agent " + str(Simulations) + " |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
            sys.stdout.write("\n")
            # Calculate correlations
            TrueCosts = [item for sublist in Agents.Costs for item in sublist]
            TrueRewards = [
                item for sublist in Agents.Rewards for item in sublist]
            InferenceCosts = [
                item for sublist in InferredCosts for item in sublist]
            InferenceRewards = [
                item for sublist in InferredRewards for item in sublist]
            sys.stdout.write(
                "Costs correlation: " + str(pearsonr(TrueCosts, InferenceCosts)[0]) + "\n")
            sys.stdout.write(
                "Rewards correlation: " + str(pearsonr(TrueRewards, InferenceRewards)[0]) + "\n\n")
        # For each sample get the sequence of actions with the highest
        # likelihood
        if Verbose:
            sys.stdout.write(
                "Using inferred expected values to predict actions...\n")
        PredictedActions = [0] * Simulations
        MatchingActions = [0] * Simulations
        for i in range(Simulations):
            if Verbose:
                Percentage = round(i * 100.0 / Simulations, 2)
                sys.stdout.write("\rSimulating agent " + str(i + 1) + " |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            self.Plr.Agent.costs = InferredCosts[i]
            self.Plr.Agent.rewards = InferredRewards[i]
            self.Plr.Prepare(self.Validate)
            result = self.Plr.Simulate()
            PredictedActions[i] = result[0]
            if result[0] == Agents.Actions[i]:
                MatchingActions[i] = 1
        if Verbose:
            # Print complete progress bar
            sys.stdout.write("\rSimulating agent " + str(Simulations) + " |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.write(str(sum(MatchingActions) * 100.00 / Simulations) +
                             "% of inferences produced the observed actions.\n")
        if Return:
            InferredAgents = AgentSimulation.AgentSimulation(
                InferredCosts, InferredRewards, PredictedActions, None)
            return [Agents, InferredAgents, MatchingActions]
        else:
            return None

    def SetCostSamplingParams(self, samplingparams):
        """
        Set sampling parameters for costs
        """
        self.Plr.Agent.SetCostSamplingParams(samplingparams)

    def SetRewardSamplingParams(self, samplingparams):
        """
        Set sampling parameters for costs
        """
        self.Plr.Agent.SetRewardSamplingParams(samplingparams)


    def PredictPlan(self, PC, CSV=False, Feedback=False):
        """
        Return a probability distribution of the agent's plan.
        Use PredictionAction() to predict a single action.

        Args:
            PC (PosteriorContainer): PosteriorContainer object.
            Feedback (bool): When true, function gives feedback on percentage complete.
            Samples (int): Number of samples to use.
            CSV (bool): When set to true, function returns output as a csv rather than returning the values
        """
        Samples = PC.Samples
        Costs = [0] * Samples
        Rewards = [0] * Samples
        PredictedPlans = [0] * len(self.Plr.Utilities)
        # Find what samples we already have.
        RIndices = [PC.ObjectNames.index(
            i) if i in PC.ObjectNames else -1 for i in self.Plr.Map.ObjectNames]
        CIndices = [PC.CostNames.index(
            i) if i in PC.CostNames else -1 for i in self.Plr.Map.StateNames]
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Resample the agent
            self.Plr.Agent.ResampleAgent()
            # and overwrite sample sections that we already have
            self.Plr.Agent.costs = [PC.CostSamples[i, CIndices[
                j]] if CIndices[j] != -1 else self.Plr.Agent.costs[j] for j in range(len(self.Plr.Agent.costs))]
            self.Plr.Agent.rewards = [PC.RewardSamples[i, RIndices[
                j]] if RIndices[j] != -1 else self.Plr.Agent.rewards[j] for j in range(len(self.Plr.Agent.rewards))]
            # save samples
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get predicted actions
            PlanDistribution = self.Plr.GetPlanDistribution()
            # Get the probability
            probability = np.exp(PC.LogLikelihoods[i])
            # Add all up
            PredictedPlans = [PlanDistribution[
                x] * probability + PredictedPlans[x] for x in range(len(PredictedPlans))]
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        # PredictedPlans is a list of arrays. Make it a list of integers.
        #PredictedPlans = [PredictedPlans[i][0] for i in range(len(PredictedPlans))]
        if not CSV:
            return [self.Plr.goalindices, PredictedPlans]
        else:
            stringgoalindices = [str(x) for x in self.Plr.goalindices]
            print ",".join(stringgoalindices)
            probs = [str(i) for i in PredictedPlans]
            print ",".join(probs)

    def PredictAction(self, PC, CSV=False, Feedback=False):
        """
        Return a probability distribution of the agent's next action.
        Use PredictPlan() to predict the overall plan.

        Args:
            PC (PosteriorContainer): PosteriorContainer object.
            Feedback (bool): When true, function gives feedback on percentage complete.
            Samples (int): Number of samples to use.
            CSV (bool): When set to true, function returns output as a csv rather than returning the values
        """
        Samples = PC.Samples
        Costs = [0] * Samples
        Rewards = [0] * Samples
        PredictedActions = [0] * len(self.Plr.MDP.A)
        # Find what samples we already have.
        RIndices = [PC.ObjectNames.index(
            i) if i in PC.ObjectNames else -1 for i in self.Plr.Map.ObjectNames]
        CIndices = [PC.CostNames.index(
            i) if i in PC.CostNames else -1 for i in self.Plr.Map.StateNames]
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Resample the agent
            self.Plr.Agent.ResampleAgent()
            # and overwrite sample sections that we already have
            self.Plr.Agent.costs = [PC.CostSamples[i, CIndices[
                j]] if CIndices[j] != -1 else self.Plr.Agent.costs[j] for j in range(len(self.Plr.Agent.costs))]
            self.Plr.Agent.rewards = [PC.RewardSamples[i, RIndices[
                j]] if RIndices[j] != -1 else self.Plr.Agent.rewards[j] for j in range(len(self.Plr.Agent.rewards))]
            # save samples
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get predicted actions
            ActionDistribution = self.Plr.GetActionDistribution()
            # Get the probability
            probability = np.exp(PC.LogLikelihoods[i])
            # Add all up
            PredictedActions = [ActionDistribution[
                x] * probability + PredictedActions[x] for x in range(len(PredictedActions))]
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        # PredictedActions is a list of arrays. Make it a list of integers.
        PredictedActions = [PredictedActions[i][0]
                            for i in range(len(PredictedActions))]
        if not CSV:
            return [self.Plr.Map.ActionNames, PredictedActions]
        else:
            print ",".join(self.Plr.Map.ActionNames)
            probs = [str(i) for i in PredictedActions]
            print ",".join(probs)

    def InferAgent(self, ActionSequence, Samples, Feedback=False, Normalize=True):
        """
        Compute a series of samples with their likelihoods.

        Args:
            ActionSequence (list): Sequence of actions
            Samples (int): Number of samples to use
            Feedback (bool): When true, function gives feedback on percentage complete.
            Normalize (bool): Normalize log-likelihoods? When normalized the LogLikelihoods, integrated
                over matching samples give you the posterior.
        """
        ActionSequence = self.GetActionIDs(ActionSequence)
        return self.InferAgent_ImportanceSampling(ActionSequence, Samples, Normalize, Feedback)

    def GetActionIDs(self, ActionSequence):
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                return self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        return ActionSequence

    def InferAgent_ImportanceSampling(self, ActionSequence, Samples, Normalize=True, Feedback=False):
        """
        Compute a series of samples with their likelihoods using importance sampling

        Args:
            ActionSequence (list): Sequence of actions
            Samples (int): Number of samples to use
            Normalize (bool): Normalize LogLikelihoods when done?
            Feedback (bool): When true, function gives feedback on percentage complete.
        """
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                ActionSequence = self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        Costs = [0] * Samples
        Rewards = [0] * Samples
        LogLikelihoods = [0] * Samples
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Propose a new sample
            self.Plr.Agent.ResampleAgent()
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get log-likelihood
            LogLikelihoods[i] = self.Plr.Likelihood(ActionSequence)
            # If anything went wrong just stop
            if LogLikelihoods[i] is None:
                print("ERROR: Failed to compute likelihood. OBSERVER-001")
                return None
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        if Normalize:
            # Normalize LogLikelihoods
            NormalizeConst = scipy.misc.logsumexp(LogLikelihoods)
            if np.exp(NormalizeConst) == 0:
                sys.stdout.write("\nWARNING: All likelihoods are 0.\n")
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        else:
            # Hacky way because otherwise the subtraction is on different
            # object types
            NormalizeConst = scipy.misc.logsumexp([0])
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        Results = PosteriorContainer.PosteriorContainer(np.matrix(Costs), np.matrix(
            Rewards), NormLogLikelihoods, ActionSequence, self.Plr)
        if Feedback:
            sys.stdout.write("\n\n")
            Results.Summary()
            sys.stdout.write("\n")
        return Results

    def Display(self, Full=False):
        """
        Print object attributes.

        .. Internal function::

           This function is for internal use only.

        Args:
            Full (bool): When set to False, function only prints attribute names. Otherwise, it also prints its values.

        Returns:
            standard output summary
        """
        if Full:
            for (property, value) in vars(self).iteritems():
                print(property, ': ', value)
        else:
            for (property, value) in vars(self).iteritems():
                print(property)
