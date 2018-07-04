# Machine ToM

Code for [CompDevLab](http://compdevlab.com/) machine theory of mind project.  Currently capable 
of inferring reward functions for agents in simple grid environments.  Environments are of
the form

```
- - - - - g
- w - - - -
- w - - - -
- w - - - -
- w w w w -
a - - - - -
```
where `-` represents an empty space, `g` represents a goal state, `w` represents a wall, and
`a` represents the agent's location.

## Requirements

* Python 3.4 or above
* Numpy
* Tensorflow 

## Usage

### Forward Model

Given a map, the forward model uses value iteration to find a path from the agent's starting
state to a goal state.  The agent is stochastic (rather than deterministic), and its behavior
can be controlled with a softmax parameter tau. Higher values of tau lead to more erratic behavior
while lower ones lead to more deterministic results.   

Example command:

`python3 forward_example.py --tau=0.005`

### Inverse Model (Naive Version)

The naive inverse model samples potential reward values from a Poisson prior.  These values are
then normalized by dividing by the largest sampled value.  This model can be run using `inverse_model_example.py`.
The script will run for a specified number of sampling iterations, saving the most likely reward
function based on a set of forward model demonstrations.  A graph of log likelihood vs time will be produced,
as well as a heat map displaying the most likely reward function.

Example command:

`python3 slow_sampler__example.py --tau=0.005 --num_demonstrations=25` 

### Inverse Model (Maximum Entropy Deep Inverse Reinforcement Learning)

Implementation of Maximum Entropy Deep Inverse Reinforcement Learning as introduced in 
[this paper](https://arxiv.org/pdf/1507.04888.pdf).  Model can be run using `inverse_model_example.py`.
Running the script will infer a reward function and produce a heat map to display it.

Example command:

`python3 inverse_model_example.py --tau=0.005 --num_demonstrations=25`
