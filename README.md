# Machine ToM

Code for [CompDevLab](http://compdevlab.com/) machine theory of mind project.  Currently capable 
of inferring reward functions for agents in simple grid environments.  Environments are of
the form

`
- - - - - g
- w - - - -
- w - - - -
- w - - - -
- w w w w -
a - - - - -
`
where a `-` represents an empty space, `g` represents a goal state, `w` represents a wall, and
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

`python3 forward_example.py --map="empty_map.mp" --tau=0.005`

