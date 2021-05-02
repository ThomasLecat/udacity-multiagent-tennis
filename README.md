# Udacity Multiagent Tennis

This repository implements a multi-agent version of [DDPG](https://arxiv.org/abs/1509.02971) 
that solves the two player Unity environment _Tennis_. 

Our implementation follows the centralized training and decentralized execution 
paradigm. We share the weights of the actor and critic models between agents and 
include other agent's actions and observations as input to the critic network.

## Installation

This project uses the drlnd conda environment from the Udacity Deep Reinforcement
Learning program.

1. Follow the instructions from Udacity's [README](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
to create the environment and install the dependencies.
1. Install the project's package: `$ source activate drlnd && pip install -e .`
1. Download the RL environment for your OS, place the file in the `multiagent/` directory 
and unzip (or decompress) it. 

*  Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
*  Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
*  Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
*  Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

*(Optional)* To contribute, install the pre-commits:

```bash
$ pre-commit install
```

## Usage

Before training or evaluating an agent, make sure you conda environment is activated:
```
$ source activate drlnd
```

### Training

1. Tune DDPG's learning parameters in `multiagent/config.py`
2. run `python multiagent/train.py --environment_path /path/to/Tennis.app`. You can 
also specify the number of training episodes with the argument `--num_episodes` and 
a random seed with the argument `--seed`.

At the end of training, two files are saved on disk:
*  `ddpg_actor_checkpoint.pt`: PyTorch checkpoint containing the weights of the trained actor network.
*  `reward_per_episode.csv`: score of all training episodes.

### Evaluation

Using the same config parameters as in training, run:
```
python multiagent/evaluate.py --environment_path /path/to/Tennis.app --checkpoint_path ddpg_actor_checkpoint.pt --show_graphics True
```

## Description of the environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent 
hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit 
the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal 
of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity 
of the ball and racket. Each agent receives its own, local observation. Two continuous 
actions are available, corresponding to movement toward (or away from) the net, and 
jumping.

The task is episodic, and in order to solve the environment, your agents must get an 
average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both 
agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), 
to get a score for each agent. This yields 2 (potentially different) scores. We then 
take the maximum of these 2 scores. This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those 
scores is at least +0.5.

![Screenshot of Tennis environment](doc/tennis.png)
