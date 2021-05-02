import argparse

import torch
from unityagents import UnityEnvironment

from multiagent.agent import DDPG
from multiagent.config import DDPGConfig
from multiagent.environment import MultiAgentEnvWrapper
from multiagent.preprocessors import IdentityPreprocessor
from multiagent.utils import write_list_to_csv


def train(environment_path: str, num_episodes: int, seed: int):
    """Train the agent for 'num_episodes', save the score for each training episode
    and the checkpoint of the trained agent.
    """
    config = DDPGConfig
    preprocessor = IdentityPreprocessor()
    env = UnityEnvironment(environment_path, no_graphics=True)
    env = MultiAgentEnvWrapper(env, preprocessor, skip_frames=config.SKIP_FRAMES)

    agent = DDPG(env=env, config=config, random_seed=seed)
    reward_per_episode = agent.train(num_episodes=num_episodes)
    with open("mean_reward_per_episode.csv", "w") as f:
        write_list_to_csv(f, reward_per_episode)
    with open("ddpg_actor_checkpoint.pt", "wb") as f:
        torch.save(agent.actor.state_dict(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment_path",
        "-p",
        type=str,
        help="Path to your single agent Unity environment file.",
    )
    parser.add_argument(
        "--num_episodes",
        "-n",
        type=int,
        default=1000,
        help="Number of episodes on which to train the agent",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for initialization of NN and random processes",
    )
    args = parser.parse_args()
    train(args.environment_path, args.num_episodes, args.seed)
