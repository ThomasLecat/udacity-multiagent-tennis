import argparse
import time

import torch
from unityagents import UnityEnvironment

from multiagent.agent import DDPG
from multiagent.config import DDPGConfig
from multiagent.environment import MultiAgentEnvWrapper
from multiagent.preprocessors import IdentityPreprocessor, PreprocessorInterface


def evaluate(
    environment_path: str, checkpoint_path: str, show_graphics: bool, seed: int
) -> None:
    """Play one episode with the specified actor checkpoint of the trained DDPG agent."""
    preprocessor: PreprocessorInterface = IdentityPreprocessor()
    env = UnityEnvironment(environment_path, no_graphics=not show_graphics)
    env = MultiAgentEnvWrapper(env, preprocessor, skip_frames=DDPGConfig.SKIP_FRAMES)
    agent = DDPG(env=env, config=DDPGConfig, random_seed=seed)
    # Load saved model
    agent.actor.load_state_dict(torch.load(checkpoint_path))

    # Play episode
    episode_reward: float = 0.0
    episode_length: int = 0
    observations = env.reset()
    while True:
        if show_graphics:
            time.sleep(0.05)
        actions = agent.compute_actions(observations, add_noise=False)
        observations, rewards, dones, _ = agent.env.step(actions)
        episode_reward += sum(rewards)
        episode_length += 1
        if dones is True:
            break
    print(
        f"Episode finished in {episode_length} steps, "
        f"mean agent reward: {episode_reward / env.num_agents}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment_path",
        "-p",
        type=str,
        help="Path to your single agent Unity environment file.",
    )
    parser.add_argument(
        "--checkpoint_path", "-c", type=str, help="Path to the PyTorch checkpoint"
    )
    parser.add_argument(
        "--show_graphics",
        "-g",
        type=bool,
        default=True,
        help="Visualize the agent playing on the environment",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Random seed for initialization of NN and random processes",
    )
    args = parser.parse_args()
    evaluate(args.environment_path, args.checkpoint_path, args.show_graphics, args.seed)
