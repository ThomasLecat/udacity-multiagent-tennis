import argparse
import time

import torch
from unityagents import UnityEnvironment

from ccontrol.agent import DDPG
from ccontrol.config import DDPGConfig
from ccontrol.environment import MultiAgentEnvWrapper
from ccontrol.preprocessors import IdentityPreprocessor, PreprocessorInterface


def evaluate(environment_path: str, checkpoint_path: str, show_graphics: bool) -> None:
    """Play one episode with the specified actor checkpoint of the trained DDPG agent."""
    preprocessor: PreprocessorInterface = IdentityPreprocessor()
    env = UnityEnvironment(environment_path, no_graphics=not show_graphics)
    env = MultiAgentEnvWrapper(env, preprocessor, skip_frames=DDPGConfig.SKIP_FRAMES)
    agent = DDPG(env=env, config=DDPGConfig, replay_buffer=None)
    # Load saved model
    agent.actor = torch.load(checkpoint_path)

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
        "-s",
        type=bool,
        default=True,
        help="Visualize the agent playing on the environment",
    )
    args = parser.parse_args()
    evaluate(args.environment_path, args.checkpoint_path, args.show_graphics)
