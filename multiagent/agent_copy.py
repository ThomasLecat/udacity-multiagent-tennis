from typing import ClassVar, List, Optional

import numpy as np
import torch

from ccontrol.config import DDPGConfig
from ccontrol.environment import MultiAgentEnvWrapper
from ccontrol.model import Actor, Critic
from ccontrol.random_processes import OrnsteinUhlenbeckNoise
from ccontrol.replay_buffer import ReplayBufferInterface, SampleBatch, TorchSampleBatch
from ccontrol.utils import convert_to_torch


class DDPG:
    def __init__(
        self,
        env: MultiAgentEnvWrapper,
        config: ClassVar[DDPGConfig],
        replay_buffer: Optional[ReplayBufferInterface],
    ):
        """DDPG agent with Ornstein-Uhlenbeck noise for exploration, adapted to
        multi-agents environments (independent agents).
        """
        self.env: MultiAgentEnvWrapper = env
        self.replay_buffer: Optional[ReplayBufferInterface] = replay_buffer
        self.config: ClassVar[DDPGConfig] = config

        # Use GPU if available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Actor networks
        self.actor = Actor(env.obs_size, env.num_actions).to(self.device)
        self.actor_target = Actor(env.obs_size, env.num_actions).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = Critic(env.obs_size, env.num_actions).to(self.device)
        self.critic_target = Critic(env.obs_size, env.num_actions).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Random noise process for exploration
        self.random_process = OrnsteinUhlenbeckNoise(
            shape=env.action_space,
            mu=self.config.MU,
            theta=self.config.THETA,
            sigma=self.config.SIGMA,
            dt=self.config.DT,
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(),
            lr=config.ACTOR_LEARNING_RATE,
            weight_decay=config.ACTOR_WEIGHT_DECAY,
        )
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=config.CRITIC_LEARNING_RATE,
            weight_decay=config.CRITIC_WEIGHT_DECAY,
        )

    def compute_actions(self, observations: np.ndarray, add_noise: bool) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            # (num_agents, obs_size)
            observations = torch.from_numpy(observations).to(self.device)
            # (num_agents, num_actions)
            actions = self.actor(observations).squeeze().cpu().numpy()
        if add_noise:
            actions += self.random_process.sample()
        return actions

    def train(self, num_episodes: int) -> List:
        """Train the agent for 'num_episodes' and return the list of undiscounted
        cumulated rewards per episode (sum of rewards over all steps of the episode).
        """
        reward_per_episode: List[float] = []
        episodes_length: List[int] = []
        num_steps_sampled: int = 0

        for episode_idx in range(1, num_episodes + 1):
            # Log progress
            if episode_idx % self.config.LOG_EVERY == 0:
                window_rewards = reward_per_episode[-self.config.LOG_EVERY :]
                window_ep_len = episodes_length[-self.config.LOG_EVERY :]
                print(
                    f"episode {episode_idx}/{num_episodes}, "
                    f"avg. agent reward: {sum(window_rewards) / (len(window_rewards) * self.env.num_agents)}, "
                    f"avg. episode length: {sum(window_ep_len) / len(window_ep_len)}, "
                    f"num steps sampled: {num_steps_sampled}"
                )

            # Sample one episode
            observations = self.env.reset()
            self.random_process.reset()
            episode_length: int = 0
            episode_reward: float = 0.0
            while True:
                actions = self.compute_actions(observations, self.config.ADD_NOISE)
                next_obs, rewards, dones, _ = self.env.step(actions)
                self.replay_buffer.add(observations, actions, rewards, dones, next_obs)
                observations = next_obs
                episode_length += 1
                episode_reward += sum(rewards)
                if (
                    episode_length % self.config.UPDATE_EVERY == 0
                    and num_steps_sampled > self.config.LEARNING_STARTS
                ):
                    for _ in range(self.config.NUM_SGD_ITER):
                        self.update_once()
                # stop the episode as soon as one agent is done
                if any(dones) is True:
                    break

            reward_per_episode.append(episode_reward)
            episodes_length.append(episode_length)
            num_steps_sampled += episode_length * self.env.num_agents
        return reward_per_episode

    def update_once(self) -> None:
        """Perform one Adam update."""
        sample_batch: SampleBatch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        sample_batch: TorchSampleBatch = convert_to_torch(sample_batch, self.device)
        self.update_critic(sample_batch)
        self.update_actor(sample_batch)
        # Soft update of target Q network
        self.soft_update_target_networks()

    def update_actor(self, sample_batch: TorchSampleBatch) -> None:
        """Perform one Adam update of the actor."""
        self.actor.train()
        self.actor_optimizer.zero_grad()
        # Compute loss
        actor_loss = -self.critic(
            sample_batch.observations, self.actor(sample_batch.observations)
        ).mean()
        # Compute and apply gradient
        actor_loss.backward()
        if self.config.CLIP_GRADIENTS_ACTOR:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

    def update_critic(self, sample_batch: TorchSampleBatch) -> None:
        """Perform one Adam update on the critic."""
        self.critic_optimizer.zero_grad()
        q_loss = self.compute_q_loss(sample_batch)
        q_loss.backward()
        if self.config.CLIP_GRADIENTS_CRITIC:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

    def compute_q_loss(self, sample_batch: TorchSampleBatch) -> torch.Tensor:
        # Compute TD targets
        # (batch_size, num_actions)
        q_targets_tp1 = self.critic_target(
            sample_batch.next_observations,
            self.actor_target(sample_batch.next_observations),
        )
        td_targets = (
            sample_batch.rewards
            + (1 - sample_batch.dones) * self.config.DISCOUNT * q_targets_tp1
        ).detach()
        # Compute TD errors
        # (batch_size)
        q_values = self.critic(sample_batch.observations, sample_batch.actions)
        td_errors = q_values - td_targets
        if self.config.CLIP_TD_ERROR:
            td_errors = torch.clamp(td_errors, -1, 1)
        return torch.mean(td_errors ** 2)

    def soft_update_target_networks(self) -> None:
        """Update the target networks slowly towards the local networks.

        The magnitude of the update is determined by the config parameter
        TARGET_UPDATE_COEFF, often referred as \tau in papers.
        """
        for local_network, target_network in [
            (self.actor, self.actor_target),
            (self.critic, self.critic_target),
        ]:
            target_state_dict = target_network.state_dict()
            for param_name, param_tensor in local_network.state_dict().items():
                target_state_dict[param_name] = (
                    (1 - self.config.TARGET_UPDATE_COEFF)
                    * target_state_dict[param_name]
                    + self.config.TARGET_UPDATE_COEFF * param_tensor
                )
            target_network.load_state_dict(target_state_dict)
