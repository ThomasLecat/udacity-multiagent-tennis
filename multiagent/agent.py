import random
from collections import deque
from typing import ClassVar, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from multiagent.config import DDPGConfig
from multiagent.environment import MultiAgentEnvWrapper
from multiagent.model import Actor, Critic
from multiagent.random_processes import OrnsteinUhlenbeckNoise
from multiagent.replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        env: MultiAgentEnvWrapper,
        config: ClassVar[DDPGConfig],
        random_seed: Optional[int],
    ):
        """

        Params
        ======
            env (MultiAgentWrapper): wrapped unity environment
            config (ClassVar[DDPGConfig]): hyperparameters
            random_seed (int): random seed
        """
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.env: MultiAgentEnvWrapper = env
        self.config: ClassVar[DDPGConfig] = config

        # Actor(s)
        self.actor = Actor(env.obs_size, env.num_actions).to(device)
        self.actor_target = Actor(env.obs_size, env.num_actions).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic(s)
        self.critic = Critic(env.obs_size, env.num_actions, env.num_agents).to(device)
        self.critic_target = Critic(env.obs_size, env.num_actions, env.num_agents).to(
            device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.ACTOR_LEARNING_RATE
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.CRITIC_LEARNING_RATE,
            weight_decay=config.CRITIC_WEIGHT_DECAY,
        )

        # Noise process
        self.noise = OrnsteinUhlenbeckNoise(
            (env.num_agents, env.num_actions), config.MU, config.THETA, config.SIGMA
        )
        self.eps = config.EPS_START
        self.t_step = 0

        # Replay memory
        self.replay_buffer = ReplayBuffer(
            env.num_actions, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, device
        )

    def compute_actions(self, observations, add_noise):
        """Returns actions for given state as per current policy."""
        self.actor.eval()
        observations = torch.from_numpy(observations).float().to(device)
        with torch.no_grad():
            actions = self.actor(observations).cpu().numpy()
        if add_noise:
            actions += self.eps * self.noise.sample()
            actions = np.clip(actions, -1, 1)
        self.actor.train()
        return actions

    def train(self, num_episodes: int) -> List[float]:
        reward_per_episode: List[float] = []
        scores_window = deque(maxlen=100)  # last 100 scores

        for episode_idx in range(1, num_episodes + 1):
            if episode_idx % self.config.LOG_EVERY == 0:
                print(
                    "Episode {}\tAverage Reward: {:.3f}".format(
                        episode_idx, np.mean(scores_window)
                    )
                )
            # Sample one episode
            scores = np.zeros(self.env.num_agents)
            self.noise.reset()

            observations = self.env.reset(train_mode=True)
            while True:
                actions = self.compute_actions(observations, self.config.ADD_NOISE)
                next_state, rewards, done, _ = self.env.step(actions)
                self.step(observations, actions, rewards, next_state, done)

                observations = next_state
                scores += rewards
                if np.any(done):
                    break
            scores_window.append(np.max(scores))
            reward_per_episode.append(np.max(scores))

            # Early stopping
            if np.mean(scores_window) >= 0.5:
                print(
                    "\nEnvironment solved in {:d} episodes,\tAverage Score: {:.3f}".format(
                        episode_idx - 100, np.mean(scores_window)
                    )
                )
                break

        return reward_per_episode

    def step(self, observations, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.
        """
        self.t_step += 1
        # Save transitions in replay buffer
        for agent_idx, (obs, act, rew, next_obs, done) in enumerate(
            zip(observations, actions, rewards, next_states, dones)
        ):
            other_act = actions[1 - agent_idx]
            self.replay_buffer.add(obs, act, other_act, rew, next_obs, done, agent_idx)

        # Learn, if enough samples are available in memory and at interval settings
        if len(self.replay_buffer) > self.config.BATCH_SIZE:
            if self.t_step % self.config.UPDATE_EVERY == 0:
                for _ in range(self.config.NUM_SGD_ITER):
                    experiences = self.replay_buffer.sample()
                    self.learn(experiences, self.config.DISCOUNT)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        observations, actions, other_actions, rewards, next_states, dones, agent_idx = (
            experiences
        )

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions_pred = self.actor_target(next_states)

        # next_actions_all = torch.cat(
        #     (
        #         next_actions_pred * (1 - agent_idx) + other_actions * agent_idx,
        #         next_actions_pred * agent_idx + other_actions * (1 - agent_idx),
        #     ),
        #     dim=1,
        # )

        q_targets_next = self.critic_target(
            next_states, next_actions_pred, other_actions
        )
        # Compute Q targets for current observations (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic(observations, actions, other_actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(observations)
        # all_actions = torch.cat(
        #     (
        #         actions_pred * (1 - agent_idx) + other_actions * agent_idx,
        #         actions_pred * agent_idx + other_actions * (1 - agent_idx),
        #     ),
        #     dim=1,
        # )

        actor_loss = -self.critic(observations, actions_pred, other_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(
            self.critic, self.critic_target, self.config.TARGET_UPDATE_COEFF
        )
        self.soft_update(self.actor, self.actor_target, self.config.TARGET_UPDATE_COEFF)

        # Update epsilon noise value
        self.eps = self.eps - (1 / self.config.EPS_DECAY)
        if self.eps < self.config.EPS_END:
            self.eps = self.config.EPS_END

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
