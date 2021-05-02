import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "obs",
                "action",
                "other_action",
                "reward",
                "next_obs",
                "done",
                "agent_idx",
            ],
        )
        random.seed(seed)
        self.device = device

    def add(self, obs, action, other_action, reward, next_obs, done, agent_idx):
        """Add a new experience to memory."""
        e = self.experience(
            obs, action, other_action, reward, next_obs, done, agent_idx
        )
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        observations = (
            torch.from_numpy(np.vstack([e.obs for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        other_actions = (
            torch.from_numpy(
                np.vstack([e.other_action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_observations = (
            torch.from_numpy(
                np.vstack([e.next_obs for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )
        agent_idx = (
            torch.from_numpy(
                np.vstack([e.agent_idx for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (
            observations,
            actions,
            other_actions,
            rewards,
            next_observations,
            dones,
            agent_idx,
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
