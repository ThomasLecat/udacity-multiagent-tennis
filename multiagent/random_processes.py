import numpy as np


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process, used to add noise to the action to increase
    exploration.
    """

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (
            self.mu - self.state
        ) + self.sigma * np.random.standard_normal(self.size)
        self.state += dx
        return self.state
