import abc

import numpy as np


class PreprocessorInterface:
    @abc.abstractmethod
    def transform(self, raw_observation: np.ndarray) -> np.ndarray:
        """Preprocess the raw observation into an array to feed to the NN."""

    @abc.abstractmethod
    def observation_size(self, raw_obs_size) -> int:
        """Return the shape of the observation after preprocessing."""


class IdentityPreprocessor(PreprocessorInterface):
    """Return raw observations."""

    def transform(self, observation: np.ndarray) -> np.ndarray:
        return observation

    def observation_size(self, raw_obs_size) -> int:
        return raw_obs_size
