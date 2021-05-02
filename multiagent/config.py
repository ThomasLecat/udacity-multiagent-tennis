from ccontrol.types import NumberOfSteps


class DDPGConfig:
    # Sampling
    SKIP_FRAMES: int = 1

    # Ornstein-Uhlenbeck noise generator
    ADD_NOISE: bool = True
    MU: float = 0.0
    THETA: float = 0.13
    SIGMA: float = 0.2
    EPS_START = 6
    EPS_END = 0
    EPS_DECAY = 250  # Number of episodes to decay over

    # Optimisation
    BUFFER_SIZE: int = 1_000_000
    BATCH_SIZE: int = 128
    DISCOUNT: float = 0.99
    ACTOR_LEARNING_RATE: float = 1e-3
    CRITIC_LEARNING_RATE: float = 1e-3
    CRITIC_WEIGHT_DECAY: float = 0.0
    TARGET_UPDATE_COEFF: float = 6e-2
    UPDATE_EVERY: NumberOfSteps = 1
    NUM_SGD_ITER: int = 1

    # Logging
    LOG_EVERY: NumberOfSteps = 10

    def __setattr__(self, key, value):
        raise AttributeError("Config objets are immutable")
