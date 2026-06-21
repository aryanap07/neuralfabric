"""Global configuration: default dtype, random seed, device."""

DEFAULT_DTYPE = "float32"
RANDOM_SEED = 42
DEFAULT_DEVICE = "cpu"


def set_seed(seed: int) -> None:
    """Set the global random seed for reproducibility."""
    global RANDOM_SEED
    RANDOM_SEED = seed
    import numpy as np

    np.random.seed(seed)
