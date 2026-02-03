"""Deterministic seeding utilities for reproducibility.

Provides a single function to set seeds across all random number generators
used in the project: Python's random, NumPy, and PyTorch.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> int:
    """Set random seeds for reproducibility across all RNGs.

    Sets seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generators (CPU and CUDA)

    Args:
        seed: The seed value to use. If None, a random seed will be generated
              and returned for later reproducibility.

    Returns:
        The seed value that was used (useful when seed=None was passed).

    Example:
        >>> from rl_poker import set_seed
        >>> set_seed(42)  # Deterministic
        42
        >>> seed = set_seed()  # Random seed, but returns it for logging
        >>> print(f"Using seed: {seed}")
    """
    if seed is None:
        # Generate a random seed if none provided
        seed = random.randint(0, 2**32 - 1)

    # Python's random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # For extra determinism in CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
