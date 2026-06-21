"""Shared pytest fixtures for the LevelGraphs test suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic NumPy random generator for reproducible tests."""
    return np.random.default_rng(42)
