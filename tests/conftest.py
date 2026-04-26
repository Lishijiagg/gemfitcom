"""Shared pytest fixtures for GemFitCom tests."""

from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed stdlib and numpy RNGs so tests are deterministic."""
    random.seed(0)
    np.random.seed(0)
