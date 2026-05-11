"""Spatial-only pytest fixtures.

Inherits RNG seeding from tests/conftest.py automatically.
"""

import pytest


@pytest.fixture(scope="session")
def textbook_model():
    """Load the cobra textbook (E. coli core) model once per session."""
    import cobra

    return cobra.io.load_model("textbook")


@pytest.fixture
def fresh_textbook(textbook_model):
    """A deep copy of the textbook model, safe to mutate per test."""
    return textbook_model.copy()
