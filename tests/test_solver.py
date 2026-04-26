"""Tests for LP solver auto-detection."""

from __future__ import annotations

from gemfitcom.utils.solver import (
    SOLVER_PREFERENCE,
    available_solvers,
    get_best_solver,
)


def test_available_solvers_nonempty() -> None:
    # cobra ships with GLPK, so at least one solver must be present.
    assert len(available_solvers()) >= 1


def test_get_best_solver_returns_available() -> None:
    solver = get_best_solver()
    assert solver in available_solvers()


def test_get_best_solver_respects_preference() -> None:
    avail = available_solvers()
    # Build a preference list where the first entry is guaranteed available.
    expected = avail[0]
    assert get_best_solver((expected, *SOLVER_PREFERENCE)) == expected


def test_get_best_solver_falls_back_when_none_preferred() -> None:
    avail = available_solvers()
    # Supply only a clearly unavailable solver to force the fallback path.
    fake_preference = ("this_solver_does_not_exist",)
    fallback = get_best_solver(fake_preference)
    assert fallback in avail


def test_package_version_importable() -> None:
    import gemfitcom

    assert gemfitcom.__version__ == "0.1.0"
