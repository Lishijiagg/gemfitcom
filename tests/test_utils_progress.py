"""Tests for utils.progress — tqdm wrapper behavior and no-op fast path."""

from __future__ import annotations

import sys

import pytest

from gemfitcom.utils.progress import progress_bar


def test_disabled_returns_iterable_unchanged() -> None:
    """enabled=False must not even import tqdm — it returns the iterable as-is."""
    src = [1, 2, 3]
    wrapped = progress_bar(src, enabled=False)
    # Same object — proves zero-overhead fast path.
    assert wrapped is src


def test_disabled_does_not_import_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm the lazy import only happens when enabled=True."""
    # Remove any previously-loaded tqdm so we can observe a fresh import.
    for mod in list(sys.modules):
        if mod == "tqdm" or mod.startswith("tqdm."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    wrapped = progress_bar(iter(range(3)), enabled=False)
    assert "tqdm" not in sys.modules
    # Iterable still works.
    assert list(wrapped) == [0, 1, 2]


def test_enabled_wraps_iterable_and_yields_same_values() -> None:
    src = [10, 20, 30]
    out = list(progress_bar(src, enabled=True, desc="test", total=len(src)))
    assert out == src


def test_enabled_accepts_generator_with_total() -> None:
    def gen():
        yield from range(5)

    out = list(progress_bar(gen(), enabled=True, total=5, desc="gen-test"))
    assert out == [0, 1, 2, 3, 4]


# ---------- integration with simulation loops ----------


def test_mono_dfba_progress_true_does_not_break(capsys: pytest.CaptureFixture) -> None:
    """progress=True must not change the result, only add a bar."""
    from cobra import Metabolite, Model, Reaction

    from gemfitcom.kinetics.mm import MMParams
    from gemfitcom.kinetics.mono_dfba import simulate_mono_dfba
    from gemfitcom.medium import Medium, apply_medium

    m = Model("toy")
    glc = Metabolite("glc__D_e", compartment="e")
    bio = Metabolite("biomass_c", compartment="c")
    ex = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex.add_metabolites({glc: -1})
    br = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    br.add_metabolites({glc: -10.0, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex, br, sink])
    m.objective = "BIOMASS"

    medium = Medium(
        name="toy",
        pool_components={"EX_glc__D_e": 10.0},
        unlimited_components=frozenset(),
    )
    apply_medium(m, medium, close_others=False)

    res_quiet = simulate_mono_dfba(
        m,
        medium,
        {"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        initial_biomass=0.01,
        t_total=2.0,
        dt=0.5,
        progress=False,
    )

    # Rebuild medium state for a clean second run.
    apply_medium(m, medium, close_others=False)
    res_loud = simulate_mono_dfba(
        m,
        medium,
        {"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        initial_biomass=0.01,
        t_total=2.0,
        dt=0.5,
        progress=True,
    )

    assert res_quiet.biomass[-1] == pytest.approx(res_loud.biomass[-1], rel=1e-6)


def test_sequential_dfba_progress_true_does_not_break() -> None:
    from cobra import Metabolite, Model, Reaction

    from gemfitcom.kinetics.mm import MMParams
    from gemfitcom.medium import Medium, apply_medium
    from gemfitcom.simulate import StrainSpec, simulate_sequential_dfba

    def toy(name: str) -> Model:
        m = Model(name)
        glc = Metabolite("glc__D_e", compartment="e")
        bio = Metabolite("biomass_c", compartment="c")
        ex = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
        ex.add_metabolites({glc: -1})
        br = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
        br.add_metabolites({glc: -10.0, bio: 1.0})
        sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
        sink.add_metabolites({bio: -1})
        m.add_reactions([ex, br, sink])
        m.objective = "BIOMASS"
        return m

    medium = Medium(
        name="toy",
        pool_components={"EX_glc__D_e": 10.0},
        unlimited_components=frozenset(),
    )

    m1, m2 = toy("A"), toy("B")
    apply_medium(m1, medium, close_others=False)
    apply_medium(m2, medium, close_others=False)
    strains = [
        StrainSpec(
            name="A", model=m1, mm_params={"EX_glc__D_e": MMParams(5.0, 1.0)}, initial_biomass=0.01
        ),
        StrainSpec(
            name="B", model=m2, mm_params={"EX_glc__D_e": MMParams(5.0, 1.0)}, initial_biomass=0.01
        ),
    ]
    res = simulate_sequential_dfba(strains, medium, t_total=1.0, dt=0.25, progress=True)
    assert res.biomass["A"].iloc[-1] >= res.biomass["A"].iloc[0]
    assert res.biomass["B"].iloc[-1] >= res.biomass["B"].iloc[0]
