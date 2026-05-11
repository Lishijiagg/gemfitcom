# Spatial dFBA PR 2: Kinetics + Reaction + Serial Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Couple the PR 1 numerical core to genome-scale metabolic models. Each grid cell solves an FBA per species with Michaelis–Menten-bounded uptake rates from local substrate concentrations, returning per-species growth rates and per-metabolite per-species exchange fluxes. Ships a Serial backend (single process) and a skeleton example demonstrating end-to-end reaction stepping; full diffusion-coupled `Simulator` lands in PR 3.

**Architecture:** Three new modules in `gemfitcom.spatial` — `kinetics.py` (MM-bound calculator per species), `reaction.py` (`ReactionEngine` orchestrating per-cell × per-species FBA), `backends.py` (`SerialBackend` strategy). Extends `config.py` with `SpeciesConfig` and a `cobra://` URI scheme for built-in cobra models. Run-time errors (infeasible FBA, exponential overflow) are warn-and-continue; startup errors (missing exchanges, malformed YAML) are fail-fast.

**Tech Stack:** Python ≥3.10, NumPy ≥1.24, cobra ≥0.29 (already core dep), pydantic ≥2.5, pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-05-spatial-dfba-v0.1-design.md` §3.1 (ExchangeKinetics, ReactionEngine), §3.2 (single-step algorithm — reaction half), §3.3 (Serial backend), §5.3 (species + kinetics YAML schema), §5.4 (kinetics YAML), §6 (error handling), §7.2 (unit tests for kinetics + reaction), §8 PR 2.

**Estimated work:** ~700 LOC core + ~700 LOC tests; 2.5–3.5 weeks at 1–2 h/day.

**Out of scope (live in later PRs):**
- `Simulator` time loop and `diffuse → react → boundary` orchestration → PR 3
- CLI subcommand (`gemfitcom spatial run`) → PR 3
- `JoblibBackend`, quantization `cache.py` → PR 4
- `viz.py` → PR 5
- `well_mixed_limit` and `pipeline_e2e` invariants → PR 3 (need Simulator)

---

## Key Design Decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| K1 | Per-species kinetics container | `ExchangeKinetics` dataclass (one per species, holds list of `(exchange_id, vmax, km, mode)`) | Mirrors design doc §3.1; cleanly decoupled from cobra `Model` so it can be inspected without loading SBML |
| K2 | `mm_upper_bound(C_local)` return shape | `ndarray(n_exchanges,)` ordered by the species's own exchange list | Allows direct vectorised assignment into `model.reactions[exch_id].lower_bound` without dict lookup in the inner loop |
| K3 | Exchange ↔ metabolite mapping | Convention `EX_<met_id>` resolved at engine build time | BiGG-style; same convention used by `gemfitcom.medium.constraints`. Cached per species at `ReactionEngine.__init__` so the hot loop is index-only |
| K4 | `flux_field` shape | `(n_metabolites, n_species, n_grid)`, units mmol/gDW/h | Direct match to `np.einsum('jix,ix->jx', flux, B)` in spec §3.2 |
| K5 | Sign convention | Cobra exchange flux sign passthrough: `+` = secretion (medium gains), `−` = uptake (medium loses) | Matches cobra semantics; no flip needed |
| K6 | Cells with B[i,x] < ε skipped | ε = 1e-12 gDW/L, configurable on `SerialBackend` | Avoid pointless LP solves for empty cells; ε threshold tested against well-mixed reference |
| K7 | GEM URI scheme | Support both filesystem paths and `cobra://<name>` (e.g. `cobra://textbook`) for built-in cobra-shipped models | Lets the skeleton example run with zero external SBML downloads |
| K8 | FBA infeasible per cell | `mu=0, flux=0`, append to `ReactionEngine.warnings`; never raise | Spec §6 run-time policy |
| K9 | mu·dt overflow guard | If `mu * dt > 5`, clip mu and warn (exp(5)≈148× growth in one step is non-physical) | Spec §6 |
| K10 | `bidirectional` mode | Apply MM bound to *both* `lower_bound` (uptake) and `upper_bound` (secretion) | Spec §5.4 |
| K11 | No `Simulator` in PR 2 | Smoke test orchestrates reaction step inline | Simulator (with diffusion+timestepping+recorder coupling) is PR 3 |

---

## File Manifest

### New files

| Path | Responsibility |
|---|---|
| `src/gemfitcom/spatial/kinetics.py` | `ExchangeKinetics` dataclass, kinetics YAML loader, GEM URI resolver |
| `src/gemfitcom/spatial/reaction.py` | `ReactionEngine`, per-cell FBA primitive, exchange↔metabolite mapping |
| `src/gemfitcom/spatial/backends.py` | `SerialBackend` (and `Backend` protocol for PR 4 to plug `JoblibBackend` into) |
| `src/gemfitcom/spatial/_init_fields.py` | Private helper: build 1D init fields from `InitConfig`/`SpeciesInitConfig` |
| `tests/spatial/test_kinetics.py` | MM upper bound monotonicity / boundary / negative defence; YAML round-trip |
| `tests/spatial/test_reaction_serial.py` | Single-cell dFBA equality with bare cobra; SerialBackend over multi-cell grid |
| `tests/spatial/test_reaction_e2e.py` | 10-step smoke: 3-grid × 2-species, real cobra textbook model, no diffusion |
| `tests/spatial/test_init_fields.py` | Init-field builder unit tests |
| `examples/spatial/ecoli_fprau_skeleton/sim.yaml` | Skeleton config using `cobra://textbook` for both species |
| `examples/spatial/ecoli_fprau_skeleton/kinetics/ecoli.yaml` | E. coli core kinetics |
| `examples/spatial/ecoli_fprau_skeleton/kinetics/fprau.yaml` | "Fprau" placeholder kinetics (also textbook model) |
| `examples/spatial/ecoli_fprau_skeleton/README.md` | How to run the example |

### Modified files

| Path | Change |
|---|---|
| `src/gemfitcom/spatial/config.py` | Add `SpeciesConfig`, `SpeciesInitConfig`; wire `species` field into `SpatialConfig` |
| `src/gemfitcom/spatial/__init__.py` | Export PR 2 surface: `ExchangeKinetics`, `ReactionEngine`, `SerialBackend`, `SpeciesConfig`, `load_kinetics_yaml`, `resolve_gem`, `build_field_1d` |
| `tests/spatial/conftest.py` | Add fixtures: `textbook_model` (session), `fresh_textbook` |
| `tests/spatial/test_public_api.py` | Add `TestPr2PublicApi` assertions |
| `tests/spatial/test_config.py` | Add `TestSpeciesConfig` assertions |

### No-touch (PR 1 outputs, depended on but not modified)

`state.py`, `geometry.py`, `diffusion.py`, `recorder.py`.

---

## Task 1: Pre-flight — verify cobra dep + branch + import smoke

**Files:** none modified. Pure environment check.

- [ ] **Step 1: Confirm we're on the PR 2 feature branch**

Run:
```bash
git status
git rev-parse --abbrev-ref HEAD
```

Expected output:
```
On branch spatial/pr2-kinetics-reaction
nothing to commit, working tree clean
spatial/pr2-kinetics-reaction
```

If not on `spatial/pr2-kinetics-reaction`, abort and re-run setup.

- [ ] **Step 2: Verify cobra is already importable in the dev env**

Run:
```bash
python -c "import cobra; print(cobra.__version__); m = cobra.io.load_model('textbook'); print(m.id, len(m.reactions), len(m.metabolites), len(m.exchanges))"
```

Expected: prints cobra version (≥0.29) and `e_coli_core 95 72 20` (or close — exact counts may vary by cobra version).

If cobra fails to import, run `pip install -e ".[spatial,dev]"` and retry.

- [ ] **Step 3: Run the existing PR 1 test suite to establish a green baseline**

Run:
```bash
pytest tests/spatial/ -v
```

Expected: all PR 1 spatial tests pass. Record the count (e.g. "9 passed in 4.32s") — Task 14 will assert no regressions.

- [ ] **Step 4: No commit yet — this is a verification step.**

---

## Task 2: `ExchangeKinetics` dataclass + per-species MM upper bound

**Files:**
- Create: `src/gemfitcom/spatial/kinetics.py`
- Create: `tests/spatial/test_kinetics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spatial/test_kinetics.py`:

```python
"""Tests for spatial/kinetics.py — ExchangeKinetics + YAML loader + GEM URI."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.kinetics import ExchangeEntry, ExchangeKinetics


class TestExchangeKineticsBasics:
    def test_construct_with_two_substrates(self):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry(exchange_id="EX_glc__D_e", vmax=10.0, km=0.5, mode="uptake_only"),
                ExchangeEntry(exchange_id="EX_o2_e", vmax=15.0, km=0.005, mode="uptake_only"),
            ),
        )
        assert ek.species == "ecoli"
        assert ek.n_exchanges == 2
        assert ek.exchange_ids == ("EX_glc__D_e", "EX_o2_e")

    def test_mm_upper_bound_at_saturation(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(
                ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
            ),
        )
        bounds = ek.mm_upper_bound(np.array([1000.0]))
        assert bounds.shape == (1,)
        assert np.isclose(bounds[0], 10.0, rtol=1e-3)

    def test_mm_upper_bound_at_zero(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(
                ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
            ),
        )
        bounds = ek.mm_upper_bound(np.array([0.0]))
        assert bounds[0] == 0.0

    def test_mm_upper_bound_monotonic(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(
                ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
            ),
        )
        C = np.array([0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
        bounds = np.array([ek.mm_upper_bound(np.array([c]))[0] for c in C])
        assert np.all(np.diff(bounds) > 0)

    def test_mm_upper_bound_negative_input_clipped(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(
                ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
            ),
        )
        bounds = ek.mm_upper_bound(np.array([-1e-9]))
        assert bounds[0] == 0.0

    def test_mm_upper_bound_shape_mismatch_raises(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(
                ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
                ExchangeEntry(exchange_id="EX_b_e", vmax=20.0, km=1.0, mode="uptake_only"),
            ),
        )
        with pytest.raises(ValueError, match="length"):
            ek.mm_upper_bound(np.array([0.5]))

    def test_invalid_vmax_rejected(self):
        with pytest.raises(ValueError, match="vmax"):
            ExchangeEntry(exchange_id="EX_a_e", vmax=0.0, km=0.5, mode="uptake_only")

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="weird_mode")

    def test_duplicate_exchange_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ExchangeKinetics(
                species="x",
                entries=(
                    ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
                    ExchangeEntry(exchange_id="EX_a_e", vmax=20.0, km=1.0, mode="uptake_only"),
                ),
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_kinetics.py -v
```

Expected: `ModuleNotFoundError: No module named 'gemfitcom.spatial.kinetics'`.

- [ ] **Step 3: Implement `kinetics.py` to make these tests pass**

Create `src/gemfitcom/spatial/kinetics.py`:

```python
"""Per-species Michaelis–Menten kinetics for spatial dFBA.

Wraps the lower-level :mod:`gemfitcom.kinetics.mm` MM formula into a
per-species container ``ExchangeKinetics`` that produces the upper bound on
substrate uptake at a given local concentration vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gemfitcom.kinetics.mm import michaelis_menten

ExchangeMode = Literal["uptake_only", "bidirectional"]
_VALID_MODES: tuple[ExchangeMode, ...] = ("uptake_only", "bidirectional")


@dataclass(frozen=True, slots=True)
class ExchangeEntry:
    """Single substrate's MM parameters for one species.

    Attributes:
        exchange_id: Cobra exchange reaction id (e.g. ``"EX_glc__D_e"``).
        vmax: Maximum uptake rate (mmol / gDW / h). Must be > 0.
        km: Half-saturation concentration (mM). Must be > 0.
        mode: ``"uptake_only"`` writes only the lower bound; ``"bidirectional"``
            also caps the upper bound (secretion) with the same MM expression.
    """

    exchange_id: str
    vmax: float
    km: float
    mode: ExchangeMode = "uptake_only"

    def __post_init__(self) -> None:
        if self.vmax <= 0:
            raise ValueError(f"{self.exchange_id}: vmax must be > 0, got {self.vmax}")
        if self.km <= 0:
            raise ValueError(f"{self.exchange_id}: km must be > 0, got {self.km}")
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"{self.exchange_id}: mode must be one of {_VALID_MODES}, got {self.mode!r}"
            )


@dataclass(frozen=True, slots=True)
class ExchangeKinetics:
    """All exchange kinetics for a single species."""

    species: str
    entries: tuple[ExchangeEntry, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for e in self.entries:
            if e.exchange_id in seen:
                raise ValueError(
                    f"{self.species}: duplicate exchange_id {e.exchange_id!r} in entries"
                )
            seen.add(e.exchange_id)

    @property
    def n_exchanges(self) -> int:
        return len(self.entries)

    @property
    def exchange_ids(self) -> tuple[str, ...]:
        return tuple(e.exchange_id for e in self.entries)

    def mm_upper_bound(self, C_local: np.ndarray) -> np.ndarray:
        """Compute MM-derived uptake upper bound at a single grid cell."""
        C_local = np.asarray(C_local, dtype=float)
        if C_local.shape != (self.n_exchanges,):
            raise ValueError(
                f"{self.species}: C_local length {C_local.shape} does not match "
                f"n_exchanges={self.n_exchanges}"
            )
        out = np.empty(self.n_exchanges, dtype=float)
        for k, entry in enumerate(self.entries):
            out[k] = float(michaelis_menten(C_local[k], entry.vmax, entry.km))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_kinetics.py -v
```

Expected: All 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/kinetics.py tests/spatial/test_kinetics.py
git commit -m "feat(spatial): add ExchangeKinetics with per-species MM upper bound"
```

---

## Task 3: Kinetics YAML loader

**Files:**
- Modify: `src/gemfitcom/spatial/kinetics.py` (add `load_kinetics_yaml`)
- Modify: `tests/spatial/test_kinetics.py` (add YAML round-trip tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/spatial/test_kinetics.py`:

```python
class TestLoadKineticsYaml:
    def test_load_minimal(self, tmp_path):
        yaml_path = tmp_path / "ecoli.yaml"
        yaml_path.write_text(
            "species: ecoli\n"
            "exchanges:\n"
            "  EX_glc__D_e: {v_max: 10.0, K_m: 0.5}\n"
            "  EX_o2_e:     {v_max: 15.0, K_m: 0.005}\n"
        )
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        ek = load_kinetics_yaml(yaml_path)
        assert ek.species == "ecoli"
        assert ek.exchange_ids == ("EX_glc__D_e", "EX_o2_e")
        assert ek.entries[0].vmax == 10.0
        assert ek.entries[1].km == 0.005
        assert all(e.mode == "uptake_only" for e in ek.entries)

    def test_load_with_bidirectional_mode(self, tmp_path):
        yaml_path = tmp_path / "k.yaml"
        yaml_path.write_text(
            "species: x\n"
            "exchanges:\n"
            "  EX_ac_e: {v_max: 5.0, K_m: 0.1, mode: bidirectional}\n"
        )
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        ek = load_kinetics_yaml(yaml_path)
        assert ek.entries[0].mode == "bidirectional"

    def test_missing_file_raises(self, tmp_path):
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        with pytest.raises(FileNotFoundError):
            load_kinetics_yaml(tmp_path / "nope.yaml")

    def test_missing_required_field_raises(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "species: x\n"
            "exchanges:\n"
            "  EX_a_e: {v_max: 1.0}\n"
        )
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        with pytest.raises(KeyError, match="K_m"):
            load_kinetics_yaml(yaml_path)

    def test_top_level_species_required(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "exchanges:\n"
            "  EX_a_e: {v_max: 1.0, K_m: 0.1}\n"
        )
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        with pytest.raises(KeyError, match="species"):
            load_kinetics_yaml(yaml_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_kinetics.py::TestLoadKineticsYaml -v
```

Expected: `ImportError`.

- [ ] **Step 3: Add the loader to `kinetics.py`**

Append to `src/gemfitcom/spatial/kinetics.py`:

```python
from pathlib import Path

import yaml


def load_kinetics_yaml(path: str | Path) -> ExchangeKinetics:
    """Load a per-species kinetics YAML into :class:`ExchangeKinetics`.

    Expected schema (design doc §5.4)::

        species: ecoli
        exchanges:
          EX_glc__D_e: {v_max: 10.0, K_m: 0.5}
          EX_o2_e:     {v_max: 15.0, K_m: 0.005}
          EX_ac_e:     {v_max: 5.0,  K_m: 0.1, mode: bidirectional}

    YAML keys are ``v_max`` / ``K_m`` (scientific convention); the Python
    dataclass uses lowercase ``vmax`` / ``km``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Kinetics YAML not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    if "species" not in data:
        raise KeyError(f"{path}: missing required field 'species'")
    if "exchanges" not in data:
        raise KeyError(f"{path}: missing required field 'exchanges'")

    entries: list[ExchangeEntry] = []
    for exch_id, params in data["exchanges"].items():
        if not isinstance(params, dict):
            raise ValueError(f"{path}: exchange {exch_id} must map to a dict")
        if "v_max" not in params:
            raise KeyError(f"{path}: exchange {exch_id} missing 'v_max'")
        if "K_m" not in params:
            raise KeyError(f"{path}: exchange {exch_id} missing 'K_m'")
        entries.append(
            ExchangeEntry(
                exchange_id=exch_id,
                vmax=float(params["v_max"]),
                km=float(params["K_m"]),
                mode=params.get("mode", "uptake_only"),
            )
        )
    return ExchangeKinetics(species=data["species"], entries=tuple(entries))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_kinetics.py -v
```

Expected: 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/kinetics.py tests/spatial/test_kinetics.py
git commit -m "feat(spatial): add load_kinetics_yaml for per-species YAML"
```

---

## Task 4: GEM URI resolver — paths and `cobra://` built-ins

**Files:**
- Modify: `src/gemfitcom/spatial/kinetics.py` (add `resolve_gem`)
- Modify: `tests/spatial/test_kinetics.py` (add resolver tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/spatial/test_kinetics.py`:

```python
class TestResolveGem:
    def test_resolve_cobra_textbook(self):
        from gemfitcom.spatial.kinetics import resolve_gem

        m = resolve_gem("cobra://textbook")
        assert hasattr(m, "reactions")
        assert hasattr(m, "metabolites")
        assert any(r.id.startswith("EX_") for r in m.reactions)

    def test_resolve_filesystem_path(self, tmp_path):
        import cobra
        from gemfitcom.spatial.kinetics import resolve_gem

        m = cobra.io.load_model("textbook")
        sbml = tmp_path / "core.xml"
        cobra.io.write_sbml_model(m, str(sbml))
        loaded = resolve_gem(str(sbml))
        assert len(loaded.reactions) == len(m.reactions)

    def test_resolve_unknown_cobra_name_raises(self):
        from gemfitcom.spatial.kinetics import resolve_gem

        with pytest.raises(ValueError, match="cobra"):
            resolve_gem("cobra://this_model_does_not_exist_42")

    def test_resolve_missing_path_raises(self, tmp_path):
        from gemfitcom.spatial.kinetics import resolve_gem

        with pytest.raises(FileNotFoundError):
            resolve_gem(str(tmp_path / "nope.xml"))

    def test_resolve_unknown_scheme_raises(self):
        from gemfitcom.spatial.kinetics import resolve_gem

        with pytest.raises(ValueError, match="scheme"):
            resolve_gem("http://example.com/model.xml")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_kinetics.py::TestResolveGem -v
```

Expected: `ImportError`.

- [ ] **Step 3: Add `resolve_gem` to `kinetics.py`**

Append to `src/gemfitcom/spatial/kinetics.py`:

```python
import cobra

from gemfitcom.io.models import load_model as load_sbml_model

_COBRA_URI_PREFIX = "cobra://"


def resolve_gem(uri: str) -> cobra.Model:
    """Resolve a GEM identifier to a loaded :class:`cobra.Model`.

    Supported forms:
        - ``cobra://<name>``: built-in cobra-shipped models, e.g.
          ``cobra://textbook`` (E. coli core).
        - Filesystem path: delegates to :func:`gemfitcom.io.models.load_model`.
    """
    if uri.startswith(_COBRA_URI_PREFIX):
        name = uri[len(_COBRA_URI_PREFIX):]
        try:
            return cobra.io.load_model(name)
        except Exception as exc:
            raise ValueError(
                f"Unknown cobra built-in model {name!r} (URI={uri!r}): {exc}"
            ) from exc
    if "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ValueError(
            f"Unknown GEM URI scheme {scheme!r} in {uri!r}; expected 'cobra://' or a path"
        )
    return load_sbml_model(uri)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_kinetics.py -v
```

Expected: 19 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/kinetics.py tests/spatial/test_kinetics.py
git commit -m "feat(spatial): add resolve_gem with cobra:// URI scheme"
```

---

## Task 5: `SpeciesConfig` pydantic schema — wire into `SpatialConfig`

**Files:**
- Modify: `src/gemfitcom/spatial/config.py`
- Modify: `tests/spatial/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/spatial/test_config.py`:

```python
import pytest


class TestSpeciesConfig:
    def _base_yaml(self):
        """Return the geometry/simulation/metabolites portion shared by tests."""
        return """
            geometry:
              dim: 1
              n_grid: 10
              length: 1.0e-3
              boundary:
                mucosa: {type: reflecting}
                lumen:  {type: reflecting}
            simulation:
              t_end: 1.0
              dt: 0.1
              snapshot_every: 0.5
            metabolites:
              - id: glc__D_e
                diffusion: 6.7e-10
                init: {mode: uniform, value: 5.0}
            """

    def test_minimal_species_section(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(
            self._base_yaml()
            + """
            species:
              - name: ecoli
                gem: cobra://textbook
                kinetics: ./kinetics/ecoli.yaml
                init: {mode: uniform, value: 1.0e-3}
            """
        )
        from gemfitcom.spatial.config import SpatialConfig

        cfg = SpatialConfig.from_yaml(cfg_path)
        assert len(cfg.species) == 1
        sp = cfg.species[0]
        assert sp.name == "ecoli"
        assert sp.gem == "cobra://textbook"
        assert sp.kinetics.name == "ecoli.yaml"
        assert sp.init.mode == "uniform"
        assert sp.init.value == 1.0e-3

    def test_species_gaussian_init(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(
            self._base_yaml()
            + """
            species:
              - name: fprau
                gem: cobra://textbook
                kinetics: ./kinetics/fprau.yaml
                init:
                  mode: gaussian
                  center: 0.7
                  sigma: 0.1
                  peak: 1.0e-3
            """
        )
        from gemfitcom.spatial.config import SpatialConfig

        cfg = SpatialConfig.from_yaml(cfg_path)
        sp = cfg.species[0]
        assert sp.init.mode == "gaussian"
        assert sp.init.center == 0.7
        assert sp.init.sigma == 0.1
        assert sp.init.peak == 1.0e-3

    def test_species_empty_list_rejected(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(self._base_yaml() + "\nspecies: []\n")
        from gemfitcom.spatial.config import SpatialConfig

        with pytest.raises(ValueError, match="at least one"):
            SpatialConfig.from_yaml(cfg_path)

    def test_species_field_optional_for_pr1_compat(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(self._base_yaml())
        from gemfitcom.spatial.config import SpatialConfig

        cfg = SpatialConfig.from_yaml(cfg_path)
        assert cfg.species == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_config.py::TestSpeciesConfig -v
```

Expected: failures (`SpatialConfig` does not yet accept `species`).

- [ ] **Step 3: Extend `config.py`**

Edit `src/gemfitcom/spatial/config.py`. Add these classes before `SpatialConfig`:

```python
class SpeciesInitConfig(BaseModel):
    """Initial biomass distribution for one species."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["uniform", "gaussian", "step", "from_array"]
    value: float | None = None
    center: float | None = None
    sigma: float | None = None
    peak: float | None = None
    path: Path | None = None


class SpeciesConfig(BaseModel):
    """One species' GEM + kinetics + initial biomass."""

    model_config = ConfigDict(extra="forbid")

    name: str
    gem: str
    biomass_reaction: str | None = None
    kinetics: Path
    init: SpeciesInitConfig
```

Replace the existing `SpatialConfig` body with:

```python
class SpatialConfig(BaseModel):
    """Top-level spatial config (PR 1 + PR 2 surface)."""

    model_config = ConfigDict(extra="forbid")

    geometry: GeometryConfig
    metabolites: list[MetaboliteConfig]
    simulation: SimulationConfig
    species: list[SpeciesConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> SpatialConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls(**data)
        if "species" in data and len(cfg.species) == 0:
            raise ValueError(
                f"{path}: 'species' is present but empty; remove the field "
                "or provide at least one species"
            )
        return cfg

    def check_cfl(self) -> None:
        d_max = max((m.diffusion for m in self.metabolites), default=0.0)
        if d_max <= 0.0:
            return
        dx = self.geometry.length / (self.geometry.n_grid - 1)
        dt_limit = cfl_dt_max(dx, d_max, self.simulation.cfl_safety)
        if self.simulation.dt > dt_limit:
            raise RuntimeError(
                f"dt={self.simulation.dt} violates CFL stability "
                f"(dx={dx}, D_max={d_max}, safety={self.simulation.cfl_safety}). "
                f"Reduce dt to <= {dt_limit:.4g} or coarsen the grid."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_config.py -v
```

Expected: all existing config tests still pass + 4 new species tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/config.py tests/spatial/test_config.py
git commit -m "feat(spatial): add SpeciesConfig schema with init + kinetics path"
```

---

## Task 6: Biomass/concentration initial-condition builder

**Files:**
- Create: `src/gemfitcom/spatial/_init_fields.py`
- Create: `tests/spatial/test_init_fields.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spatial/test_init_fields.py`:

```python
"""Tests for spatial/_init_fields.py — concentration/biomass initialisation."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial._init_fields import build_field_1d
from gemfitcom.spatial.config import InitConfig, SpeciesInitConfig


class TestBuildField1D:
    def test_uniform(self):
        cfg = InitConfig(mode="uniform", value=2.5)
        field = build_field_1d(cfg, n_grid=10)
        assert field.shape == (10,)
        assert np.all(field == 2.5)

    def test_uniform_zero(self):
        cfg = InitConfig(mode="uniform", value=0.0)
        field = build_field_1d(cfg, n_grid=5)
        assert np.all(field == 0.0)

    def test_gaussian_peak_position(self):
        cfg = SpeciesInitConfig(mode="gaussian", center=0.5, sigma=0.1, peak=1.0)
        field = build_field_1d(cfg, n_grid=11)
        assert np.argmax(field) == 5
        assert np.isclose(field[5], 1.0, rtol=1e-3)

    def test_gaussian_non_negative(self):
        cfg = SpeciesInitConfig(mode="gaussian", center=0.7, sigma=0.05, peak=1.0e-3)
        field = build_field_1d(cfg, n_grid=50)
        assert np.isclose(field.max(), 1.0e-3, rtol=1e-2)
        assert np.all(field >= 0)

    def test_step(self):
        cfg = SpeciesInitConfig(mode="step", center=0.5, peak=1.0)
        field = build_field_1d(cfg, n_grid=10)
        x = np.linspace(0.0, 1.0, 10)
        assert np.all(field[x < 0.5] == 0.0)
        assert np.all(field[x >= 0.5] == 1.0)

    def test_from_array(self, tmp_path):
        arr_path = tmp_path / "init.npy"
        np.save(arr_path, np.linspace(0, 1, 8))
        cfg = SpeciesInitConfig(mode="from_array", path=arr_path)
        field = build_field_1d(cfg, n_grid=8)
        assert np.allclose(field, np.linspace(0, 1, 8))

    def test_from_array_shape_mismatch(self, tmp_path):
        arr_path = tmp_path / "init.npy"
        np.save(arr_path, np.zeros(5))
        cfg = SpeciesInitConfig(mode="from_array", path=arr_path)
        with pytest.raises(ValueError, match="shape"):
            build_field_1d(cfg, n_grid=8)

    def test_missing_required_field_raises(self):
        cfg = SpeciesInitConfig(mode="uniform")
        with pytest.raises(ValueError, match="value"):
            build_field_1d(cfg, n_grid=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_init_fields.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `_init_fields.py`**

```python
# src/gemfitcom/spatial/_init_fields.py
"""Build initial 1D fields (concentration or biomass) from an InitConfig."""

from __future__ import annotations

import numpy as np

from .config import InitConfig, SpeciesInitConfig

_InitLike = InitConfig | SpeciesInitConfig


def build_field_1d(cfg: _InitLike, n_grid: int) -> np.ndarray:
    """Build a 1D field of length ``n_grid`` from an init config.

    Supported modes:
        - ``uniform``:    constant ``value`` everywhere
        - ``gaussian``:   ``peak * exp(-((x - center)**2) / (2 * sigma**2))``,
                          x in [0, 1] (relative)
        - ``step``:       0 for ``x < center``, ``peak`` for ``x >= center``
        - ``from_array``: ``np.load(path)``; shape must equal ``(n_grid,)``
    """
    mode = cfg.mode
    if mode == "uniform":
        if cfg.value is None:
            raise ValueError("uniform mode requires 'value'")
        return np.full(n_grid, cfg.value, dtype=float)

    if mode == "gaussian":
        if cfg.center is None or cfg.sigma is None or cfg.peak is None:
            raise ValueError("gaussian mode requires 'center', 'sigma', and 'peak'")
        x = np.linspace(0.0, 1.0, n_grid)
        return cfg.peak * np.exp(-((x - cfg.center) ** 2) / (2.0 * cfg.sigma ** 2))

    if mode == "step":
        if cfg.center is None or cfg.peak is None:
            raise ValueError("step mode requires 'center' and 'peak'")
        x = np.linspace(0.0, 1.0, n_grid)
        return np.where(x >= cfg.center, cfg.peak, 0.0).astype(float)

    if mode == "from_array":
        if cfg.path is None:
            raise ValueError("from_array mode requires 'path'")
        arr = np.load(cfg.path)
        if arr.shape != (n_grid,):
            raise ValueError(
                f"from_array shape {arr.shape} does not match n_grid={n_grid}"
            )
        return arr.astype(float)

    raise ValueError(f"unknown init mode {mode!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_init_fields.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/_init_fields.py tests/spatial/test_init_fields.py
git commit -m "feat(spatial): add build_field_1d for uniform/gaussian/step/from_array"
```

---

## Task 7: Single-cell FBA primitive + exchange index

**Files:**
- Create: `src/gemfitcom/spatial/reaction.py`
- Create: `tests/spatial/test_reaction_serial.py`
- Modify: `tests/spatial/conftest.py` (add `textbook_model` + `fresh_textbook` fixtures)

- [ ] **Step 1: Add session-scoped textbook model fixture**

Edit `tests/spatial/conftest.py`:

```python
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
```

- [ ] **Step 2: Write the failing test for the single-cell FBA primitive**

Create `tests/spatial/test_reaction_serial.py`:

```python
"""Tests for spatial/reaction.py — single-cell FBA + SerialBackend."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.kinetics import ExchangeEntry, ExchangeKinetics
from gemfitcom.spatial.reaction import (
    SingleCellResult,
    build_exchange_index,
    solve_cell,
)


class TestBuildExchangeIndex:
    def test_returns_indices_for_present_exchanges(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        metabolite_ids = ("glc__D_e", "o2_e", "ac_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        assert idx == {0: 0, 1: 1}

    def test_kinetics_entry_missing_from_model_raises(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_does_not_exist_e", 10.0, 0.5, "uptake_only"),
            ),
        )
        with pytest.raises(KeyError, match="EX_does_not_exist_e"):
            build_exchange_index(fresh_textbook, ek, ("does_not_exist_e",))


class TestSolveCell:
    def test_solve_at_glucose_saturation_returns_positive_growth(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        metabolite_ids = ("glc__D_e", "o2_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([100.0, 100.0])
        result = solve_cell(
            model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local
        )
        assert isinstance(result, SingleCellResult)
        assert result.mu > 0
        assert not result.infeasible

    def test_solve_with_zero_glucose_zero_growth_or_minimal(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        metabolite_ids = ("glc__D_e", "o2_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([0.0, 100.0])
        result = solve_cell(
            model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local
        )
        assert result.mu >= 0
        glc_met_idx = metabolite_ids.index("glc__D_e")
        assert np.isclose(result.flux[glc_met_idx], 0.0)

    def test_solve_matches_bare_cobra_call(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_glc__D_e", 8.0, 0.5, "uptake_only"),),
        )
        metabolite_ids = ("glc__D_e",)
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([1.0])
        result = solve_cell(
            model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local
        )

        ref = fresh_textbook.copy()
        mm = 8.0 * 1.0 / (0.5 + 1.0)
        ref.reactions.get_by_id("EX_glc__D_e").lower_bound = -mm
        bare = ref.optimize()
        assert np.isclose(result.mu, bare.objective_value, rtol=1e-8)

    def test_solve_bidirectional_caps_secretion(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_ac_e", 5.0, 0.1, "bidirectional"),
            ),
        )
        metabolite_ids = ("glc__D_e", "ac_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([10.0, 0.0])
        result = solve_cell(
            model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local
        )
        ac_idx = metabolite_ids.index("ac_e")
        assert result.flux[ac_idx] <= 1e-9
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `reaction.py` — primitive only (engine in Task 9)**

Create `src/gemfitcom/spatial/reaction.py`:

```python
"""Per-cell dFBA: apply MM bounds, optimise, extract growth + flux.

Sign convention:
    - cobra exchange flux > 0 ⇒ secretion (medium gains)
    - cobra exchange flux < 0 ⇒ uptake (medium loses)
    The sign is passed straight through into ``flux``; the caller scales by
    biomass and dt to get a mmol/L delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinetics import ExchangeKinetics


@dataclass(frozen=True, slots=True)
class SingleCellResult:
    """Result of one species × one grid cell FBA solve."""

    mu: float
    flux: np.ndarray
    infeasible: bool


def build_exchange_index(
    model,
    kinetics: ExchangeKinetics,
    metabolite_ids: tuple[str, ...],
) -> dict[int, int]:
    """Cache `met_idx → kinetics entry idx` for the hot loop."""
    model_rxn_ids = {r.id for r in model.reactions}
    kin_idx_by_exch = {e.exchange_id: k for k, e in enumerate(kinetics.entries)}

    for exch_id in kin_idx_by_exch:
        if exch_id not in model_rxn_ids:
            raise KeyError(
                f"Kinetics for {kinetics.species!r} references {exch_id} "
                f"which is not in the model"
            )

    out: dict[int, int] = {}
    for met_idx, met_id in enumerate(metabolite_ids):
        exch_id = f"EX_{met_id}"
        if exch_id in kin_idx_by_exch:
            out[met_idx] = kin_idx_by_exch[exch_id]
    return out


def solve_cell(
    *,
    model,
    kinetics: ExchangeKinetics,
    exchange_index: dict[int, int],
    C_local: np.ndarray,
) -> SingleCellResult:
    """Solve one species × one grid cell FBA at local concentrations."""
    n_metabolites = C_local.shape[0]
    flux = np.zeros(n_metabolites, dtype=float)

    n_exch = kinetics.n_exchanges
    C_for_kinetics = np.zeros(n_exch, dtype=float)
    for met_idx, kin_idx in exchange_index.items():
        C_for_kinetics[kin_idx] = C_local[met_idx]

    mm_bounds = kinetics.mm_upper_bound(C_for_kinetics)

    for k, entry in enumerate(kinetics.entries):
        rxn = model.reactions.get_by_id(entry.exchange_id)
        rxn.lower_bound = -mm_bounds[k]
        if entry.mode == "bidirectional":
            rxn.upper_bound = mm_bounds[k]

    try:
        sol = model.optimize()
        infeasible = sol.status != "optimal"
    except Exception:
        infeasible = True
        sol = None

    if infeasible or sol is None:
        return SingleCellResult(mu=0.0, flux=flux, infeasible=True)

    mu = float(sol.objective_value) if sol.objective_value is not None else 0.0
    for met_idx, kin_idx in exchange_index.items():
        exch_id = kinetics.entries[kin_idx].exchange_id
        flux[met_idx] = float(sol.fluxes[exch_id])

    return SingleCellResult(mu=mu, flux=flux, infeasible=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py -v
```

Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/gemfitcom/spatial/reaction.py tests/spatial/test_reaction_serial.py tests/spatial/conftest.py
git commit -m "feat(spatial): add solve_cell + build_exchange_index"
```

---

## Task 8: `SerialBackend` — loop over grid cells × species

**Files:**
- Create: `src/gemfitcom/spatial/backends.py`
- Modify: `tests/spatial/test_reaction_serial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/spatial/test_reaction_serial.py`:

```python
class TestSerialBackend:
    def _make_setup(self):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        return ek, ("glc__D_e", "o2_e")

    def test_serial_step_shapes(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend()
        n_grid = 3
        C = np.array([[10.0, 5.0, 1.0], [100.0, 100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3, 1.0e-3]])
        mu, flux = backend.step(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=met_ids, C=C, B=B,
        )
        assert mu.shape == (1, n_grid)
        assert flux.shape == (2, 1, n_grid)
        assert np.all(mu > 0)

    def test_serial_skips_empty_cells(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend(empty_eps=1e-12)
        C = np.array([[10.0, 10.0, 10.0], [100.0, 100.0, 100.0]])
        B = np.array([[1.0e-3, 0.0, 1.0e-15]])
        mu, flux = backend.step(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=met_ids, C=C, B=B,
        )
        assert mu[0, 0] > 0
        assert mu[0, 1] == 0.0
        assert mu[0, 2] == 0.0
        assert np.all(flux[:, 0, 1] == 0)
        assert np.all(flux[:, 0, 2] == 0)

    def test_serial_two_species(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek1, met_ids = self._make_setup()
        ek2 = ExchangeKinetics(
            species="fprau",
            entries=(ExchangeEntry("EX_glc__D_e", 8.0, 0.5, "uptake_only"),),
        )
        backend = SerialBackend()
        C = np.array([[10.0, 10.0], [100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3], [1.0e-3, 1.0e-3]])
        mu, flux = backend.step(
            models=[fresh_textbook, fresh_textbook.copy()],
            kinetics=[ek1, ek2],
            metabolite_ids=met_ids, C=C, B=B,
        )
        assert mu.shape == (2, 2)
        assert flux.shape == (2, 2, 2)

    def test_negative_concentration_clipped(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend()
        C = np.array([[-1e-12, 10.0], [100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3]])
        mu, flux = backend.step(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=met_ids, C=C, B=B,
        )
        assert mu[0, 0] >= 0
        glc_idx = met_ids.index("glc__D_e")
        assert np.isclose(flux[glc_idx, 0, 0], 0.0, atol=1e-9)

    def test_serial_rejects_wrong_B_shape(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend()
        C = np.zeros((2, 3))
        B = np.zeros((1, 4))  # n_grid mismatch
        with pytest.raises(ValueError, match="n_grid"):
            backend.step(
                models=[fresh_textbook], kinetics=[ek],
                metabolite_ids=met_ids, C=C, B=B,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py::TestSerialBackend -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `backends.py`**

```python
# src/gemfitcom/spatial/backends.py
"""Strategy pattern for the reaction sub-step.

Backends iterate over grid cells × species and return per-step (mu, flux)
fields. ``SerialBackend`` is single-process. PR 4 adds ``JoblibBackend``
against the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .kinetics import ExchangeKinetics
from .reaction import build_exchange_index, solve_cell


class Backend(Protocol):
    """Reaction backend protocol.

    Returns:
        - ``mu``:   ``(n_species, n_grid)``, 1/h
        - ``flux``: ``(n_metabolites, n_species, n_grid)``, mmol/gDW/h
    """

    def step(
        self,
        *,
        models: list,
        kinetics: list[ExchangeKinetics],
        metabolite_ids: tuple[str, ...],
        C: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class SerialBackend:
    """Single-process loop over grid cells × species."""

    empty_eps: float = 1.0e-12

    def step(
        self,
        *,
        models: list,
        kinetics: list[ExchangeKinetics],
        metabolite_ids: tuple[str, ...],
        C: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_species = len(models)
        if n_species != len(kinetics):
            raise ValueError(
                f"models length {n_species} != kinetics length {len(kinetics)}"
            )
        n_metabolites, n_grid = C.shape
        if B.shape != (n_species, n_grid):
            raise ValueError(
                f"B shape {B.shape} does not match (n_species={n_species}, "
                f"n_grid={n_grid})"
            )

        indices = [
            build_exchange_index(models[i], kinetics[i], metabolite_ids)
            for i in range(n_species)
        ]

        mu = np.zeros((n_species, n_grid), dtype=float)
        flux = np.zeros((n_metabolites, n_species, n_grid), dtype=float)
        C_safe = np.maximum(C, 0.0)

        for i in range(n_species):
            for x in range(n_grid):
                if B[i, x] < self.empty_eps:
                    continue
                result = solve_cell(
                    model=models[i],
                    kinetics=kinetics[i],
                    exchange_index=indices[i],
                    C_local=C_safe[:, x],
                )
                mu[i, x] = result.mu
                flux[:, i, x] = result.flux

        return mu, flux
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py -v
```

Expected: 11 tests pass (6 + 5 backend tests).

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/backends.py tests/spatial/test_reaction_serial.py
git commit -m "feat(spatial): add SerialBackend with empty-cell skip + shape checks"
```

---

## Task 9: `ReactionEngine` — high-level orchestrator

**Files:**
- Modify: `src/gemfitcom/spatial/reaction.py` (add `ReactionEngine` class)
- Modify: `tests/spatial/test_reaction_serial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/spatial/test_reaction_serial.py`:

```python
class TestReactionEngine:
    def test_engine_step_returns_correct_shapes(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        engine = ReactionEngine(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=("glc__D_e", "o2_e"),
            backend=SerialBackend(),
        )
        C = np.array([[10.0, 5.0], [100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3]])
        mu, flux = engine.step(C, B, dt=0.1)
        assert mu.shape == (1, 2)
        assert flux.shape == (2, 1, 2)

    def test_engine_rejects_mismatched_lengths(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_glc__D_e", 10.0, 0.5),),
        )
        with pytest.raises(ValueError, match="models"):
            ReactionEngine(
                models=[fresh_textbook, fresh_textbook],
                kinetics=[ek],
                metabolite_ids=("glc__D_e",),
                backend=SerialBackend(),
            )

    def test_engine_apply_to_state_grows_biomass_consumes_substrate(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        engine = ReactionEngine(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=("glc__D_e", "o2_e"),
            backend=SerialBackend(),
        )
        C = np.array([[10.0], [100.0]])
        B = np.array([[1.0e-3]])
        C_new, B_new = engine.apply_to_state(C.copy(), B.copy(), dt=0.1)
        assert B_new[0, 0] > B[0, 0]
        assert C_new[0, 0] < C[0, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py::TestReactionEngine -v
```

Expected: `ImportError` for `ReactionEngine`.

- [ ] **Step 3: Add `ReactionEngine` to `reaction.py`**

Append to `src/gemfitcom/spatial/reaction.py`:

```python
@dataclass
class ReactionEngine:
    """Per-step reaction orchestrator.

    Attributes:
        models: One :class:`cobra.Model` per species (order matches ``kinetics``).
        kinetics: One :class:`ExchangeKinetics` per species.
        metabolite_ids: Tuple of cobra metabolite ids tracked on the grid.
        backend: Implementation of the :class:`Backend` protocol.
    """

    models: list
    kinetics: list[ExchangeKinetics]
    metabolite_ids: tuple[str, ...]
    backend: object  # actually backends.Backend; runtime check not needed

    def __post_init__(self) -> None:
        if len(self.models) != len(self.kinetics):
            raise ValueError(
                f"models length {len(self.models)} != kinetics length "
                f"{len(self.kinetics)}"
            )

    @property
    def n_species(self) -> int:
        return len(self.models)

    @property
    def n_metabolites(self) -> int:
        return len(self.metabolite_ids)

    def step(
        self, C: np.ndarray, B: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (mu, flux) fields. Does NOT mutate state."""
        return self.backend.step(
            models=self.models,
            kinetics=self.kinetics,
            metabolite_ids=self.metabolite_ids,
            C=C,
            B=B,
        )

    def apply_to_state(
        self, C: np.ndarray, B: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Take one reaction substep and return updated ``(C, B)``.

        Algorithm (spec §3.2)::

            mu, flux = step(C, B, dt)
            B_new = B * exp(mu * dt)
            C_new = max(C + einsum('jix,ix->jx', flux, B) * dt, 0)
        """
        mu, flux = self.step(C, B, dt)
        B_new = B * np.exp(mu * dt)
        dC = np.einsum("jix,ix->jx", flux, B) * dt
        C_new = np.maximum(C + dC, 0.0)
        return C_new, B_new
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py -v
```

Expected: 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/reaction.py tests/spatial/test_reaction_serial.py
git commit -m "feat(spatial): add ReactionEngine with step + apply_to_state"
```

---

## Task 10: Run-time guards — mu·dt overflow + warning capture

**Files:**
- Modify: `src/gemfitcom/spatial/reaction.py`
- Modify: `tests/spatial/test_reaction_serial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/spatial/test_reaction_serial.py`:

```python
class TestEngineGuards:
    def test_engine_clips_mu_when_overflow(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),),
        )
        engine = ReactionEngine(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=("glc__D_e",),
            backend=SerialBackend(),
            mu_dt_clip=5.0,
        )
        C = np.array([[100.0]])
        B = np.array([[1.0e-3]])
        _, B_new = engine.apply_to_state(C.copy(), B.copy(), dt=100.0)
        assert B_new[0, 0] <= 1.0e-3 * np.exp(5.0) + 1e-9
        assert len(engine.warnings) >= 1
        assert any("mu" in w.lower() for w in engine.warnings)

    def test_warnings_accumulate_and_clear(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),),
        )
        engine = ReactionEngine(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=("glc__D_e",),
            backend=SerialBackend(),
            mu_dt_clip=5.0,
        )
        engine.apply_to_state(np.array([[100.0]]), np.array([[1.0e-3]]), dt=100.0)
        first_n = len(engine.warnings)
        engine.apply_to_state(np.array([[100.0]]), np.array([[1.0e-3]]), dt=100.0)
        assert len(engine.warnings) > first_n
        engine.clear_warnings()
        assert engine.warnings == []

    def test_no_warnings_at_default_clip(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),),
        )
        engine = ReactionEngine(
            models=[fresh_textbook], kinetics=[ek],
            metabolite_ids=("glc__D_e",),
            backend=SerialBackend(),
        )
        engine.apply_to_state(np.array([[10.0]]), np.array([[1.0e-3]]), dt=0.1)
        assert engine.warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py::TestEngineGuards -v
```

Expected: failures (no `mu_dt_clip` kwarg, no `warnings` attribute, no `clear_warnings`).

- [ ] **Step 3: Modify `ReactionEngine`**

In `src/gemfitcom/spatial/reaction.py`, replace the `ReactionEngine` class with:

```python
@dataclass
class ReactionEngine:
    """Per-step reaction orchestrator.

    Attributes:
        models: One :class:`cobra.Model` per species.
        kinetics: One :class:`ExchangeKinetics` per species.
        metabolite_ids: Tuple of cobra metabolite ids tracked on the grid.
        backend: Implementation of the :class:`Backend` protocol.
        mu_dt_clip: If ``mu * dt > mu_dt_clip`` in any cell, mu is clipped
            in place and a warning is appended to :attr:`warnings`. Default
            5.0 (exp(5) ≈ 148× growth per step is non-physical). Use
            ``float('inf')`` to disable.

    Non-init attributes:
        warnings: Mutable list of warning strings accumulated across calls.
    """

    models: list
    kinetics: list[ExchangeKinetics]
    metabolite_ids: tuple[str, ...]
    backend: object
    mu_dt_clip: float = 5.0

    def __post_init__(self) -> None:
        if len(self.models) != len(self.kinetics):
            raise ValueError(
                f"models length {len(self.models)} != kinetics length "
                f"{len(self.kinetics)}"
            )
        self.warnings: list[str] = []

    @property
    def n_species(self) -> int:
        return len(self.models)

    @property
    def n_metabolites(self) -> int:
        return len(self.metabolite_ids)

    def clear_warnings(self) -> None:
        self.warnings = []

    def step(
        self, C: np.ndarray, B: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.backend.step(
            models=self.models,
            kinetics=self.kinetics,
            metabolite_ids=self.metabolite_ids,
            C=C,
            B=B,
        )

    def apply_to_state(
        self, C: np.ndarray, B: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        mu, flux = self.step(C, B, dt)

        threshold = self.mu_dt_clip
        if np.any(mu * dt > threshold):
            mask = mu * dt > threshold
            n_clipped = int(mask.sum())
            self.warnings.append(
                f"clipped mu in {n_clipped} cells where mu*dt > {threshold} "
                f"(max mu*dt={float((mu * dt).max()):.3g}, dt={dt})"
            )
            mu = np.where(mask, threshold / dt, mu)

        B_new = B * np.exp(mu * dt)
        dC = np.einsum("jix,ix->jx", flux, B) * dt
        C_new = np.maximum(C + dC, 0.0)
        return C_new, B_new
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_reaction_serial.py -v
```

Expected: 17 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/reaction.py tests/spatial/test_reaction_serial.py
git commit -m "feat(spatial): add mu*dt overflow clip + warning capture"
```

---

## Task 11: Skeleton example dir

**Files:**
- Create: `examples/spatial/ecoli_fprau_skeleton/sim.yaml`
- Create: `examples/spatial/ecoli_fprau_skeleton/kinetics/ecoli.yaml`
- Create: `examples/spatial/ecoli_fprau_skeleton/kinetics/fprau.yaml`
- Create: `examples/spatial/ecoli_fprau_skeleton/README.md`

- [ ] **Step 1: Create the example sim.yaml**

```yaml
# examples/spatial/ecoli_fprau_skeleton/sim.yaml
#
# Spatial dFBA skeleton example for PR 2.
# Both species use the cobra-bundled textbook (E. coli core) model — the
# "fprau" entry is a placeholder pointing at the same GEM so the example runs
# offline. Replace with a real F. prausnitzii SBML when available.

geometry:
  dim: 1
  n_grid: 10
  length: 1.0e-3
  boundary:
    mucosa: { type: reflecting }
    lumen:  { type: reflecting }

simulation:
  t_end: 1.0
  dt: 0.1
  snapshot_every: 0.5
  cfl_safety: 0.4

metabolites:
  - id: glc__D_e
    diffusion: 6.7e-10
    init: { mode: uniform, value: 5.0 }
  - id: o2_e
    diffusion: 2.1e-9
    init: { mode: uniform, value: 0.21 }

species:
  - name: ecoli
    gem: cobra://textbook
    kinetics: ./kinetics/ecoli.yaml
    init:
      mode: uniform
      value: 1.0e-3
  - name: fprau
    gem: cobra://textbook
    kinetics: ./kinetics/fprau.yaml
    init:
      mode: gaussian
      center: 0.7
      sigma: 0.1
      peak: 1.0e-3

output:
  format: npz
  precision: float32
```

- [ ] **Step 2: Create kinetics YAMLs**

```yaml
# examples/spatial/ecoli_fprau_skeleton/kinetics/ecoli.yaml
species: ecoli
exchanges:
  EX_glc__D_e: { v_max: 10.0, K_m: 0.5 }
  EX_o2_e:     { v_max: 15.0, K_m: 0.005 }
```

```yaml
# examples/spatial/ecoli_fprau_skeleton/kinetics/fprau.yaml
species: fprau
exchanges:
  EX_glc__D_e: { v_max: 8.0, K_m: 0.5 }
  EX_o2_e:     { v_max: 0.5, K_m: 0.001 }
```

- [ ] **Step 3: Create README**

```markdown
# Spatial dFBA Skeleton: E. coli + Fprau (placeholder)

Minimal 2-species × 2-metabolite × 10-grid example used by the PR 2 smoke
test. Both species point at the cobra-bundled `textbook` (E. coli core)
model via `cobra://textbook` so the example runs without external SBML
downloads. The "fprau" entry is a placeholder; replace with a real
F. prausnitzii SBML when available.

## Files

| File | Purpose |
|---|---|
| `sim.yaml` | Top-level config — geometry, metabolites, species, simulation |
| `kinetics/ecoli.yaml` | E. coli MM parameters for glucose + oxygen |
| `kinetics/fprau.yaml` | "Fprau" placeholder MM parameters (near-anaerobic) |

## Run (PR 2 surface — no CLI yet)

```python
import numpy as np
from pathlib import Path
from gemfitcom.spatial import (
    SpatialConfig, ReactionEngine, SerialBackend,
    resolve_gem, load_kinetics_yaml, build_field_1d,
)

cfg_path = Path("examples/spatial/ecoli_fprau_skeleton/sim.yaml")
cfg = SpatialConfig.from_yaml(cfg_path)
base = cfg_path.parent

models = [resolve_gem(s.gem) for s in cfg.species]
kinetics = [
    load_kinetics_yaml(s.kinetics if s.kinetics.is_absolute() else base / s.kinetics)
    for s in cfg.species
]
metabolite_ids = tuple(m.id for m in cfg.metabolites)

n_grid = cfg.geometry.n_grid
C = np.stack([build_field_1d(m.init, n_grid) for m in cfg.metabolites])
B = np.stack([build_field_1d(s.init, n_grid) for s in cfg.species])

engine = ReactionEngine(
    models=models, kinetics=kinetics,
    metabolite_ids=metabolite_ids,
    backend=SerialBackend(),
)
for _ in range(10):
    C, B = engine.apply_to_state(C, B, dt=cfg.simulation.dt)
```

CLI form `gemfitcom spatial run sim.yaml` will land in PR 3.
```

- [ ] **Step 4: Verify the example loads**

Run:
```bash
python -c "
from gemfitcom.spatial.config import SpatialConfig
cfg = SpatialConfig.from_yaml('examples/spatial/ecoli_fprau_skeleton/sim.yaml')
print('species:', [s.name for s in cfg.species])
print('metabolites:', [m.id for m in cfg.metabolites])
print('n_grid:', cfg.geometry.n_grid)
"
```

Expected: `species: ['ecoli', 'fprau']  metabolites: ['glc__D_e', 'o2_e']  n_grid: 10`

- [ ] **Step 5: Commit**

```bash
git add examples/spatial/ecoli_fprau_skeleton/
git commit -m "docs(spatial): add ecoli+fprau skeleton example for PR 2"
```

---

## Task 12: End-to-end smoke test — sim.yaml × 10 steps

**Files:**
- Create: `tests/spatial/test_reaction_e2e.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/spatial/test_reaction_e2e.py
"""End-to-end smoke test for PR 2: example sim.yaml × 10 reaction steps.

Reaction-only (no diffusion). Full pipeline lands in PR 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gemfitcom.spatial._init_fields import build_field_1d
from gemfitcom.spatial.backends import SerialBackend
from gemfitcom.spatial.config import SpatialConfig
from gemfitcom.spatial.kinetics import load_kinetics_yaml, resolve_gem
from gemfitcom.spatial.reaction import ReactionEngine

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "examples" / "spatial" / "ecoli_fprau_skeleton"


@pytest.fixture
def loaded_engine():
    cfg_path = EXAMPLE_DIR / "sim.yaml"
    if not cfg_path.is_file():
        pytest.skip(f"Example not found at {cfg_path}")
    cfg = SpatialConfig.from_yaml(cfg_path)
    models = [resolve_gem(s.gem) for s in cfg.species]
    kinetics = [
        load_kinetics_yaml(
            s.kinetics if s.kinetics.is_absolute() else EXAMPLE_DIR / s.kinetics
        )
        for s in cfg.species
    ]
    metabolite_ids = tuple(m.id for m in cfg.metabolites)
    engine = ReactionEngine(
        models=models, kinetics=kinetics,
        metabolite_ids=metabolite_ids,
        backend=SerialBackend(),
    )
    return cfg, engine


def _initial_state(cfg):
    n_grid = cfg.geometry.n_grid
    C = np.stack([build_field_1d(m.init, n_grid) for m in cfg.metabolites])
    B = np.stack([build_field_1d(s.init, n_grid) for s in cfg.species])
    return C, B


class TestExampleSmoke:
    def test_ten_step_run_completes_without_exception(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert C.shape == (len(cfg.metabolites), cfg.geometry.n_grid)
        assert B.shape == (len(cfg.species), cfg.geometry.n_grid)

    def test_ten_step_biomass_and_conc_non_negative(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert np.all(B >= 0)
        assert np.all(C >= 0)

    def test_ten_step_glucose_does_not_increase(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        glc_idx = next(i for i, m in enumerate(cfg.metabolites) if m.id == "glc__D_e")
        initial_total = float(C[glc_idx].sum())
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        final_total = float(C[glc_idx].sum())
        assert final_total <= initial_total + 1e-9

    def test_at_least_one_species_grows(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        B0 = B.copy()
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert np.any(B.sum(axis=1) > B0.sum(axis=1))

    def test_no_engine_warnings_at_default_clip(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert engine.warnings == []
```

- [ ] **Step 2: Run test to verify it passes**

Run:
```bash
pytest tests/spatial/test_reaction_e2e.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/spatial/test_reaction_e2e.py
git commit -m "test(spatial): add end-to-end 10-step smoke for PR 2"
```

---

## Task 13: Public API — export PR 2 surface

**Files:**
- Modify: `src/gemfitcom/spatial/__init__.py`
- Modify: `tests/spatial/test_public_api.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/spatial/test_public_api.py`:

```python
class TestPr2PublicApi:
    def test_pr2_kinetics_classes_exported(self):
        from gemfitcom.spatial import (
            ExchangeEntry, ExchangeKinetics,
            load_kinetics_yaml, resolve_gem,
        )
        assert ExchangeKinetics is not None
        assert ExchangeEntry is not None
        assert callable(load_kinetics_yaml)
        assert callable(resolve_gem)

    def test_pr2_reaction_classes_exported(self):
        from gemfitcom.spatial import ReactionEngine, SerialBackend
        assert ReactionEngine is not None
        assert SerialBackend is not None

    def test_pr2_config_classes_exported(self):
        from gemfitcom.spatial import SpeciesConfig, SpeciesInitConfig
        assert SpeciesConfig is not None
        assert SpeciesInitConfig is not None

    def test_pr2_init_field_helper_exported(self):
        from gemfitcom.spatial import build_field_1d
        assert callable(build_field_1d)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/spatial/test_public_api.py::TestPr2PublicApi -v
```

Expected: `ImportError`.

- [ ] **Step 3: Extend `__init__.py`**

Replace `src/gemfitcom/spatial/__init__.py` with:

```python
"""Spatial dFBA subpackage for gemfitcom.

PR 1 surface: numerical core (state, geometry, diffusion, recorder, config).
PR 2 surface: kinetics + reaction + serial backend + species config + GEM URI.
PR 3 will add: Simulator + CLI.
PR 4 will add: JoblibBackend + cache.
PR 5 will add: viz.
"""

from ._init_fields import build_field_1d
from .backends import Backend, SerialBackend
from .config import (
    BoundaryConfig,
    GeometryConfig,
    InitConfig,
    MetaboliteConfig,
    OutputConfig,
    SimulationConfig,
    SpatialConfig,
    SpeciesConfig,
    SpeciesInitConfig,
)
from .diffusion import build_laplacian_1d, cfl_dt_max, check_cfl, diffuse_step
from .geometry import BoundarySpec, Geometry1D
from .kinetics import (
    ExchangeEntry,
    ExchangeKinetics,
    load_kinetics_yaml,
    resolve_gem,
)
from .reaction import ReactionEngine, SingleCellResult, solve_cell
from .recorder import SnapshotRecorder
from .state import SpatialState

__all__ = [
    # PR 1
    "BoundaryConfig",
    "BoundarySpec",
    "GeometryConfig",
    "Geometry1D",
    "InitConfig",
    "MetaboliteConfig",
    "OutputConfig",
    "SimulationConfig",
    "SnapshotRecorder",
    "SpatialConfig",
    "SpatialState",
    "build_laplacian_1d",
    "cfl_dt_max",
    "check_cfl",
    "diffuse_step",
    # PR 2
    "Backend",
    "ExchangeEntry",
    "ExchangeKinetics",
    "ReactionEngine",
    "SerialBackend",
    "SingleCellResult",
    "SpeciesConfig",
    "SpeciesInitConfig",
    "build_field_1d",
    "load_kinetics_yaml",
    "resolve_gem",
    "solve_cell",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/spatial/test_public_api.py -v
```

Expected: existing PR 1 API tests + 4 new PR 2 tests all pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/__init__.py tests/spatial/test_public_api.py
git commit -m "feat(spatial): export PR 2 public API surface"
```

---

## Task 14: Cross-cutting verification — full suite + lint

**Files:** none modified.

- [ ] **Step 1: Full spatial test suite green**

Run:
```bash
pytest tests/spatial/ -v
```

Expected: all tests pass. Roughly +60 tests over PR 1 baseline.

- [ ] **Step 2: Full repo test suite — no regressions outside spatial**

Run:
```bash
pytest tests/ -x --ignore=tests/spatial -q
```

Expected: all non-spatial tests still pass.

- [ ] **Step 3: Lint check**

Run:
```bash
ruff check src/gemfitcom/spatial tests/spatial
ruff format --check src/gemfitcom/spatial tests/spatial
```

Expected: no warnings, formatting clean. If formatting drifts, run `ruff format src/gemfitcom/spatial tests/spatial`.

- [ ] **Step 4: Commit any final formatting fix (skip if nothing changed)**

```bash
git add -A
git diff --cached --quiet || git commit -m "style(spatial): ruff format pass for PR 2"
```

---

## Task 15: PR 2 finalisation — branch summary + push

**Files:** none modified.

- [ ] **Step 1: Show the full PR 2 commit list**

Run:
```bash
git log main..HEAD --oneline
```

Expected: roughly 11–14 commits on `spatial/pr2-kinetics-reaction`.

- [ ] **Step 2: Show diff stat against main**

Run:
```bash
git diff --stat main..HEAD
```

Expected: changes confined to `src/gemfitcom/spatial/`, `tests/spatial/`, `examples/spatial/ecoli_fprau_skeleton/`.

- [ ] **Step 3: HARD STOP — confirm with user before pushing**

`git push` requires explicit user authorisation per project policy. Ask:

> "PR 2 implementation complete. Ready to push `spatial/pr2-kinetics-reaction` to `origin`?"

- [ ] **Step 4: After approval, push**

```bash
git push -u origin spatial/pr2-kinetics-reaction
```

- [ ] **Step 5: Offer next steps**

1. Open a draft PR via `gh pr create --draft --base main` (requires authorisation).
2. Hand off to PR 3 planning (Simulator + CLI + diffusion-reaction coupling).

---

## Self-Review Checklist

**1. Spec coverage:** Every PR 2 deliverable from spec §8 maps to a task.
- `kinetics.py` → Tasks 2, 3, 4
- `reaction.py` → Tasks 7, 9, 10
- `backends.py` (SerialBackend) → Task 8
- `config.py` (species + kinetics) → Task 5
- `examples/spatial/ecoli_fprau_skeleton/` → Task 11
- Unit tests → Tasks 2, 3, 4, 6, 7, 8, 9, 10, 13
- `test_simulator_short` equivalent (renamed `test_reaction_e2e` since Simulator is PR 3) → Task 12
- "示例 sim.yaml 跑 10 步不报错" verification → Task 12

**2. Placeholder scan:** Every task contains complete code. No "TBD" / "implement later" / "similar to Task N" without code.

**3. Type consistency:**
- `ExchangeKinetics.entries`: `tuple[ExchangeEntry, ...]` everywhere.
- `mu` shape: `(n_species, n_grid)` everywhere.
- `flux` shape: `(n_metabolites, n_species, n_grid)` everywhere.
- `metabolite_ids`: `tuple[str, ...]` everywhere.
- `exchange_index`: `dict[int, int]` (met_idx → kinetics_idx) everywhere.
- `SpeciesConfig.gem`: `str` (URI or path); `SpeciesConfig.kinetics`: `Path`.

**4. Out-of-scope discipline:** No task implements diffusion-reaction coupling, time loop, CLI, joblib, viz.

**5. Test discipline:** Every implementation task has a failing-test step before the implement step (TDD enforced). Smoke test loads the actual example sim.yaml from disk.

**6. Commit cadence:** Each task ends with one commit (Task 1 and 14 are verification-only).

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-11-spatial-dfba-pr2-kinetics-reaction.md`.

Two execution options:

1. **Subagent-driven (recommended)** — dispatch fresh subagents per task, two-stage review between tasks, fast iteration with isolated context.
2. **Inline execution** — execute tasks in-session with checkpoints; smaller overhead but heavier on this session's context budget.

Choose the approach you want before starting Task 1.
