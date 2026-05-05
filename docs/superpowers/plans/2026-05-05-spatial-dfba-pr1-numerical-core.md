# Spatial dFBA PR 1: Numerical Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FBA-decoupled numerical foundation for `gemfitcom.spatial`: state representation, 1D geometry with boundary conditions, explicit-FTCS diffusion solver, snapshot recorder, and pydantic-validated YAML config — all with TDD coverage including mass conservation and positivity invariants.

**Architecture:** A new `gemfitcom.spatial` subpackage with five modules (`state`, `geometry`, `diffusion`, `recorder`, `config`). Pure NumPy + SciPy + pydantic; zero dependency on `cobra` (FBA wiring lands in PR 2). Operator-splitting algorithm structure baked in but only the diffusion + boundary half is wired in this PR.

**Tech Stack:** Python ≥3.10, NumPy ≥1.24, SciPy ≥1.11 (sparse Laplacian), pydantic ≥2.5 (config schema), pytest (TDD).

**Spec reference:** `docs/superpowers/specs/2026-05-05-spatial-dfba-v0.1-design.md` §3.1, §3.2 (diffusion half), §3.4, §3.5, §5.3 (config), §6 (errors), §7 (tests), §8 PR 1.

**Estimated work:** ~600 LOC including tests; 1.5–2 weeks at 1–2 h/day.

---

## File Manifest

### New files

| Path | Responsibility |
|---|---|
| `src/gemfitcom/spatial/__init__.py` | Public API exports for PR 1 surface |
| `src/gemfitcom/spatial/state.py` | `SpatialState` dataclass + shape invariants |
| `src/gemfitcom/spatial/geometry.py` | `Geometry1D` + `BoundarySpec` + boundary source application |
| `src/gemfitcom/spatial/diffusion.py` | Sparse Laplacian builder, `diffuse_step`, CFL helpers |
| `src/gemfitcom/spatial/recorder.py` | `SnapshotRecorder` (npz writer with `maybe_save` cadence) |
| `src/gemfitcom/spatial/config.py` | Pydantic models: `SpatialConfig` and components |
| `tests/spatial/__init__.py` | Marks tests dir as a package |
| `tests/spatial/conftest.py` | Spatial-only fixtures (overrides allowed; inherits RNG seeding from `tests/conftest.py`) |
| `tests/spatial/test_state.py` | SpatialState shape/invariant tests |
| `tests/spatial/test_geometry.py` | Geometry1D + boundary tests |
| `tests/spatial/test_diffusion.py` | Laplacian, diffuse_step, analytical accuracy, CFL |
| `tests/spatial/test_recorder.py` | Save/load/maybe_save tests |
| `tests/spatial/test_config.py` | YAML schema validation, CFL integration |
| `tests/spatial/test_invariant_mass_conservation.py` | Closed BC long-run mass conservation |
| `tests/spatial/test_invariant_positivity.py` | Random IC positivity |

### Modified files

| Path | Change |
|---|---|
| `pyproject.toml` | Add `[project.optional-dependencies] spatial = ["pydantic>=2.5"]` |

### Out of scope for PR 1 (live in later PRs)

- `reaction.py`, `kinetics.py`, `backends.py`, `cache.py` — PR 2
- `simulator.py`, `cli.py` — PR 3
- joblib/parallel — PR 4
- `viz.py` — PR 5
- `species` and `kinetics` config sections — PR 2
- `backend` config section — PR 4

---

## Task 1: Setup — subpackage skeleton + extras + install

**Files:**
- Create: `src/gemfitcom/spatial/__init__.py`
- Create: `tests/spatial/__init__.py`
- Create: `tests/spatial/conftest.py`
- Modify: `pyproject.toml` (add spatial extras)

- [ ] **Step 1: Create subpackage `__init__.py` (empty placeholder for now)**

```python
# src/gemfitcom/spatial/__init__.py
"""Spatial dFBA subpackage for gemfitcom.

PR 1 scope: numerical core (state, geometry, diffusion, recorder, config).
FBA coupling, simulator, CLI, and parallel backends added in subsequent PRs.
"""

# Public API populated as modules land in later tasks of PR 1.
```

- [ ] **Step 2: Create test package marker and conftest**

```python
# tests/spatial/__init__.py
```

```python
# tests/spatial/conftest.py
"""Spatial-only pytest fixtures.

Inherits RNG seeding from tests/conftest.py automatically.
Add fixtures here as later tasks need them.
"""
```

- [ ] **Step 3: Add spatial optional-dependency to pyproject.toml**

In `pyproject.toml`, locate the existing `[project.optional-dependencies]` block (currently has `dev` and `docs`) and add a `spatial` entry:

```toml
[project.optional-dependencies]
spatial = [
    "pydantic>=2.5",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.5",
    "pre-commit>=3.7",
]
docs = [
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
]
```

(Note: `joblib` is intentionally not added in PR 1 — added in PR 4 when it's actually used.)

- [ ] **Step 4: Install spatial extras and verify import**

Run:
```bash
pip install -e ".[spatial,dev]"
python -c "import gemfitcom.spatial; print(gemfitcom.spatial.__doc__)"
```
Expected: prints the docstring without errors. Pydantic must be importable: `python -c "import pydantic; print(pydantic.VERSION)"` should print 2.x.

- [ ] **Step 5: Run existing test suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: all pre-PR1 tests still pass (PR 1 added no test files yet).

- [ ] **Step 6: Commit**

```bash
git add src/gemfitcom/spatial/__init__.py tests/spatial/__init__.py tests/spatial/conftest.py pyproject.toml
git commit -m "feat(spatial): scaffold subpackage + add spatial extras (pydantic)"
```

---

## Task 2: SpatialState dataclass

**Files:**
- Create: `src/gemfitcom/spatial/state.py`
- Test: `tests/spatial/test_state.py`

`SpatialState` holds metabolite concentrations `(n_metabolites, n_grid)` and biomass `(n_species, n_grid)` plus current time. Validates shapes on construction.

- [ ] **Step 1: Write failing tests**

```python
# tests/spatial/test_state.py
"""Tests for SpatialState dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.state import SpatialState


def test_construct_from_arrays() -> None:
    C = np.zeros((3, 50))
    B = np.ones((2, 50))
    state = SpatialState(metabolites=C, biomass=B, t=0.5)
    assert state.n_metabolites == 3
    assert state.n_species == 2
    assert state.n_grid == 50
    assert state.t == 0.5


def test_default_t_is_zero() -> None:
    state = SpatialState(metabolites=np.zeros((1, 4)), biomass=np.zeros((1, 4)))
    assert state.t == 0.0


def test_metabolites_must_be_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        SpatialState(metabolites=np.zeros(10), biomass=np.zeros((1, 10)))


def test_biomass_must_be_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        SpatialState(metabolites=np.zeros((1, 10)), biomass=np.zeros(10))


def test_grid_dim_must_match() -> None:
    with pytest.raises(ValueError, match="n_grid"):
        SpatialState(metabolites=np.zeros((1, 10)), biomass=np.zeros((1, 8)))


def test_from_arrays_classmethod_uses_kwargs() -> None:
    state = SpatialState.from_arrays(C=np.zeros((1, 5)), B=np.zeros((1, 5)), t=2.0)
    assert state.t == 2.0
    assert state.metabolites.dtype == np.float64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_state.py`
Expected: ImportError (module doesn't exist yet) or all tests collect-error.

- [ ] **Step 3: Implement SpatialState**

```python
# src/gemfitcom/spatial/state.py
"""SpatialState: concentration + biomass fields on a 1D grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialState:
    """Current state of a spatial simulation.

    Attributes:
        metabolites: ndarray (n_metabolites, n_grid), mmol/L
        biomass:     ndarray (n_species, n_grid), gDW/L
        t:           current simulation time in hours
    """

    metabolites: np.ndarray
    biomass: np.ndarray
    t: float = 0.0

    def __post_init__(self) -> None:
        if self.metabolites.ndim != 2:
            raise ValueError(
                f"metabolites must be 2D (n_metabolites, n_grid); got shape {self.metabolites.shape}"
            )
        if self.biomass.ndim != 2:
            raise ValueError(
                f"biomass must be 2D (n_species, n_grid); got shape {self.biomass.shape}"
            )
        if self.metabolites.shape[1] != self.biomass.shape[1]:
            raise ValueError(
                f"metabolites and biomass must share n_grid: "
                f"{self.metabolites.shape[1]} vs {self.biomass.shape[1]}"
            )

    @property
    def n_metabolites(self) -> int:
        return int(self.metabolites.shape[0])

    @property
    def n_species(self) -> int:
        return int(self.biomass.shape[0])

    @property
    def n_grid(self) -> int:
        return int(self.metabolites.shape[1])

    @classmethod
    def from_arrays(
        cls,
        *,
        C: np.ndarray,
        B: np.ndarray,
        t: float = 0.0,
    ) -> "SpatialState":
        """Construct with explicit dtype coercion to float64."""
        return cls(
            metabolites=np.asarray(C, dtype=np.float64),
            biomass=np.asarray(B, dtype=np.float64),
            t=t,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_state.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/state.py tests/spatial/test_state.py
git commit -m "feat(spatial): add SpatialState dataclass with shape validation"
```

---

## Task 3: Geometry1D — grid, positions, dx

**Files:**
- Create: `src/gemfitcom/spatial/geometry.py`
- Test: `tests/spatial/test_geometry.py`

Basic 1D grid (no boundary handling yet — that's Task 4).

- [ ] **Step 1: Write failing tests**

```python
# tests/spatial/test_geometry.py
"""Tests for Geometry1D and BoundarySpec."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.geometry import Geometry1D


def test_geometry_dx_from_length_and_n_grid() -> None:
    geom = Geometry1D(n_grid=11, length=1.0)
    # 11 nodes, 10 intervals -> dx = 0.1
    assert geom.dx == pytest.approx(0.1)


def test_geometry_positions_are_uniform_from_zero_to_length() -> None:
    geom = Geometry1D(n_grid=5, length=2.0)
    np.testing.assert_allclose(geom.positions, [0.0, 0.5, 1.0, 1.5, 2.0])


def test_geometry_n_grid_must_be_at_least_2() -> None:
    with pytest.raises(ValueError, match="n_grid"):
        Geometry1D(n_grid=1, length=1.0)


def test_geometry_length_must_be_positive() -> None:
    with pytest.raises(ValueError, match="length"):
        Geometry1D(n_grid=10, length=0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_geometry.py`
Expected: ImportError.

- [ ] **Step 3: Implement Geometry1D (basic)**

```python
# src/gemfitcom/spatial/geometry.py
"""Geometry1D: 1D mucosa-lumen grid with boundary specifications."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Geometry1D:
    """Uniform 1D grid from mucosa (index 0) to lumen (index n_grid - 1)."""

    n_grid: int
    length: float

    def __post_init__(self) -> None:
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2; got {self.n_grid}")
        if self.length <= 0:
            raise ValueError(f"length must be > 0; got {self.length}")

    @property
    def dx(self) -> float:
        return self.length / (self.n_grid - 1)

    @property
    def positions(self) -> np.ndarray:
        return np.linspace(0.0, self.length, self.n_grid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_geometry.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/geometry.py tests/spatial/test_geometry.py
git commit -m "feat(spatial): add Geometry1D with grid spacing and node positions"
```

---

## Task 4: BoundarySpec + Geometry1D boundary application

**Files:**
- Modify: `src/gemfitcom/spatial/geometry.py` (add `BoundarySpec`, extend `Geometry1D`)
- Modify: `tests/spatial/test_geometry.py` (append boundary tests)

Boundary semantics for PR 1 (will be wired into the simulator in PR 3):

- `reflecting`: no-flux Neumann BC. Diffusion handles this naturally (Task 5). `apply_boundary_sources` is a no-op on this side.
- `flux`: add a source term. The user supplies values keyed by exchange reaction ID (e.g. `EX_o2_e` → `1e-3`); we strip the `EX_` prefix to get the metabolite ID (`o2_e`) and add `flux * dt` to the boundary cell.
- `dirichlet`: pin the boundary cell's concentration to the given value (overwrites whatever diffusion would have done).

**Unit convention for PR 1 (documented; refined in later PRs):** flux values are in `mmol/(L·h)` (added directly to a `mmol/L` cell over time `h`); Dirichlet values are in `mmol/L`. No physical unit conversion — that lives in a future PR.

- [ ] **Step 1: Write failing tests (append to existing file)**

Append to `tests/spatial/test_geometry.py`:

```python
from gemfitcom.spatial.geometry import BoundarySpec


def test_boundary_spec_reflecting_default_empty_values() -> None:
    bc = BoundarySpec(type="reflecting")
    assert bc.values == {}


def test_boundary_spec_flux_holds_source_dict() -> None:
    bc = BoundarySpec(type="flux", values={"EX_o2_e": 1.0e-3})
    assert bc.values["EX_o2_e"] == 1.0e-3


def test_boundary_spec_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="type"):
        BoundarySpec(type="bogus")  # type: ignore[arg-type]


def test_apply_boundary_sources_flux_adds_to_left_cell() -> None:
    geom = Geometry1D(
        n_grid=5,
        length=1.0,
        bc_left=BoundarySpec(type="flux", values={"EX_o2_e": 2.0}),
        bc_right=BoundarySpec(type="reflecting"),
    )
    C = np.zeros((1, 5))
    geom.apply_boundary_sources(C, metabolite_ids=["o2_e"], dt=0.5)
    # 2.0 mmol/(L·h) * 0.5 h = 1.0 mmol/L added to left cell only
    np.testing.assert_allclose(C[0], [1.0, 0.0, 0.0, 0.0, 0.0])


def test_apply_boundary_sources_dirichlet_pins_right_cell() -> None:
    geom = Geometry1D(
        n_grid=5,
        length=1.0,
        bc_left=BoundarySpec(type="reflecting"),
        bc_right=BoundarySpec(type="dirichlet", values={"EX_glc__D_e": 5.0}),
    )
    C = np.full((1, 5), 1.0)
    geom.apply_boundary_sources(C, metabolite_ids=["glc__D_e"], dt=0.1)
    # Dirichlet pins the right cell to 5.0, leaves others alone
    np.testing.assert_allclose(C[0], [1.0, 1.0, 1.0, 1.0, 5.0])


def test_apply_boundary_sources_unknown_metabolite_skipped() -> None:
    geom = Geometry1D(
        n_grid=3,
        length=1.0,
        bc_left=BoundarySpec(type="flux", values={"EX_xyz_e": 99.0}),
        bc_right=BoundarySpec(type="reflecting"),
    )
    C = np.zeros((1, 3))
    geom.apply_boundary_sources(C, metabolite_ids=["o2_e"], dt=1.0)
    # xyz_e not in metabolite_ids -> silently skipped
    np.testing.assert_allclose(C, np.zeros((1, 3)))


def test_apply_boundary_sources_rejects_non_exchange_key() -> None:
    geom = Geometry1D(
        n_grid=3,
        length=1.0,
        bc_left=BoundarySpec(type="flux", values={"o2_e": 1.0}),  # missing EX_ prefix
        bc_right=BoundarySpec(type="reflecting"),
    )
    C = np.zeros((1, 3))
    with pytest.raises(ValueError, match="EX_"):
        geom.apply_boundary_sources(C, metabolite_ids=["o2_e"], dt=1.0)


def test_geometry_default_boundaries_are_reflecting() -> None:
    geom = Geometry1D(n_grid=5, length=1.0)
    assert geom.bc_left.type == "reflecting"
    assert geom.bc_right.type == "reflecting"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_geometry.py`
Expected: 8 failures / errors (BoundarySpec doesn't exist; Geometry1D doesn't take bc_left/bc_right).

- [ ] **Step 3: Extend `geometry.py`**

Replace the contents of `src/gemfitcom/spatial/geometry.py` with:

```python
# src/gemfitcom/spatial/geometry.py
"""Geometry1D: 1D mucosa-lumen grid with boundary specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BCType = Literal["flux", "dirichlet", "reflecting"]
_VALID_BC_TYPES: frozenset[str] = frozenset({"flux", "dirichlet", "reflecting"})


@dataclass
class BoundarySpec:
    """Boundary condition spec for one side of the 1D grid.

    type:
        - 'reflecting': no-flux Neumann (default; values must be empty)
        - 'flux':       add `value * dt` to the boundary cell each step
                        (value units: mmol/(L*h) for PR 1; physical conversion deferred)
        - 'dirichlet':  pin boundary cell to `value` (mmol/L)

    values: mapping from exchange-reaction ID (e.g. 'EX_o2_e') to numeric value.
            Keys must start with 'EX_'; the metabolite ID is the suffix ('o2_e').
    """

    type: BCType
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in _VALID_BC_TYPES:
            raise ValueError(
                f"BoundarySpec.type must be one of {sorted(_VALID_BC_TYPES)}; got {self.type!r}"
            )


def _ex_to_metabolite_id(key: str) -> str:
    if not key.startswith("EX_"):
        raise ValueError(
            f"Boundary key must start with 'EX_' (exchange reaction ID); got {key!r}"
        )
    return key[3:]


@dataclass
class Geometry1D:
    """Uniform 1D grid from mucosa (index 0) to lumen (index n_grid - 1)."""

    n_grid: int
    length: float
    bc_left: BoundarySpec = field(default_factory=lambda: BoundarySpec(type="reflecting"))
    bc_right: BoundarySpec = field(default_factory=lambda: BoundarySpec(type="reflecting"))

    def __post_init__(self) -> None:
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2; got {self.n_grid}")
        if self.length <= 0:
            raise ValueError(f"length must be > 0; got {self.length}")

    @property
    def dx(self) -> float:
        return self.length / (self.n_grid - 1)

    @property
    def positions(self) -> np.ndarray:
        return np.linspace(0.0, self.length, self.n_grid)

    def apply_boundary_sources(
        self,
        C: np.ndarray,
        metabolite_ids: list[str],
        dt: float,
    ) -> None:
        """In-place: apply flux sources and Dirichlet pinning to boundary cells.

        Reflecting BCs are no-ops here; the diffusion operator (Task 5) handles
        the no-flux condition by construction.
        """
        for bc, idx in ((self.bc_left, 0), (self.bc_right, -1)):
            if bc.type == "reflecting":
                continue
            for ex_id, value in bc.values.items():
                met_id = _ex_to_metabolite_id(ex_id)
                if met_id not in metabolite_ids:
                    continue
                j = metabolite_ids.index(met_id)
                if bc.type == "flux":
                    C[j, idx] += value * dt
                elif bc.type == "dirichlet":
                    C[j, idx] = value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_geometry.py`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/geometry.py tests/spatial/test_geometry.py
git commit -m "feat(spatial): add BoundarySpec + flux/dirichlet/reflecting application"
```

---

## Task 5: Diffusion — sparse Laplacian builder

**Files:**
- Create: `src/gemfitcom/spatial/diffusion.py`
- Test: `tests/spatial/test_diffusion.py`

Build the second-difference operator `(1/dx²) * tridiag(1, -2, 1)` with Neumann (no-flux) BC by default; Dirichlet zeros out the boundary row so external pinning isn't undone.

- [ ] **Step 1: Write failing tests**

```python
# tests/spatial/test_diffusion.py
"""Tests for the diffusion operator and step function."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from gemfitcom.spatial.diffusion import build_laplacian_1d


def test_laplacian_returns_csr_matrix() -> None:
    L = build_laplacian_1d(n_grid=5, dx=0.1)
    assert sp.isspmatrix_csr(L)
    assert L.shape == (5, 5)


def test_laplacian_neumann_interior_row_is_standard_stencil() -> None:
    L = build_laplacian_1d(n_grid=5, dx=2.0)
    # Interior row i should be [..., 1, -2, 1, ...] / dx**2
    expected_row = np.array([0.0, 1.0, -2.0, 1.0, 0.0]) / 4.0
    np.testing.assert_allclose(L.toarray()[2], expected_row)


def test_laplacian_neumann_left_boundary_is_one_sided() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0, bc_left="neumann", bc_right="neumann")
    # Reflecting (ghost = interior): row 0 -> (C[1] - C[0]) / dx**2
    expected_row = np.array([-1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(L.toarray()[0], expected_row)


def test_laplacian_neumann_right_boundary_is_one_sided() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0)
    expected_row = np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    np.testing.assert_allclose(L.toarray()[-1], expected_row)


def test_laplacian_dirichlet_boundary_row_is_zero() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0, bc_left="dirichlet", bc_right="neumann")
    # Dirichlet: external code pins the boundary, so the operator must not move it
    np.testing.assert_allclose(L.toarray()[0], np.zeros(5))


def test_laplacian_invalid_bc_raises() -> None:
    with pytest.raises(ValueError, match="bc_left"):
        build_laplacian_1d(n_grid=5, dx=1.0, bc_left="bogus")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_diffusion.py`
Expected: ImportError.

- [ ] **Step 3: Implement Laplacian builder**

```python
# src/gemfitcom/spatial/diffusion.py
"""Diffusion: sparse Laplacian + explicit FTCS step + CFL helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp

LaplacianBC = Literal["neumann", "dirichlet"]
_VALID_LAPLACIAN_BCS: frozenset[str] = frozenset({"neumann", "dirichlet"})


def build_laplacian_1d(
    n_grid: int,
    dx: float,
    bc_left: LaplacianBC = "neumann",
    bc_right: LaplacianBC = "neumann",
) -> sp.csr_matrix:
    """Construct (1/dx**2) * second-difference operator for a 1D grid.

    Neumann (reflecting) BC: ghost cell equals the boundary cell; row reduces
    to a one-sided difference, giving zero net flux through the wall.

    Dirichlet BC: row is zeroed out, so the operator does NOT update the
    boundary cell (caller pins the value externally each step).
    """
    if bc_left not in _VALID_LAPLACIAN_BCS:
        raise ValueError(
            f"bc_left must be one of {sorted(_VALID_LAPLACIAN_BCS)}; got {bc_left!r}"
        )
    if bc_right not in _VALID_LAPLACIAN_BCS:
        raise ValueError(
            f"bc_right must be one of {sorted(_VALID_LAPLACIAN_BCS)}; got {bc_right!r}"
        )

    main = -2.0 * np.ones(n_grid)
    off = np.ones(n_grid - 1)
    L = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(n_grid, n_grid), format="lil")

    if bc_left == "neumann":
        L[0, 0] = -1.0
        L[0, 1] = 1.0
    else:  # dirichlet
        L[0, 0] = 0.0
        L[0, 1] = 0.0

    if bc_right == "neumann":
        L[-1, -1] = -1.0
        L[-1, -2] = 1.0
    else:  # dirichlet
        L[-1, -1] = 0.0
        L[-1, -2] = 0.0

    return (L / (dx * dx)).tocsr()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_diffusion.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/diffusion.py tests/spatial/test_diffusion.py
git commit -m "feat(spatial): add sparse 1D Laplacian builder with Neumann/Dirichlet BC"
```

---

## Task 6: Diffusion — explicit FTCS step function

**Files:**
- Modify: `src/gemfitcom/spatial/diffusion.py` (add `diffuse_step`)
- Modify: `tests/spatial/test_diffusion.py` (append step tests)

Vectorized step: `C_new = C + dt * D[:, None] * (L @ C.T).T`. Each metabolite gets its own `D`; sparse-matrix-times-dense uses SciPy's optimized C path.

- [ ] **Step 1: Write failing tests (append)**

Append to `tests/spatial/test_diffusion.py`:

```python
from gemfitcom.spatial.diffusion import diffuse_step


def test_diffuse_step_constant_field_unchanged() -> None:
    L = build_laplacian_1d(n_grid=10, dx=0.1)
    C = np.full((1, 10), 3.7)
    D = np.array([1e-3])
    out = diffuse_step(C, L, D, dt=0.5)
    np.testing.assert_allclose(out, C, atol=1e-12)


def test_diffuse_step_returns_new_array_does_not_mutate_input() -> None:
    L = build_laplacian_1d(n_grid=5, dx=0.1)
    C = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    D = np.array([1e-3])
    C_before = C.copy()
    _ = diffuse_step(C, L, D, dt=0.1)
    np.testing.assert_allclose(C, C_before)


def test_diffuse_step_pulse_spreads_to_neighbors() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0)
    C = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    D = np.array([0.1])
    out = diffuse_step(C, L, D, dt=1.0)
    # Center loses to neighbors, edges still 0
    assert out[0, 2] < 1.0
    assert out[0, 1] > 0.0
    assert out[0, 3] > 0.0
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[0, 4], 0.0)


def test_diffuse_step_independent_metabolites() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0)
    C = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],   # met 0: pulse in middle
            [1.0, 1.0, 1.0, 1.0, 1.0],   # met 1: uniform
        ]
    )
    D = np.array([0.1, 0.5])
    out = diffuse_step(C, L, D, dt=1.0)
    # Met 0 spreads; met 1 unchanged regardless of D
    assert out[0, 2] < 1.0
    np.testing.assert_allclose(out[1], C[1])


def test_diffuse_step_neumann_conserves_total_one_step() -> None:
    L = build_laplacian_1d(n_grid=20, dx=0.05)
    C = np.zeros((1, 20))
    C[0, 10] = 1.0
    D = np.array([1e-3])
    out = diffuse_step(C, L, D, dt=0.1)
    assert out.sum() == pytest.approx(C.sum(), abs=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_diffusion.py`
Expected: 5 new failures (`diffuse_step` not defined).

- [ ] **Step 3: Add `diffuse_step` to `diffusion.py`**

Append to `src/gemfitcom/spatial/diffusion.py`:

```python
def diffuse_step(
    C: np.ndarray,
    L: sp.csr_matrix,
    D: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Explicit FTCS diffusion step.

    Args:
        C: shape (n_metabolites, n_grid), current concentrations.
        L: sparse (n_grid, n_grid) Laplacian from `build_laplacian_1d`.
        D: shape (n_metabolites,), per-metabolite diffusion coefficients.
        dt: time step.

    Returns:
        New concentration array of shape (n_metabolites, n_grid). Does not
        mutate the input.
    """
    # L @ C.T has shape (n_grid, n_metabolites); transpose back to match C.
    return C + dt * D[:, None] * (L @ C.T).T
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_diffusion.py`
Expected: 11 passed (6 from Task 5 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/diffusion.py tests/spatial/test_diffusion.py
git commit -m "feat(spatial): add explicit FTCS diffuse_step (vectorized over metabolites)"
```

---

## Task 7: CFL stability check

**Files:**
- Modify: `src/gemfitcom/spatial/diffusion.py` (add `cfl_dt_max`, `check_cfl`)
- Modify: `tests/spatial/test_diffusion.py` (append CFL tests)

Explicit FTCS in 1D requires `dt ≤ dx² / (2D)`. We add a `safety` factor (default 0.4) and provide both a query function and a guard.

- [ ] **Step 1: Write failing tests (append)**

Append to `tests/spatial/test_diffusion.py`:

```python
from gemfitcom.spatial.diffusion import cfl_dt_max, check_cfl


def test_cfl_dt_max_formula() -> None:
    # safety=1 corresponds to the marginal CFL limit
    assert cfl_dt_max(dx=0.1, D_max=1e-3, safety=1.0) == pytest.approx(0.01 / 2e-3)


def test_cfl_dt_max_default_safety_is_0_4() -> None:
    assert cfl_dt_max(dx=0.1, D_max=1e-3) == pytest.approx(0.4 * 0.01 / 2e-3)


def test_check_cfl_passes_when_dt_within_limit() -> None:
    check_cfl(dt=1.0, dx=0.1, D_max=1e-3, safety=0.4)  # well below limit


def test_check_cfl_raises_when_dt_too_large() -> None:
    with pytest.raises(RuntimeError, match="CFL"):
        check_cfl(dt=100.0, dx=0.1, D_max=1.0, safety=0.4)


def test_check_cfl_error_message_includes_suggested_dt() -> None:
    with pytest.raises(RuntimeError, match=r"Reduce dt to"):
        check_cfl(dt=100.0, dx=0.1, D_max=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_diffusion.py`
Expected: 5 new failures.

- [ ] **Step 3: Add CFL helpers**

Append to `src/gemfitcom/spatial/diffusion.py`:

```python
def cfl_dt_max(dx: float, D_max: float, safety: float = 0.4) -> float:
    """Maximum stable dt for explicit FTCS in 1D.

    The strict limit is dt <= dx**2 / (2 * D_max); `safety` (default 0.4)
    pulls back from the boundary for robustness.
    """
    if D_max <= 0:
        return float("inf")
    return safety * dx * dx / (2.0 * D_max)


def check_cfl(dt: float, dx: float, D_max: float, safety: float = 0.4) -> None:
    """Raise RuntimeError if `dt` violates the CFL stability limit."""
    dt_limit = cfl_dt_max(dx, D_max, safety)
    if dt > dt_limit:
        raise RuntimeError(
            f"dt={dt} violates CFL stability "
            f"(dx={dx}, D_max={D_max}, safety={safety}). "
            f"Reduce dt to <= {dt_limit:.4g} or coarsen the grid."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_diffusion.py`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/diffusion.py tests/spatial/test_diffusion.py
git commit -m "feat(spatial): add cfl_dt_max + check_cfl stability guards"
```

---

## Task 8: Diffusion accuracy — analytical Gaussian comparison

**Files:**
- Modify: `tests/spatial/test_diffusion.py` (append accuracy test)

No new production code. This is a quality test: a Gaussian initial condition diffuses into a wider Gaussian with `σ(t) = sqrt(σ₀² + 2Dt)` in an infinite medium. Use a domain wide enough that the pulse hasn't reached the boundaries.

- [ ] **Step 1: Write the accuracy test (append)**

Append to `tests/spatial/test_diffusion.py`:

```python
def test_diffuse_step_matches_analytical_gaussian() -> None:
    """Gaussian initial condition → Gaussian solution, sigma grows as sqrt(sigma0**2 + 2Dt).

    Domain is wide enough that the pulse never reaches the boundaries during
    the test window, so closed BCs don't pollute the comparison.
    """
    n_grid = 201
    length = 1.0
    dx = length / (n_grid - 1)
    D = 1.0e-3
    sigma0 = 0.05
    center = 0.5
    t_end = 0.1

    x = np.linspace(0.0, length, n_grid)
    C = np.exp(-((x - center) ** 2) / (2.0 * sigma0 ** 2))[None, :]  # shape (1, n_grid)

    L = build_laplacian_1d(n_grid, dx)
    D_arr = np.array([D])
    dt = cfl_dt_max(dx, D, safety=0.4)
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # exact match for t_end

    for _ in range(n_steps):
        C = diffuse_step(C, L, D_arr, dt)

    sigma_t = np.sqrt(sigma0 ** 2 + 2.0 * D * t_end)
    expected = (sigma0 / sigma_t) * np.exp(-((x - center) ** 2) / (2.0 * sigma_t ** 2))

    rel_error = np.max(np.abs(C[0] - expected)) / np.max(expected)
    assert rel_error < 0.01, f"max relative error {rel_error:.4g} exceeds 1%"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/spatial/test_diffusion.py::test_diffuse_step_matches_analytical_gaussian`
Expected: PASS (relative error well under 1%; FTCS is O(dx², dt)).

- [ ] **Step 3: Commit**

```bash
git add tests/spatial/test_diffusion.py
git commit -m "test(spatial): verify diffusion matches analytical Gaussian solution"
```

---

## Task 9: Mass conservation invariant

**Files:**
- Create: `tests/spatial/test_invariant_mass_conservation.py`

Closed (reflecting) BC + many steps → total mass `∫C dx ≈ Σ C * dx` is conserved.

- [ ] **Step 1: Write the invariant test**

```python
# tests/spatial/test_invariant_mass_conservation.py
"""Long-run pure diffusion preserves total mass under reflecting BCs."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.diffusion import (
    build_laplacian_1d,
    cfl_dt_max,
    diffuse_step,
)


def test_mass_conservation_1000_steps_neumann() -> None:
    n_grid = 50
    length = 1.0e-3
    dx = length / (n_grid - 1)
    D = 2.1e-9
    L = build_laplacian_1d(n_grid, dx, bc_left="neumann", bc_right="neumann")

    rng = np.random.default_rng(seed=42)
    C = rng.uniform(low=0.5, high=1.5, size=(2, n_grid))
    D_arr = np.array([D, D * 5])

    initial_mass = C.sum(axis=1) * dx
    dt = cfl_dt_max(dx, D_arr.max(), safety=0.4)

    for _ in range(1000):
        C = diffuse_step(C, L, D_arr, dt)

    final_mass = C.sum(axis=1) * dx
    rel_drift = np.abs(final_mass - initial_mass) / initial_mass
    assert np.max(rel_drift) < 1e-10, f"mass drift {rel_drift} exceeds 1e-10"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/spatial/test_invariant_mass_conservation.py`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/spatial/test_invariant_mass_conservation.py
git commit -m "test(spatial): pure-diffusion mass conservation invariant (Neumann BC)"
```

---

## Task 10: Positivity invariant

**Files:**
- Create: `tests/spatial/test_invariant_positivity.py`

Random nonnegative IC + many steps → concentrations stay ≥ 0 (allowing tiny float noise).

- [ ] **Step 1: Write the invariant test**

```python
# tests/spatial/test_invariant_positivity.py
"""Pure diffusion preserves nonnegativity within floating-point tolerance."""

from __future__ import annotations

import numpy as np

from gemfitcom.spatial.diffusion import (
    build_laplacian_1d,
    cfl_dt_max,
    diffuse_step,
)


def test_positivity_random_initial_conditions_neumann() -> None:
    n_grid = 30
    length = 1.0
    dx = length / (n_grid - 1)
    D = 1.0e-2
    L = build_laplacian_1d(n_grid, dx, bc_left="neumann", bc_right="neumann")

    rng = np.random.default_rng(seed=7)
    C = rng.uniform(low=0.0, high=1.0, size=(3, n_grid))
    D_arr = np.full(3, D)
    dt = cfl_dt_max(dx, D, safety=0.4)

    for _ in range(500):
        C = diffuse_step(C, L, D_arr, dt)
        assert C.min() >= -1e-12, f"negative concentration {C.min()} at step"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/spatial/test_invariant_positivity.py`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/spatial/test_invariant_positivity.py
git commit -m "test(spatial): pure-diffusion positivity invariant"
```

---

## Task 11: SnapshotRecorder

**Files:**
- Create: `src/gemfitcom/spatial/recorder.py`
- Create: `tests/spatial/test_recorder.py`

`save` writes one `.npz` per snapshot named `snapshot_t={t:.4f}.npz`. `maybe_save` saves only when `state.t - last_save >= every` (with a tiny tolerance for float math). `load` round-trips losslessly back to a `SpatialState`.

- [ ] **Step 1: Write failing tests**

```python
# tests/spatial/test_recorder.py
"""Tests for SnapshotRecorder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gemfitcom.spatial.recorder import SnapshotRecorder
from gemfitcom.spatial.state import SpatialState


def _make_state(t: float = 0.0) -> SpatialState:
    return SpatialState(
        metabolites=np.array([[1.0, 2.0, 3.0]]),
        biomass=np.array([[0.1, 0.2, 0.3]]),
        t=t,
    )


def test_recorder_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "snapshots"
    SnapshotRecorder(output_dir=out, every=1.0)
    assert out.is_dir()


def test_save_writes_npz_with_state_data(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    state = _make_state(t=2.5)
    path = rec.save(state)
    assert path.exists()
    assert "t=2.5000" in path.name
    data = np.load(path)
    assert float(data["t"]) == pytest.approx(2.5)
    np.testing.assert_allclose(data["metabolites"], state.metabolites, rtol=1e-6)
    np.testing.assert_allclose(data["biomass"], state.biomass, rtol=1e-6)


def test_load_roundtrip(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0, precision="float64")
    original = _make_state(t=1.25)
    path = rec.save(original)
    restored = SnapshotRecorder.load(path)
    np.testing.assert_array_equal(restored.metabolites, original.metabolites)
    np.testing.assert_array_equal(restored.biomass, original.biomass)
    assert restored.t == original.t


def test_maybe_save_first_call_always_saves(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    path = rec.maybe_save(_make_state(t=0.0))
    assert path is not None and path.exists()


def test_maybe_save_skips_when_too_soon(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    rec.maybe_save(_make_state(t=0.0))
    skipped = rec.maybe_save(_make_state(t=0.5))
    assert skipped is None


def test_maybe_save_resumes_after_interval(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    rec.maybe_save(_make_state(t=0.0))
    rec.maybe_save(_make_state(t=0.5))
    saved = rec.maybe_save(_make_state(t=1.0))
    assert saved is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_recorder.py`
Expected: ImportError.

- [ ] **Step 3: Implement SnapshotRecorder**

```python
# src/gemfitcom/spatial/recorder.py
"""SnapshotRecorder: periodic .npz dumps of SpatialState during a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .state import SpatialState

Precision = Literal["float32", "float64"]


@dataclass
class SnapshotRecorder:
    """Save SpatialState snapshots to disk at a configurable cadence.

    Snapshots are written as compressed .npz files named 'snapshot_t={t:.4f}.npz'.
    """

    output_dir: Path
    every: float
    precision: Precision = "float32"
    _last_save: float = field(default=-float("inf"), init=False, repr=False)
    _saved_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.every <= 0:
            raise ValueError(f"every must be > 0; got {self.every}")

    def save(self, state: SpatialState) -> Path:
        """Force-save the current state. Returns the file path written."""
        path = self.output_dir / f"snapshot_t={state.t:.4f}.npz"
        dtype = np.dtype(self.precision)
        np.savez_compressed(
            path,
            t=np.float64(state.t),
            metabolites=state.metabolites.astype(dtype),
            biomass=state.biomass.astype(dtype),
        )
        self._last_save = state.t
        self._saved_count += 1
        return path

    def maybe_save(self, state: SpatialState) -> Path | None:
        """Save only if `every` hours have elapsed since the last save.

        First call always saves (last_save starts at -inf).
        """
        # Tiny tolerance to handle float accumulation: 1e-12 is well below any
        # realistic dt and avoids missing a save by a billionth of an hour.
        if state.t - self._last_save >= self.every - 1e-12:
            return self.save(state)
        return None

    @staticmethod
    def load(path: Path) -> SpatialState:
        """Load a snapshot back into a SpatialState (always float64)."""
        data = np.load(path)
        return SpatialState(
            metabolites=data["metabolites"].astype(np.float64),
            biomass=data["biomass"].astype(np.float64),
            t=float(data["t"]),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_recorder.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/recorder.py tests/spatial/test_recorder.py
git commit -m "feat(spatial): add SnapshotRecorder with maybe_save cadence + npz round-trip"
```

---

## Task 12: Config — pydantic models

**Files:**
- Create: `src/gemfitcom/spatial/config.py`
- Create: `tests/spatial/test_config.py`

PR 1 covers the geometry, metabolites, simulation, and output sections only. `species`, `kinetics`, and `backend` are PR 2/PR 4 additions and are intentionally absent from the schema.

- [ ] **Step 1: Write failing tests**

```python
# tests/spatial/test_config.py
"""Tests for the SpatialConfig pydantic schema (PR 1 surface)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gemfitcom.spatial.config import (
    BoundaryConfig,
    GeometryConfig,
    InitConfig,
    MetaboliteConfig,
    OutputConfig,
    SimulationConfig,
    SpatialConfig,
)


# ---- Component schemas ----


def test_geometry_config_minimum_valid() -> None:
    cfg = GeometryConfig(
        n_grid=50,
        length=1e-3,
        boundary={
            "mucosa": BoundaryConfig(type="reflecting"),
            "lumen": BoundaryConfig(type="reflecting"),
        },
    )
    assert cfg.n_grid == 50


def test_geometry_config_n_grid_must_be_above_one() -> None:
    with pytest.raises(ValidationError):
        GeometryConfig(n_grid=1, length=1e-3, boundary={})


def test_geometry_config_length_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        GeometryConfig(n_grid=10, length=0.0, boundary={})


def test_metabolite_config_requires_diffusion_and_init() -> None:
    cfg = MetaboliteConfig(
        id="o2_e",
        diffusion=2.1e-9,
        init=InitConfig(mode="uniform", value=0.21),
    )
    assert cfg.id == "o2_e"


def test_metabolite_config_negative_diffusion_rejected() -> None:
    with pytest.raises(ValidationError):
        MetaboliteConfig(
            id="o2_e",
            diffusion=-1.0,
            init=InitConfig(mode="uniform", value=0.0),
        )


def test_simulation_config_requires_positive_dt_and_t_end() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(t_end=0.0, dt=0.1, snapshot_every=1.0)
    with pytest.raises(ValidationError):
        SimulationConfig(t_end=1.0, dt=0.0, snapshot_every=1.0)


def test_simulation_config_default_cfl_safety_is_0_4() -> None:
    cfg = SimulationConfig(t_end=1.0, dt=0.1, snapshot_every=0.5)
    assert cfg.cfl_safety == 0.4


def test_output_config_defaults() -> None:
    cfg = OutputConfig()
    assert cfg.format == "npz"
    assert cfg.precision == "float32"


# ---- Top-level + YAML round-trip ----


def _minimal_config_dict() -> dict:
    return {
        "geometry": {
            "n_grid": 11,
            "length": 1.0e-3,
            "boundary": {
                "mucosa": {"type": "flux", "sources": {"EX_o2_e": 1.0e-3}},
                "lumen": {"type": "dirichlet", "values": {"EX_glc__D_e": 5.0}},
            },
        },
        "metabolites": [
            {"id": "o2_e", "diffusion": 2.1e-9, "init": {"mode": "uniform", "value": 0.21}},
            {"id": "glc__D_e", "diffusion": 6.7e-10, "init": {"mode": "uniform", "value": 0.0}},
        ],
        "simulation": {"t_end": 24.0, "dt": 0.1, "snapshot_every": 1.0},
    }


def test_spatial_config_parses_minimal_dict() -> None:
    cfg = SpatialConfig(**_minimal_config_dict())
    assert cfg.geometry.n_grid == 11
    assert len(cfg.metabolites) == 2
    assert cfg.simulation.t_end == 24.0
    assert cfg.output.format == "npz"  # default


def test_spatial_config_from_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "sim.yaml"
    yaml_path.write_text(yaml.safe_dump(_minimal_config_dict()))
    cfg = SpatialConfig.from_yaml(yaml_path)
    assert cfg.geometry.n_grid == 11


def test_spatial_config_missing_required_field_raises() -> None:
    bad = _minimal_config_dict()
    del bad["simulation"]
    with pytest.raises(ValidationError):
        SpatialConfig(**bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_config.py`
Expected: ImportError.

- [ ] **Step 3: Implement config models**

```python
# src/gemfitcom/spatial/config.py
"""Pydantic schema for the spatial simulation YAML config (PR 1 surface).

Sections covered:
    - geometry
    - metabolites
    - simulation
    - output

Sections deferred to later PRs:
    - species   (PR 2)
    - kinetics  (PR 2)
    - backend   (PR 4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BoundaryConfig(BaseModel):
    """One side of the geometry boundary."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["flux", "dirichlet", "reflecting"]
    sources: dict[str, float] = Field(default_factory=dict)
    values: dict[str, float] = Field(default_factory=dict)


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dim: Literal[1] = 1
    n_grid: int = Field(gt=1)
    length: float = Field(gt=0.0)
    boundary: dict[str, BoundaryConfig]


class InitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["uniform", "gaussian", "step", "from_array"]
    value: float | None = None
    center: float | None = None
    sigma: float | None = None
    peak: float | None = None
    path: Path | None = None


class MetaboliteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    diffusion: float = Field(ge=0.0)
    init: InitConfig


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_end: float = Field(gt=0.0)
    dt: float = Field(gt=0.0)
    snapshot_every: float = Field(gt=0.0)
    cfl_safety: float = Field(default=0.4, gt=0.0, le=1.0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["npz", "netcdf"] = "npz"
    precision: Literal["float32", "float64"] = "float32"


class SpatialConfig(BaseModel):
    """Top-level spatial config (PR 1 surface)."""

    model_config = ConfigDict(extra="forbid")

    geometry: GeometryConfig
    metabolites: list[MetaboliteConfig]
    simulation: SimulationConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "SpatialConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/spatial/test_config.py`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/config.py tests/spatial/test_config.py
git commit -m "feat(spatial): add pydantic SpatialConfig schema (PR 1 surface: geometry/metabolites/simulation/output)"
```

---

## Task 13: Config — CFL stability check method

**Files:**
- Modify: `src/gemfitcom/spatial/config.py` (add `check_cfl` method)
- Modify: `tests/spatial/test_config.py` (append CFL tests)

A first-class method on `SpatialConfig` that uses `diffusion.cfl_dt_max` so users get one-call stability validation before launching a long run.

- [ ] **Step 1: Write failing tests (append)**

Append to `tests/spatial/test_config.py`:

```python
def test_spatial_config_check_cfl_passes_when_safe() -> None:
    cfg_dict = _minimal_config_dict()
    cfg_dict["simulation"]["dt"] = 1e-6  # tiny dt -> safe
    SpatialConfig(**cfg_dict).check_cfl()


def test_spatial_config_check_cfl_raises_when_dt_too_large() -> None:
    cfg_dict = _minimal_config_dict()
    cfg_dict["simulation"]["dt"] = 1.0e6  # absurdly large dt
    with pytest.raises(RuntimeError, match="CFL"):
        SpatialConfig(**cfg_dict).check_cfl()


def test_spatial_config_check_cfl_no_op_when_all_diffusion_zero() -> None:
    cfg_dict = _minimal_config_dict()
    for met in cfg_dict["metabolites"]:
        met["diffusion"] = 0.0
    cfg_dict["simulation"]["dt"] = 1.0e6  # would violate CFL if any D > 0
    SpatialConfig(**cfg_dict).check_cfl()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/spatial/test_config.py`
Expected: 3 new failures (`check_cfl` not defined).

- [ ] **Step 3: Add `check_cfl` to `SpatialConfig`**

In `src/gemfitcom/spatial/config.py`, add the import at the top:

```python
from .diffusion import cfl_dt_max
```

Then add the method to the `SpatialConfig` class:

```python
    def check_cfl(self) -> None:
        """Raise RuntimeError if the configured dt violates CFL stability.

        Uses the largest diffusion coefficient across metabolites to set the
        bound. If all diffusion coefficients are zero, the check is a no-op.
        """
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

Run: `pytest tests/spatial/test_config.py`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gemfitcom/spatial/config.py tests/spatial/test_config.py
git commit -m "feat(spatial): add SpatialConfig.check_cfl convenience method"
```

---

## Task 14: Public API surface + integration smoke

**Files:**
- Modify: `src/gemfitcom/spatial/__init__.py` (export PR 1 names)
- Create: `tests/spatial/test_public_api.py` (assert imports + tiny integration)

Closes PR 1 with a single test that imports everything via the package surface and runs one realistic mini scenario: load YAML → build geometry → run 10 pure-diffusion steps with boundary sources → save snapshot → reload.

- [ ] **Step 1: Update the public API**

Replace `src/gemfitcom/spatial/__init__.py`:

```python
# src/gemfitcom/spatial/__init__.py
"""Spatial dFBA subpackage for gemfitcom.

PR 1 surface: numerical core (state, geometry, diffusion, recorder, config).
FBA coupling, simulator, CLI, and parallel backends added in subsequent PRs.
"""

from .config import (
    BoundaryConfig,
    GeometryConfig,
    InitConfig,
    MetaboliteConfig,
    OutputConfig,
    SimulationConfig,
    SpatialConfig,
)
from .diffusion import build_laplacian_1d, cfl_dt_max, check_cfl, diffuse_step
from .geometry import BoundarySpec, Geometry1D
from .recorder import SnapshotRecorder
from .state import SpatialState

__all__ = [
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
]
```

- [ ] **Step 2: Write the smoke test**

```python
# tests/spatial/test_public_api.py
"""Public API: verify all PR 1 names import and a tiny end-to-end run works."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from gemfitcom import spatial


def test_public_api_exports_pr1_surface() -> None:
    expected = {
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
    }
    assert expected.issubset(set(spatial.__all__))
    for name in expected:
        assert hasattr(spatial, name), f"missing public symbol: {name}"


def test_pr1_integration_smoke(tmp_path: Path) -> None:
    """Load YAML config -> diffuse for 10 steps with boundary sources -> snapshot -> reload."""
    cfg_dict = {
        "geometry": {
            "n_grid": 21,
            "length": 1.0e-3,
            "boundary": {
                "mucosa": {"type": "flux", "sources": {"EX_o2_e": 1.0e-3}},
                "lumen": {"type": "dirichlet", "values": {"EX_glc__D_e": 5.0}},
            },
        },
        "metabolites": [
            {"id": "o2_e", "diffusion": 2.1e-9, "init": {"mode": "uniform", "value": 0.0}},
            {"id": "glc__D_e", "diffusion": 6.7e-10, "init": {"mode": "uniform", "value": 0.0}},
        ],
        "simulation": {"t_end": 1.0, "dt": 0.1, "snapshot_every": 0.5},
    }
    yaml_path = tmp_path / "sim.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg_dict))

    cfg = spatial.SpatialConfig.from_yaml(yaml_path)
    cfg.check_cfl()

    geom = spatial.Geometry1D(
        n_grid=cfg.geometry.n_grid,
        length=cfg.geometry.length,
        bc_left=spatial.BoundarySpec(
            type="flux", values=cfg.geometry.boundary["mucosa"].sources
        ),
        bc_right=spatial.BoundarySpec(
            type="dirichlet", values=cfg.geometry.boundary["lumen"].values
        ),
    )

    metabolite_ids = [m.id for m in cfg.metabolites]
    n_grid = geom.n_grid
    C = np.zeros((len(metabolite_ids), n_grid))
    B = np.zeros((1, n_grid))  # placeholder; FBA wired in PR 2
    state = spatial.SpatialState(metabolites=C, biomass=B, t=0.0)

    L = spatial.build_laplacian_1d(
        n_grid, geom.dx, bc_left="neumann", bc_right="dirichlet"
    )
    D = np.array([m.diffusion for m in cfg.metabolites])

    rec = spatial.SnapshotRecorder(output_dir=tmp_path / "snaps", every=cfg.simulation.snapshot_every)

    n_steps = int(round(cfg.simulation.t_end / cfg.simulation.dt))
    for _ in range(n_steps):
        # Reaction sub-step is a no-op in PR 1 (no FBA yet).
        new_C = spatial.diffuse_step(state.metabolites, L, D, cfg.simulation.dt)
        geom.apply_boundary_sources(new_C, metabolite_ids, cfg.simulation.dt)
        state = spatial.SpatialState(metabolites=new_C, biomass=state.biomass, t=state.t + cfg.simulation.dt)
        rec.maybe_save(state)

    # Mucosa cell received O2 flux; lumen cell pinned to 5 mmol/L of glucose.
    assert state.metabolites[0, 0] > 0.0          # O2 accumulated at mucosa
    assert state.metabolites[1, -1] == pytest.approx(5.0)  # glucose Dirichlet held

    # At least one snapshot landed on disk and is readable.
    snaps = sorted((tmp_path / "snaps").glob("snapshot_*.npz"))
    assert len(snaps) >= 1
    restored = spatial.SnapshotRecorder.load(snaps[-1])
    assert restored.t > 0.0
```

- [ ] **Step 3: Run the public API tests**

Run: `pytest tests/spatial/test_public_api.py`
Expected: 2 passed.

- [ ] **Step 4: Run the entire spatial test suite**

Run: `pytest tests/spatial/`
Expected: ~50 passed in well under 30 seconds. Check that no test takes more than a second.

- [ ] **Step 5: Run the entire repo test suite to check for regressions**

Run: `pytest tests/ -q`
Expected: all tests still pass; no v01 regressions.

- [ ] **Step 6: Commit**

```bash
git add src/gemfitcom/spatial/__init__.py tests/spatial/test_public_api.py
git commit -m "feat(spatial): finalize PR 1 public API + integration smoke test"
```

---

## Verification checklist (run before opening PR)

- [ ] `pytest tests/spatial/` — all green, < 30 s total
- [ ] `pytest tests/` — no v01 regressions
- [ ] `ruff format src/gemfitcom/spatial tests/spatial`
- [ ] `ruff check src/gemfitcom/spatial tests/spatial`
- [ ] `pip install -e ".[spatial,dev]"` — clean install on a fresh env
- [ ] All commits use `feat(spatial):` / `test(spatial):` prefix
- [ ] PR description references `docs/superpowers/specs/2026-05-05-spatial-dfba-v0.1-design.md` PR 1 scope

---

## What's NOT included (and why)

| Excluded | Reason |
|---|---|
| Any `cobra` import | PR 1 is FBA-decoupled by design (spec §8 PR 1) |
| `species` / `kinetics` config | Lands in PR 2 with `reaction.py` |
| `backend` config + joblib | Lands in PR 4 |
| `Simulator` class | PR 3 — orchestrates reaction + diffusion + boundary + recorder together |
| `viz.py` | PR 5 |
| Well-mixed limit invariant | Needs full Simulator; lands in PR 3 |
| Physical unit conversion (mol/m²·s ↔ mmol/L·h) | Boundary values stay in their natural units in PR 1; conversion deferred so users pre-compute correct numbers — formalize in PR 3 once Simulator is in place |

---

## Spec ↔ task mapping (self-review)

| Spec section | Implemented in |
|---|---|
| §3.1 SpatialState | Task 2 |
| §3.1 Geometry1D | Tasks 3, 4 |
| §3.2 Lie splitting (diffusion + boundary half) | Tasks 5, 6, 14 (smoke) |
| §3.4 modules state/geometry/diffusion/recorder/config | Tasks 2, 3-4, 5-8, 11, 12-13 |
| §3.5 spatial extras (pydantic only in PR 1) | Task 1 |
| §5.3 YAML config (geometry/metabolites/simulation/output) | Task 12 |
| §6 errors: CFL fail-fast | Tasks 7, 13 |
| §6 errors: shape validation | Task 2 |
| §7.2 unit tests for state/geometry/diffusion/recorder/config | Tasks 2-7, 11-13 |
| §7.3 mass conservation invariant | Task 9 |
| §7.3 positivity invariant | Task 10 |
| §7.4 well-mixed limit invariant | **Deferred to PR 3** (needs Simulator) |
| §8 PR 1 deliverable: pure-diffusion sim runs end-to-end | Task 14 (smoke) |
