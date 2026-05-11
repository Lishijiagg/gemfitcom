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

The CLI form `gemfitcom spatial run sim.yaml` will land in PR 3.
