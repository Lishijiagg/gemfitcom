# GemFitCom

**GEM + Fit + Community** — a Python pipeline to calibrate genome-scale metabolic models (GEMs) of gut bacterial strains against in vitro growth and metabolite data, and to simulate multi-strain metabolic interactions.

> Status: v0.1 — under active development. APIs are not yet stable.

## What it does

1. **GEM calibration** for a single strain, given an OD growth curve + HPLC metabolite measurements:
   - Auto-detect and add missing reactions (for AGORA2- or CarveMe-sourced models) using a product-to-reaction knowledge base.
   - Fit Michaelis–Menten kinetic parameters (Vmax, Km) for substrate exchange reactions via global optimization (differential evolution) followed by local grid refinement.
   - Constrain exchange flux bounds from fitted kinetics.
2. **Community simulation** for multi-strain consortia with three modes:
   - `sequential_dfba` — independent per-strain FBA at each time step, sharing a common metabolite pool.
   - `micom` — MICOM steady-state cooperative tradeoff community FBA.
   - `fusion` (dMICOM) — dynamic simulation where each time step runs a MICOM cooperative tradeoff optimization instead of per-species FBA.
3. **Interaction analysis** — derive cross-feeding and competition matrices / networks from simulated flux trajectories.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .[dev]
```

CPLEX (if installed) is auto-detected; otherwise GLPK (bundled with cobra) is used.

> **Enabling CPLEX**: installing IBM CPLEX Studio alone is not enough — cobra needs the CPLEX Python bindings. After installing CPLEX Studio, register its Python API via:
> ```bash
> cd "$CPLEX_STUDIO_DIR/python"
> python setup.py install
> ```
> Verify with `gemfitcom solvers` (CPLEX should appear in the "Available" list).

## Quickstart

Four subcommands, one per pipeline stage. Stages communicate through on-disk
artifacts (YAML / CSV / SBML), so each step is auditable and replayable.

The repository ships a synthetic mini dataset under `data/examples/` so step
1 below runs end-to-end without any external downloads. Regenerate it with
`python scripts/build_example_data.py`.

```bash
# 1. Calibrate Vmax/Km for one strain against OD (+ HPLC for gap-fill)
gemfitcom fit configs/example_strain.yaml --output results/

# 2. Run a multi-strain community simulation
gemfitcom simulate configs/example_community.yaml --output results/

# 3. Derive cross-feeding / competition edges and a summary graph
gemfitcom interactions \
    results/demo_community_exchange_panel.csv \
    --biomass results/demo_community_biomass_panel.csv \
    --output results/network/

# 4. Standalone KB-driven gap-fill on an arbitrary SBML
gemfitcom gapfill path/to/model.xml \
    --source agora2 --observed EX_ac_e,EX_but_e \
    --output results/
```

Run `gemfitcom COMMAND --help` for full option lists. See
`configs/example_strain.yaml` and `configs/example_community.yaml` for
schema references.

## Project layout

```
src/gemfitcom/
├── io/           # Data loaders (OD, HPLC, SBML, YAML config)
├── preprocess/   # Growth-rate / lag extraction, HPLC cleanup
├── medium/       # Medium composition registry (YCFA, LB, M9, ...)
├── gapfill/      # Missing-reaction detection and addition
├── kinetics/     # MM kinetics fitting (DE + grid refinement)
├── simulate/     # mono dFBA, sequential dFBA, MICOM, fusion (dMICOM)
├── interactions/ # Cross-feeding / competition network construction
├── viz/          # Plotting utilities
├── utils/        # Shared helpers (solver auto-detect, ...)
└── cli.py        # Typer CLI
```

## License

MIT — see [LICENSE](LICENSE).
