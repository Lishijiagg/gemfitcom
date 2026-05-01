<p align="center">
  <img src="docs/assets/logo-wordmark.svg" alt="GemFitCom" width="540">
</p>

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

Requires Python ≥ 3.10. Pick the path that matches your role.

### Recommended for users — conda/mamba env + pip

This is the most reliable path on Windows and macOS because conda-forge ships
prebuilt wheels for the heavier scientific stack (`numpy`, `scipy`, `cobra`,
`micom`'s native deps), avoiding compiler issues:

```bash
mamba create -n gemfitcom python=3.11
mamba activate gemfitcom
pip install git+https://github.com/Lishijiagg/gemfitcom.git
```

(Substitute `conda` for `mamba` if you don't have mamba/micromamba; mamba is
just much faster at solving the environment.)

### Quick install — pip only

If you already have a working Python environment:

```bash
pip install git+https://github.com/Lishijiagg/gemfitcom.git
```

A PyPI release (`pip install gemfitcom`) will follow once the API stabilizes.

### For contributors — editable install with dev extras

```bash
git clone https://github.com/Lishijiagg/gemfitcom
cd gemfitcom
pip install -e ".[dev]"
```

The `dev` extra installs `pytest`, `ruff`, and `pre-commit`. Use `".[docs]"`
to additionally install `mkdocs-material` and `mkdocstrings` for docs work.

### LP solver — GLPK by default, CPLEX optional

GLPK ships bundled with cobra and runs out of the box. CPLEX is auto-detected
if installed and is noticeably faster on community simulations.

> **Enabling CPLEX**: installing IBM CPLEX Studio alone is not enough — cobra
> needs the CPLEX Python bindings. After installing CPLEX Studio, register
> its Python API via:
> ```bash
> cd "$CPLEX_STUDIO_DIR/python"
> python setup.py install
> ```
> Verify with `gemfitcom solvers` (CPLEX should appear in the "Available" list).

## Input data format

GemFitCom reads experimental data from **CSV/TSV files** (long format) and
threads them together via a **YAML config**. The YAML is *not* the data — it
points at the data files and sets per-strain parameters.

See `data/examples/` for a runnable mini dataset that demonstrates the exact
structure. The files involved per strain are:

- **OD growth curve** (`*_od.csv`) — long format, one row per
  `(time_h, carbon_source, replicate)`:

  | column          | type  | meaning                                          |
  | --------------- | ----- | ------------------------------------------------ |
  | `time_h`        | float | hours since inoculation; t=0 is the initial point |
  | `carbon_source` | str   | label for the carbon condition (e.g. `glc__D`)    |
  | `replicate`     | int   | 1, 2, 3, …                                        |
  | `od`            | float | OD600 reading (baseline-subtracted or absolute — declare which in the config) |

- **HPLC metabolite panel** (`*_hplc.csv`) — long format, one row per
  `(time_h, carbon_source, metabolite, replicate)`:

  | column          | type  | meaning                                          |
  | --------------- | ----- | ------------------------------------------------ |
  | `time_h`        | float | hours since inoculation; **optional** — leave empty for endpoint-only data |
  | `carbon_source` | str   | matches the `carbon_source` column in the OD file |
  | `metabolite`    | str   | display name (`acetate`, `butyrate`, `propionate`, `lactate`, …) |
  | `value_mM`      | float | concentration in millimolar                       |
  | `replicate`     | int   | 1, 2, 3, … (defaults to 1 if omitted)            |

- **Strain config** (`configs/your_strain.yaml`) — paths to the two CSVs
  above plus carbon-source ID, medium name, biomass conversion, kinetics
  bounds, etc. See `configs/example_strain.yaml` for the full schema.

- **GEM** (`*.xml`) — a COBRA-compatible SBML, e.g. an AGORA2 / CarveMe / hand-curated model.

TSV files are accepted: pass `sep="\t"` to `load_od` / `load_hplc` if the
auto-detect doesn't pick it up. Wide HPLC tables (rows = carbon sources,
columns = metabolites) can be converted via `gemfitcom.io.hplc.hplc_wide_to_long`.

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

## Realistic data example

The synthetic `toy_acetogen` dataset above is intentionally tiny so the
quickstart finishes in seconds and is easy to debug. To see what *real*
lab data looks like in the same long-format schema, the repository also
ships a slice of the
[GEMs-butyrate](https://github.com/Lishijiagg/GEMs-butyrate) project:
*Bifidobacterium longum* subsp. *infantis* ATCC 15697 grown on the GMC
control carbon source (3 OD replicates × ~285 timepoints, plus a
6-metabolite HPLC endpoint).

Fetch the data + the upstream SBML model (~2.6 MB, not committed):

```bash
python scripts/fetch_realistic_example.py
```

This drops three files under `data/examples/realistic/`:

- `B_infantis_GMC_od.csv`   — long-form OD growth curve
- `B_infantis_GMC_hplc.csv` — long-form HPLC endpoint, six metabolites
- `B_infantis_GEM.xml`      — SBML model from the upstream repo

Use these CSVs as a **template for your own data**: copy the column
layout, replace the values, and point a strain config at them. The
realistic example is shipped as a *format reference*; running a full
end-to-end fit on it requires picking an exchange ID for GMC (the
upstream paper handles HMOs / mixed substrates with custom transport
reactions) and is beyond the scope of this README.

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
