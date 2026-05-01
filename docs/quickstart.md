# Quickstart

GemFitCom is organized as four CLI subcommands, one per pipeline stage.
Stages communicate through on-disk artifacts (YAML, CSV, SBML), so each
step is auditable and replayable.

The repository ships a synthetic mini dataset under `data/examples/` so
the calibration step below runs end-to-end without any external downloads.
Regenerate it any time with `python scripts/build_example_data.py`.

## Input data format

Experimental data is provided as **CSV/TSV files** in long format. The YAML
config (`configs/example_strain.yaml`) is *not* the data — it points at the
CSVs and sets per-strain parameters. See `data/examples/` for a runnable
mini dataset that demonstrates the structure.

### OD growth curve (`*_od.csv`)

One row per `(time_h, carbon_source, replicate)`:

| column          | type  | meaning                                          |
| --------------- | ----- | ------------------------------------------------ |
| `time_h`        | float | hours since inoculation; t=0 is the initial point |
| `carbon_source` | str   | label for the carbon condition (e.g. `glc__D`)    |
| `replicate`     | int   | 1, 2, 3, …                                        |
| `od`            | float | OD600 reading                                     |

Declare in the strain config whether OD is baseline-subtracted or absolute,
and provide `initial_biomass_gDW_per_L` so the fitter can recover an absolute
biomass scale.

### HPLC metabolite panel (`*_hplc.csv`)

One row per `(time_h, carbon_source, metabolite, replicate)`:

| column          | type  | meaning                                          |
| --------------- | ----- | ------------------------------------------------ |
| `time_h`        | float | hours since inoculation; **optional** — leave the column empty for endpoint-only data |
| `carbon_source` | str   | matches the `carbon_source` column in the OD file |
| `metabolite`    | str   | display name (`acetate`, `butyrate`, `propionate`, `lactate`, …) |
| `value_mM`      | float | concentration in millimolar                       |
| `replicate`     | int   | 1, 2, 3, … (defaults to 1 if omitted)            |

Real HPLC runs sample multiple time points and report multiple metabolites
per injection, so the canonical form is a time series. Endpoint-only tables
(no `time_h`) still load — internally they're treated as a single snapshot.

### File format notes

- TSV is accepted: pass `sep="\t"` to `load_od` / `load_hplc` if auto-detect
  doesn't pick it up.
- Wide HPLC tables (rows = carbon sources, columns = metabolites) can be
  converted with `gemfitcom.io.hplc.hplc_wide_to_long` before loading.
- The GEM is a COBRA-compatible SBML (`*.xml`) — typically AGORA2, CarveMe,
  or a hand-curated model.

## 1. Calibrate kinetics for one strain

```bash
gemfitcom fit configs/example_strain.yaml --output results/
```

This loads the OD curve, optionally runs KB-driven gap-fill, and fits
V<sub>max</sub> and K<sub>m</sub> for the configured carbon source. On
the example dataset (truth: V<sub>max</sub> = 4.0, K<sub>m</sub> = 2.0)
you should see something like:

```
fit: strain=toy_acetogen R²=1.0000 Vmax=3.877 Km=1.831
  params → results/toy_acetogen_fitted_params.yaml
  model  → results/toy_acetogen_model_fitted.xml
  grid   → results/toy_acetogen_fit_grid.csv
```

The `params` file is consumed by step 2; the `model` file is the SBML
with constrained exchange bounds; the `grid` file is the V<sub>max</sub>
× K<sub>m</sub> R² table that `viz.plot_kinetics_heatmap` consumes.

## 2. Simulate a community

```bash
gemfitcom simulate configs/example_community.yaml --output results/
```

Three modes are available via `simulation.mode` in the config:

- `sequential_dfba` — independent per-strain FBA at each time step on a
  shared metabolite pool. Fast, good for trajectory inspection.
- `micom` — MICOM steady-state cooperative tradeoff community FBA. No
  trajectory, just a snapshot.
- `fusion` — dynamic dMICOM (cooperative MICOM optimization at each
  dFBA time step). The slowest and most expressive option.

The output is an `exchange_panel.csv` and `biomass_panel.csv`; the dFBA
modes also save full `biomass.csv` and `pool.csv` trajectories.

## 3. Derive interaction edges and a network

```bash
gemfitcom interactions \
    results/demo_community_exchange_panel.csv \
    --biomass results/demo_community_biomass_panel.csv \
    --output results/network/
```

Cross-feeding edges (donor → recipient) are written as a CSV; competition
edges (pairwise shared uptake) as another CSV; and a combined GraphML
file ready for downstream visualization. See `viz.plot_interaction_network`
to render the GraphML in Python.

## 4. Standalone gap-fill

```bash
gemfitcom gapfill path/to/model.xml \
    --source agora2 --observed EX_ac_e,EX_but_e \
    --output results/
```

Useful when you want to add missing fermentation-product exchanges to a
GEM without running the full calibration step. The KB-driven approach
adds the smallest set of reactions needed to enable each observed product.

## Help and option lists

`gemfitcom COMMAND --help` shows every flag for any subcommand. See
`configs/example_strain.yaml` and `configs/example_community.yaml` for
the full schema.

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
reactions) and is beyond the scope of this guide.
