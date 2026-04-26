# Quickstart

GemFitCom is organized as four CLI subcommands, one per pipeline stage.
Stages communicate through on-disk artifacts (YAML, CSV, SBML), so each
step is auditable and replayable.

The repository ships a synthetic mini dataset under `data/examples/` so
the calibration step below runs end-to-end without any external downloads.
Regenerate it any time with `python scripts/build_example_data.py`.

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
