# GemFitCom

**GEM + Fit + Community** — a Python pipeline to calibrate genome-scale
metabolic models (GEMs) of gut bacterial strains against in vitro growth
and metabolite data, and to simulate multi-strain metabolic interactions.

!!! warning "Status: v0.1"
    Under active development. APIs are not yet stable; pin to a specific
    commit for reproducibility.

## What it does

1. **GEM calibration** for a single strain, given an OD growth curve and
   HPLC metabolite measurements:
    - Auto-detect and add missing reactions (for AGORA2- or CarveMe-sourced
      models) using a product-to-reaction knowledge base.
    - Fit Michaelis–Menten kinetic parameters (V<sub>max</sub>, K<sub>m</sub>)
      for substrate exchange reactions via global optimization (differential
      evolution) followed by local grid refinement.
    - Constrain exchange flux bounds from fitted kinetics.
2. **Community simulation** for multi-strain consortia with three modes:
    - `sequential_dfba` — independent per-strain FBA at each time step,
      sharing a common metabolite pool.
    - `micom` — MICOM steady-state cooperative tradeoff community FBA.
    - `fusion` (dMICOM) — dynamic simulation where each time step runs a
      MICOM cooperative tradeoff optimization instead of per-species FBA.
3. **Interaction analysis** — derive cross-feeding and competition matrices
   / networks from simulated flux trajectories.

## Where to start

- New here? Read [Installation](install.md) then [Quickstart](quickstart.md).
- Want the science? See [Methodology](methods.md).
- Looking for a specific function? Use the [API reference](api/io.md).

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

MIT — see [LICENSE](https://github.com/Lishijiagg/gemfitcom/blob/main/LICENSE).
