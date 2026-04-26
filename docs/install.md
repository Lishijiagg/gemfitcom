# Installation

## Requirements

- Python ≥ 3.10
- An LP solver. GLPK (bundled with cobra) works out of the box; CPLEX is
  optional and noticeably faster on large community models.

## From source

```bash
git clone https://github.com/Lishijiagg/gemfitcom
cd gemfitcom
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`, `ruff`, and `pre-commit`. For documentation
work add `".[docs]"` to install `mkdocs-material` and `mkdocstrings`.

## Verifying

```bash
gemfitcom --help
gemfitcom solvers
```

`gemfitcom solvers` lists every LP backend cobra detected on your system and
the one GemFitCom would pick (CPLEX > Gurobi > GLPK). If that line is empty,
the install is incomplete — typically because the cobra bindings for your
solver are missing.

## Enabling CPLEX

Installing IBM CPLEX Studio alone is **not** enough. cobra needs the CPLEX
Python bindings, which ship with CPLEX Studio but are installed separately:

```bash
cd "$CPLEX_STUDIO_DIR/python"
python setup.py install
```

Verify with `gemfitcom solvers` — CPLEX should appear in the "Available" list.

## Running the test suite

```bash
pytest
```

The end-to-end smoke test under `tests/test_cli.py` runs the full CLI on
the synthetic dataset shipped under `data/examples/`. It takes ~20 seconds
because the differential-evolution stage of `fit_kinetics` actually runs.
