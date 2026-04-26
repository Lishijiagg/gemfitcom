# data/examples — provenance

These files are **synthetic** and ship with `gemfitcom` so users can run the
quickstart end-to-end without external downloads.

## How they were generated

`scripts/build_example_data.py` builds a toy GEM `toy_acetogen` (glucose ->
0.1 biomass + 1.5 acetate per mmol glucose), runs mono-strain dFBA under YCFA
with **truth parameters** `Vmax = 4.0` mmol/gDW/h and `Km = 2.0`
mM, then samples the trajectory with multiplicative Gaussian noise:

* OD: 3 replicates, sigma = 0.05
* HPLC: 3 replicates of the endpoint acetate, sigma = 0.05
* Sampling times (hours): [0, 1, 2, 3, 4, 6, 8, 10, 12, 14]
* Total horizon: 14.0 h, simulation dt: 0.25 h
* OD-to-biomass conversion: 0.35 gDW per OD unit
* RNG seed: 0

## Files

* `toy_acetogen.xml` — toy SBML (six reactions, three compartments)
* `toy_acetogen_od.csv` — long-form OD growth curve
* `toy_acetogen_hplc.csv` — long-form HPLC endpoint table

## Regenerating

    python scripts/build_example_data.py

## Real data

Real lab data (full AGORA2 SBML, multi-strain OD/HPLC) lives outside the
repository — see the project's Zenodo / GitHub Release for the v0.1 paper.
