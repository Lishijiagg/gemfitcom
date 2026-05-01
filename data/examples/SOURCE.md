# data/examples — provenance

These files are **synthetic** and ship with `gemfitcom` so users can run the
quickstart end-to-end without external downloads.

## How they were generated

`scripts/build_example_data.py` builds a toy GEM `toy_acetogen` (glucose ->
0.1 biomass + 1.5 acetate per mmol glucose), runs mono-strain dFBA under YCFA
with **truth parameters** `Vmax = 4.0` mmol/gDW/h and `Km = 2.0`
mM, then samples the trajectory with multiplicative Gaussian noise:

* OD: 3 replicates, sigma = 0.05
* OD sampling times (hours): [0, 1, 2, 3, 4, 6, 8, 10, 12, 14]
* HPLC: 3 replicates × 4 metabolites
  (['acetate', 'butyrate', 'propionate', 'lactate']), sigma = 0.05
* HPLC sampling times (hours): [0, 6, 14]
* Total horizon: 14.0 h, simulation dt: 0.25 h
* OD-to-biomass conversion: 0.35 gDW per OD unit
* RNG seed: 0

The HPLC panel includes butyrate / propionate / lactate at a near-zero
"not detected" noise floor — the toy strain only produces acetate, but a
realistic SCFA HPLC method screens multiple analytes per injection.

## Files

* `toy_acetogen.xml` — toy SBML (six reactions, three compartments)
* `toy_acetogen_od.csv` — long-form OD growth curve
* `toy_acetogen_hplc.csv` — long-form HPLC time series (multiple metabolites)

## Regenerating

    python scripts/build_example_data.py

## Real data

Real lab data (full AGORA2 SBML, multi-strain OD/HPLC) lives outside the
repository — see the project's Zenodo / GitHub Release for the v0.1 paper.
