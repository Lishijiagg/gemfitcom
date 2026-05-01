# data/examples/realistic — provenance

This is a **realistic single-strain example** sliced from the upstream
[GEMs-butyrate](https://github.com/Lishijiagg/GEMs-butyrate) project (the same
authors as gemfitcom). Use it as a reference for what real lab data looks
like in GemFitCom's expected long format. The synthetic toy under
`data/examples/` is what the quickstart actually runs end-to-end; this
realistic example is more about *data shape* than runtime convenience.

## Strain and condition

* **Strain**: *Bifidobacterium longum* subsp. *infantis* ATCC 15697
* **Carbon source**: GMC (glucose-mannose-cellobiose mix; growth-positive control in the source paper)
* **Replicates**: 3 of the original 7 (subset chosen for compactness)
* **HPLC time**: single endpoint (upstream measured at study endpoint only;
  hence `time_h` is left empty in `*_hplc.csv`)
* **HPLC metabolites**: formate, acetate, propionate, butyrate, lactate, succinate

## Files

* `B_infantis_GMC_od.csv`   — long-form OD growth curve (~60 timepoints × 3 reps)
* `B_infantis_GMC_hplc.csv` — long-form HPLC endpoint, 6 metabolites
* `B_infantis_GEM.xml`      — SBML model (downloaded by `fetch_realistic_example.py`,
  *not* checked into git because it's ~2.6 MB)

## Regenerate / refresh

    python scripts/fetch_realistic_example.py

The script pulls the latest from the upstream repo:

* OD:   `https://raw.githubusercontent.com/Lishijiagg/GEMs-butyrate/main/Mono_culture_ODs/B.infantis_raw_OD.csv`
* HPLC: `https://raw.githubusercontent.com/Lishijiagg/GEMs-butyrate/main/HPLC_results/Bin_hplc.csv`
* SBML: `https://raw.githubusercontent.com/Lishijiagg/GEMs-butyrate/main/GEMs/Bifidobacterium_longum_infantis_ATCC_15697.xml`

## Caveat

The upstream HPLC reports a single endpoint snapshot — there is no time
series here, unlike the synthetic `toy_acetogen_hplc.csv`. This still
loads correctly: GemFitCom treats endpoint-only HPLC tables as a
snapshot for gap-fill purposes.
