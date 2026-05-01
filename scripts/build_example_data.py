"""Generate the mini example dataset shipped with gemfitcom.

Builds a toy acetogen GEM (glucose → biomass + acetate), simulates it under
the YCFA medium with known Vmax/Km, then writes:

* ``data/examples/toy_acetogen.xml``         — SBML model
* ``data/examples/toy_acetogen_od.csv``      — long-form OD curve, 3 replicates
* ``data/examples/toy_acetogen_hplc.csv``    — long-form HPLC endpoint, 3 replicates
* ``data/examples/SOURCE.md``                — provenance / how to regenerate

Re-run after changing schemas or true parameters::

    python scripts/build_example_data.py
"""

from __future__ import annotations

from pathlib import Path

import cobra.io
import numpy as np
import pandas as pd
from cobra import Metabolite, Model, Reaction

from gemfitcom.kinetics.mm import MMParams
from gemfitcom.kinetics.mono_dfba import simulate_mono_dfba
from gemfitcom.medium.constraints import apply_medium
from gemfitcom.medium.registry import load_medium

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "examples"

STRAIN_NAME = "toy_acetogen"
TRUE_VMAX = 4.0  # mmol / gDW / h
TRUE_KM = 2.0  # mM
INITIAL_BIOMASS = 0.01  # gDW / L
T_TOTAL_H = 14.0
DT_H = 0.25
N_REPLICATES = 3
OD_NOISE_SIGMA = 0.05  # 5% Gaussian noise on OD
HPLC_NOISE_SIGMA = 0.05  # 5% Gaussian noise on HPLC measurements
BIOMASS_PER_OD = 0.35  # gDW / L per OD600 unit (typical bacterial conversion)
OD_SAMPLING_TIMES_H = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 14], dtype=float)
# HPLC is sampled more sparsely than OD in real experiments (each injection
# costs ~30 min of instrument time); pick t=0 baseline + mid-log + endpoint.
HPLC_SAMPLING_TIMES_H = np.array([0, 6, 14], dtype=float)
# Metabolites we report on the HPLC panel. The toy strain only produces
# acetate; butyrate / propionate / lactate are baseline-zero "not detected"
# columns to mimic a real SCFA HPLC method that screens multiple analytes.
HPLC_METABOLITES: tuple[str, ...] = ("acetate", "butyrate", "propionate", "lactate")
RNG_SEED = 0


# YCFA pool fermentation products that the toy strain does NOT actually make,
# but which need to exist as exchange reactions in the SBML model so dFBA can
# track them in the YCFA medium pool. Each is added as an inert stub.
INERT_PRODUCT_EXCHANGES: tuple[str, ...] = (
    "EX_but_e",
    "EX_ppa_e",
    "EX_for_e",
    "EX_lac__D_e",
    "EX_lac__L_e",
    "EX_succ_e",
)


def build_toy_acetogen() -> Model:
    """Toy cobrapy model: glucose → biomass + 1.5 acetate.

    Reactions:
        - EX_glc__D_e: extracellular glucose exchange (substrate).
        - EX_ac_e: extracellular acetate exchange (product).
        - GLC_t: glucose import.
        - AC_t: acetate export.
        - BIOMASS: glucose-driven biomass + acetate co-production
          (yield 0.1 gDW per mmol glucose; 1.5 mmol acetate per glucose).
        - EX_biomass: biomass sink (objective drain).
        - EX_but_e, EX_ppa_e, EX_for_e, EX_lac__{D,L}_e, EX_succ_e:
          inert stub exchanges so the strain can be simulated under the YCFA
          medium (which tracks these as pool components even when zero).
    """
    m = Model(STRAIN_NAME)

    glc_e = Metabolite("glc__D_e", compartment="e", formula="C6H12O6")
    glc_c = Metabolite("glc__D_c", compartment="c", formula="C6H12O6")
    ac_c = Metabolite("ac_c", compartment="c", formula="C2H3O2", charge=-1)
    ac_e = Metabolite("ac_e", compartment="e", formula="C2H3O2", charge=-1)
    bio = Metabolite("biomass_c", compartment="c")

    ex_glc = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc_e: -1})

    ex_ac = Reaction("EX_ac_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex_ac.add_metabolites({ac_e: -1})

    glc_t = Reaction("GLC_t", lower_bound=0.0, upper_bound=1000.0)
    glc_t.add_metabolites({glc_e: -1, glc_c: 1})

    ac_t = Reaction("AC_t", lower_bound=0.0, upper_bound=1000.0)
    ac_t.add_metabolites({ac_c: -1, ac_e: 1})

    biomass = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    # Cobra convention: the BIOMASS reaction's flux equals μ (1/h). The
    # stoichiometry encodes the per-μ yield: 10 mmol glucose / gDW per μ
    # gives an effective yield of 0.1 gDW per mmol glucose (typical for
    # carbon-limited bacterial growth). Acetate is a fixed-stoichiometry
    # byproduct (15 mmol acetate per gDW biomass = 1.5 per glucose).
    biomass.add_metabolites({glc_c: -10.0, bio: 1.0, ac_c: 15.0})

    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})

    inert_rxns: list[Reaction] = []
    for ex_id in INERT_PRODUCT_EXCHANGES:
        met_id = ex_id.removeprefix("EX_")
        met = Metabolite(met_id, compartment="e")
        rxn = Reaction(ex_id, lower_bound=-1000.0, upper_bound=1000.0)
        rxn.add_metabolites({met: -1})
        inert_rxns.append(rxn)

    m.add_reactions([ex_glc, ex_ac, glc_t, ac_t, biomass, sink, *inert_rxns])
    m.objective = "BIOMASS"
    m.annotation["strain_name"] = STRAIN_NAME
    m.annotation["source"] = "synthetic"
    return m


def simulate_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_h, biomass_gDW_per_L, acetate_conc) at the truth params."""
    model = build_toy_acetogen()
    medium = load_medium("YCFA")
    apply_medium(model, medium, close_others=False, on_missing="ignore")
    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=TRUE_VMAX, km=TRUE_KM)},
        initial_biomass=INITIAL_BIOMASS,
        t_total=T_TOTAL_H,
        dt=DT_H,
    )
    if "EX_ac_e" not in res.pool.columns:
        raise RuntimeError("simulation did not track EX_ac_e — check medium / model setup")
    return res.time_h, res.biomass, res.pool["EX_ac_e"].to_numpy()


def make_od_csv(time_h: np.ndarray, biomass: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """OD long-form: time_h, carbon_source, replicate, od (3 noisy replicates)."""
    od_truth = np.interp(OD_SAMPLING_TIMES_H, time_h, biomass) / BIOMASS_PER_OD
    rows: list[dict[str, object]] = []
    for rep in range(1, N_REPLICATES + 1):
        # Multiplicative noise — keeps OD strictly positive and proportional.
        noise = rng.normal(loc=1.0, scale=OD_NOISE_SIGMA, size=OD_SAMPLING_TIMES_H.shape)
        od_rep = np.clip(od_truth * noise, a_min=1e-4, a_max=None)
        for t, od in zip(OD_SAMPLING_TIMES_H, od_rep, strict=True):
            rows.append(
                {
                    "time_h": float(t),
                    "carbon_source": "glc__D",
                    "replicate": rep,
                    "od": float(od),
                }
            )
    return pd.DataFrame.from_records(rows, columns=["time_h", "carbon_source", "replicate", "od"])


def make_hplc_csv(
    acetate_conc: np.ndarray, time_h: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """HPLC time-series long-form.

    Columns: ``time_h, carbon_source, metabolite, value_mM, replicate``.

    The toy strain only produces acetate, so acetate concentrations come from
    the dFBA truth trajectory; butyrate / propionate / lactate are reported
    at baseline zero (with tiny noise floor) to mimic a typical SCFA HPLC
    panel that screens for multiple analytes whether or not they're present.
    """
    truth_at_sample = np.interp(HPLC_SAMPLING_TIMES_H, time_h, acetate_conc)
    rows: list[dict[str, object]] = []
    for rep in range(1, N_REPLICATES + 1):
        for t_idx, t in enumerate(HPLC_SAMPLING_TIMES_H):
            for met in HPLC_METABOLITES:
                if met == "acetate":
                    base = float(truth_at_sample[t_idx])
                    noisy = base * rng.normal(loc=1.0, scale=HPLC_NOISE_SIGMA)
                    val = max(0.0, float(noisy))
                else:
                    # Not produced by the toy strain — report a small noise
                    # floor so the column is realistic (HPLC integrators
                    # rarely return exact zero) but stays below detection.
                    val = max(0.0, float(rng.normal(loc=0.0, scale=0.02)))
                rows.append(
                    {
                        "time_h": float(t),
                        "carbon_source": "glc__D",
                        "metabolite": met,
                        "value_mM": val,
                        "replicate": rep,
                    }
                )
    return pd.DataFrame.from_records(
        rows, columns=["time_h", "carbon_source", "metabolite", "value_mM", "replicate"]
    )


def write_provenance() -> Path:
    text = f"""# data/examples — provenance

These files are **synthetic** and ship with `gemfitcom` so users can run the
quickstart end-to-end without external downloads.

## How they were generated

`scripts/build_example_data.py` builds a toy GEM `{STRAIN_NAME}` (glucose ->
0.1 biomass + 1.5 acetate per mmol glucose), runs mono-strain dFBA under YCFA
with **truth parameters** `Vmax = {TRUE_VMAX}` mmol/gDW/h and `Km = {TRUE_KM}`
mM, then samples the trajectory with multiplicative Gaussian noise:

* OD: {N_REPLICATES} replicates, sigma = {OD_NOISE_SIGMA}
* OD sampling times (hours): {OD_SAMPLING_TIMES_H.astype(int).tolist()}
* HPLC: {N_REPLICATES} replicates × {len(HPLC_METABOLITES)} metabolites
  ({list(HPLC_METABOLITES)}), sigma = {HPLC_NOISE_SIGMA}
* HPLC sampling times (hours): {HPLC_SAMPLING_TIMES_H.astype(int).tolist()}
* Total horizon: {T_TOTAL_H} h, simulation dt: {DT_H} h
* OD-to-biomass conversion: {BIOMASS_PER_OD} gDW per OD unit
* RNG seed: {RNG_SEED}

The HPLC panel includes butyrate / propionate / lactate at a near-zero
"not detected" noise floor — the toy strain only produces acetate, but a
realistic SCFA HPLC method screens multiple analytes per injection.

## Files

* `{STRAIN_NAME}.xml` — toy SBML (six reactions, three compartments)
* `{STRAIN_NAME}_od.csv` — long-form OD growth curve
* `{STRAIN_NAME}_hplc.csv` — long-form HPLC time series (multiple metabolites)

## Regenerating

    python scripts/build_example_data.py

## Real data

Real lab data (full AGORA2 SBML, multi-strain OD/HPLC) lives outside the
repository — see the project's Zenodo / GitHub Release for the v0.1 paper.
"""
    path = OUT_DIR / "SOURCE.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"building toy GEM at vmax={TRUE_VMAX}, km={TRUE_KM} ...")
    model = build_toy_acetogen()
    sbml_path = OUT_DIR / f"{STRAIN_NAME}.xml"
    cobra.io.write_sbml_model(model, str(sbml_path))
    print(f"  wrote {sbml_path.relative_to(REPO_ROOT)}")

    time_h, biomass, acetate_conc = simulate_truth()
    print(
        f"  truth trajectory: max biomass={biomass.max():.4f} gDW/L, "
        f"final acetate={acetate_conc[-1]:.3f} mM"
    )

    rng = np.random.default_rng(RNG_SEED)
    od_df = make_od_csv(time_h, biomass, rng)
    od_path = OUT_DIR / f"{STRAIN_NAME}_od.csv"
    od_df.to_csv(od_path, index=False)
    print(f"  wrote {od_path.relative_to(REPO_ROOT)}  ({len(od_df)} rows)")

    hplc_df = make_hplc_csv(acetate_conc, time_h, rng)
    hplc_path = OUT_DIR / f"{STRAIN_NAME}_hplc.csv"
    hplc_df.to_csv(hplc_path, index=False)
    print(f"  wrote {hplc_path.relative_to(REPO_ROOT)}  ({len(hplc_df)} rows)")

    src = write_provenance()
    print(f"  wrote {src.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
