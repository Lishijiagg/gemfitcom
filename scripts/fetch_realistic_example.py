"""Fetch and convert the realistic *Bifidobacterium infantis* example dataset.

Downloads raw OD and HPLC data from the upstream `GEMs-butyrate
<https://github.com/Lishijiagg/GEMs-butyrate>`_ repository, slices out a
single-strain / single-carbon-source subset (B. infantis on GMC), and
converts it to the long-format CSVs that GemFitCom consumes.

Outputs (under ``data/examples/realistic/``):

* ``B_infantis_GMC_od.csv``         — OD long-form, 3 replicates × all timepoints
* ``B_infantis_GMC_hplc.csv``       — HPLC long-form, endpoint snapshot, 6 metabolites
* ``B_infantis_GEM.xml``            — SBML model (~2.6 MB; downloaded, not committed)
* ``SOURCE.md``                     — provenance and how to regenerate

Re-run any time the upstream changes::

    python scripts/fetch_realistic_example.py

The CSVs are tiny and live in git so users can browse the format without
running this script first; the SBML is downloaded fresh because it's too
large to commit.
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "examples" / "realistic"

UPSTREAM_BASE = "https://raw.githubusercontent.com/Lishijiagg/GEMs-butyrate/main"
OD_URL = f"{UPSTREAM_BASE}/Mono_culture_ODs/B.infantis_raw_OD.csv"
HPLC_URL = f"{UPSTREAM_BASE}/HPLC_results/Bin_hplc.csv"
SBML_URL = f"{UPSTREAM_BASE}/GEMs/Bifidobacterium_longum_infantis_ATCC_15697.xml"

CARBON_SOURCE = "GMC"  # glucose-mannose-cellobiose mix; simple growth-positive control
N_REPLICATES_KEEP = 3  # upstream has 7 — drop to 3 for a more compact example


def _fetch(url: str) -> bytes:
    with urlopen(url) as resp:
        return resp.read()


def fetch_od_long() -> pd.DataFrame:
    """Pull the wide OD table, slice GMC-1..GMC-N, return long format."""
    raw = _fetch(OD_URL).decode("utf-8")
    wide = pd.read_csv(io.StringIO(raw))
    time_col = wide.columns[0]  # "Time (h)"
    rep_cols = [f"{CARBON_SOURCE}-{i}" for i in range(1, N_REPLICATES_KEEP + 1)]
    missing = [c for c in rep_cols if c not in wide.columns]
    if missing:
        raise RuntimeError(f"upstream OD missing expected columns: {missing}")

    rows: list[dict[str, object]] = []
    for _, row in wide.iterrows():
        for rep_idx, col in enumerate(rep_cols, start=1):
            rows.append(
                {
                    "time_h": float(row[time_col]),
                    "carbon_source": CARBON_SOURCE,
                    "replicate": rep_idx,
                    "od": float(row[col]),
                }
            )
    return pd.DataFrame.from_records(rows, columns=["time_h", "carbon_source", "replicate", "od"])


def fetch_hplc_long() -> pd.DataFrame:
    """Pull the wide HPLC endpoint table, slice the GMC column, return long format.

    Upstream layout: rows = metabolites (Formate, Acetate, ...), columns =
    carbon sources. We keep only the GMC column and emit one row per
    metabolite. ``time_h`` is left empty because the upstream HPLC is a
    single-endpoint measurement.
    """
    raw = _fetch(HPLC_URL).decode("utf-8")
    wide = pd.read_csv(io.StringIO(raw), index_col=0)
    if CARBON_SOURCE not in wide.columns:
        raise RuntimeError(f"upstream HPLC missing column {CARBON_SOURCE!r}")

    rows: list[dict[str, object]] = []
    for metabolite, value in wide[CARBON_SOURCE].items():
        # Lower-case to match GemFitCom's KB display-name convention.
        rows.append(
            {
                "time_h": pd.NA,
                "carbon_source": CARBON_SOURCE,
                "metabolite": str(metabolite).strip().lower(),
                # Clip negative readings (HPLC baseline noise) to 0.
                "value_mM": max(0.0, float(value)),
                "replicate": 1,
            }
        )
    return pd.DataFrame.from_records(
        rows, columns=["time_h", "carbon_source", "metabolite", "value_mM", "replicate"]
    )


def fetch_sbml(target: Path) -> None:
    """Download the B. infantis SBML if not already present."""
    if target.is_file():
        print(f"  SBML already present at {target.relative_to(REPO_ROOT)} — skipping download")
        return
    print(f"  downloading SBML ({SBML_URL}) ...")
    target.write_bytes(_fetch(SBML_URL))
    size_mb = target.stat().st_size / 1e6
    print(f"  wrote {target.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")


def write_provenance() -> Path:
    text = f"""# data/examples/realistic — provenance

This is a **realistic single-strain example** sliced from the upstream
[GEMs-butyrate](https://github.com/Lishijiagg/GEMs-butyrate) project (the same
authors as gemfitcom). Use it as a reference for what real lab data looks
like in GemFitCom's expected long format. The synthetic toy under
`data/examples/` is what the quickstart actually runs end-to-end; this
realistic example is more about *data shape* than runtime convenience.

## Strain and condition

* **Strain**: *Bifidobacterium longum* subsp. *infantis* ATCC 15697
* **Carbon source**: {CARBON_SOURCE} (glucose-mannose-cellobiose mix; growth-positive control in the source paper)
* **Replicates**: {N_REPLICATES_KEEP} of the original 7 (subset chosen for compactness)
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

* OD:   `{OD_URL}`
* HPLC: `{HPLC_URL}`
* SBML: `{SBML_URL}`

## Caveat

The upstream HPLC reports a single endpoint snapshot — there is no time
series here, unlike the synthetic `toy_acetogen_hplc.csv`. This still
loads correctly: GemFitCom treats endpoint-only HPLC tables as a
snapshot for gap-fill purposes.
"""
    path = OUT_DIR / "SOURCE.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"fetching realistic B. infantis example into {OUT_DIR.relative_to(REPO_ROOT)} ...")

    od_df = fetch_od_long()
    od_path = OUT_DIR / "B_infantis_GMC_od.csv"
    od_df.to_csv(od_path, index=False)
    print(f"  wrote {od_path.relative_to(REPO_ROOT)} ({len(od_df)} rows)")

    hplc_df = fetch_hplc_long()
    hplc_path = OUT_DIR / "B_infantis_GMC_hplc.csv"
    hplc_df.to_csv(hplc_path, index=False)
    print(f"  wrote {hplc_path.relative_to(REPO_ROOT)} ({len(hplc_df)} rows)")

    sbml_path = OUT_DIR / "B_infantis_GEM.xml"
    fetch_sbml(sbml_path)

    src = write_provenance()
    print(f"  wrote {src.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
