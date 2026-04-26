"""GemFitCom command-line interface (Typer).

Four pipeline stages each get one subcommand. They communicate via
on-disk artifacts (YAML, CSV, SBML) rather than Python objects so
intermediate state is auditable and replayable.

    gemfitcom fit CONFIG.yaml            # calibrate Vmax/Km for one strain
    gemfitcom simulate COMMUNITY.yaml    # run community dFBA / MICOM / fusion
    gemfitcom interactions PANEL.csv     # cross-feeding / competition / graph
    gemfitcom gapfill MODEL.xml ...      # standalone KB-driven gap-fill

Heavy imports (cobra, scipy, networkx) are deferred into the command
bodies so ``gemfitcom --help`` stays fast.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from gemfitcom import __version__

if TYPE_CHECKING:
    import numpy as np

    from gemfitcom.gapfill.report import GapfillReport
    from gemfitcom.io.config import CommunityConfig, Config

# Force UTF-8 stdout/stderr so Unicode in CLI output (R², ×, μ, …)
# does not crash on Windows GBK consoles.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    name="gemfitcom",
    help="GEM + Fit + Community: GEM calibration and multi-strain interaction pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("version")
def print_version() -> None:
    """Print the installed GemFitCom version."""
    typer.echo(__version__)


@app.command("solvers")
def list_solvers() -> None:
    """List LP solvers available to cobra and show the one that would be used."""
    from gemfitcom.utils.solver import available_solvers, get_best_solver

    avail = available_solvers()
    typer.echo(f"Available: {', '.join(avail) if avail else '(none)'}")
    if avail:
        typer.echo(f"Selected:  {get_best_solver()}")


# =========================================================================
# fit
# =========================================================================


@app.command("fit")
def fit_command(
    config: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to strain config YAML."
    ),
    output: Path = typer.Option(
        Path("results"), "--output", "-o", help="Output directory (created if missing)."
    ),
    gapfill: bool = typer.Option(
        True, "--gapfill/--no-gapfill", help="Run KB-driven gap-fill for non-curated models."
    ),
    kb: str = typer.Option("scfa", "--kb", help="Gap-fill KB name or path (default: scfa)."),
) -> None:
    """Calibrate Vmax/Km for one strain against OD and (optionally) HPLC."""
    from gemfitcom.io.config import load_config

    cfg = load_config(config, validate_paths=True)
    output.mkdir(parents=True, exist_ok=True)

    t_obs, biomass_obs = _load_biomass_curve(cfg)
    model = _load_model_for_fit(cfg)
    medium = _load_medium_for_fit(cfg)

    gapfill_report: GapfillReport | None = None
    if gapfill:
        gapfill_report = _maybe_gapfill(cfg, model, kb=kb)
        if gapfill_report is not None:
            _write_gapfill_report(output / f"{cfg.strain.name}_gapfill_report.json", gapfill_report)

    fit_result = _run_fit(cfg, model, medium, t_obs, biomass_obs)

    from gemfitcom.io.config import save_fitted_params

    params_path = save_fitted_params(
        output / f"{cfg.strain.name}_fitted_params.yaml",
        strain=cfg.strain.name,
        r_squared=fit_result.r_squared,
        mm_params={cfg.medium.carbon_source.exchange_id: fit_result.params},
    )
    model_path = _save_model(model, output / f"{cfg.strain.name}_model_fitted.xml")
    grid_path = _save_fit_grid(fit_result, output / f"{cfg.strain.name}_fit_grid.csv")

    typer.echo(
        f"fit: strain={cfg.strain.name} "
        f"R²={fit_result.r_squared:.4f} "
        f"Vmax={fit_result.params.vmax:.3f} Km={fit_result.params.km:.3f}"
    )
    typer.echo(f"  params → {params_path}")
    typer.echo(f"  model  → {model_path}")
    typer.echo(f"  grid   → {grid_path}")
    if gapfill_report is not None and not gapfill_report.skipped:
        typer.echo(
            f"  gapfill: added={len(gapfill_report.products_added)} "
            f"already_present={len(gapfill_report.products_already_present)} "
            f"no_kb_entry={len(gapfill_report.products_missing_kb)}"
        )


def _load_biomass_curve(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """OD → single averaged biomass curve for the configured carbon source.

    Pipeline: per-replicate baseline removal (``subtract_t0``) → floor →
    average across replicates → multiply by ``biomass_conversion``. Because
    ``subtract_t0`` aligns the curve to start at zero, the result is biomass
    *relative to t=0*. To get the absolute biomass that ``fit_kinetics``
    needs, the inoculum biomass (``experiment.initial_biomass_gDW_per_L``)
    is added back as a constant offset. When that field is omitted we fall
    back to ``OD[t=0] * biomass_conversion`` from the raw (un-baselined)
    curve — fine for synthetic data but unreliable on real lab data where
    OD[t=0] also includes media absorbance.
    """
    from gemfitcom.io.od import load_od
    from gemfitcom.preprocess.od import (
        average_replicates,
        floor_od,
        subtract_t0,
    )

    od = load_od(cfg.experiment.od_file)
    # The config picks ONE carbon source for the fit; filter down.
    carbon_label = _carbon_label_from_exchange(cfg.medium.carbon_source.exchange_id)
    sub = od[od["carbon_source"].astype(str) == carbon_label]
    if sub.empty:
        # Fall back to the whole frame if the label doesn't match — single-curve
        # OD files often use a free-form label that doesn't map 1:1 to EX_ IDs.
        sub = od

    raw_avg = average_replicates(sub).sort_values("time_h")
    raw_t0_od = float(raw_avg.iloc[0]["mean_od"])

    sub = subtract_t0(sub)
    sub = floor_od(sub)
    avg = average_replicates(sub).sort_values("time_h")
    t = avg["time_h"].to_numpy()
    biomass_increment = avg["mean_od"].to_numpy() * cfg.experiment.biomass_conversion

    if cfg.experiment.initial_biomass_gDW_per_L is not None:
        x0 = float(cfg.experiment.initial_biomass_gDW_per_L)
    else:
        x0 = raw_t0_od * cfg.experiment.biomass_conversion
    return t, biomass_increment + x0


def _carbon_label_from_exchange(exchange_id: str) -> str:
    """Heuristic EX_glc__D_e → 'glc__D' used by some OD CSVs as the label."""
    return exchange_id.removeprefix("EX_").removesuffix("_e")


def _load_model_for_fit(cfg: Config):
    from gemfitcom.io.models import load_model

    return load_model(
        cfg.strain.model_path, strain_name=cfg.strain.name, source=cfg.strain.model_source
    )


def _load_medium_for_fit(cfg: Config):
    from gemfitcom.medium.registry import load_medium

    return load_medium(cfg.medium.name)


def _maybe_gapfill(cfg: Config, model, *, kb: str) -> GapfillReport | None:
    """Run gap-fill for agora2/carveme sources; return the report or None."""
    from gemfitcom.gapfill.knowledge import load_kb
    from gemfitcom.gapfill.run import run_gapfill

    if cfg.strain.model_source == "curated":
        return run_gapfill(model, observed_products=(), source=cfg.strain.model_source)

    kb_obj = load_kb(kb)
    observed_exchange_ids = _discover_products_from_hplc(cfg, kb_obj)
    if not observed_exchange_ids:
        typer.echo(
            "fit: gapfill skipped — no HPLC-observed products match the KB display-name map."
        )
        return None
    return run_gapfill(
        model,
        observed_products=observed_exchange_ids,
        kb=kb_obj,
        source=cfg.strain.model_source,
    )


def _discover_products_from_hplc(cfg: Config, kb) -> list[str]:
    """Map HPLC metabolite names to exchange IDs via KB display_name."""

    from gemfitcom.io.hplc import load_hplc

    hplc = load_hplc(cfg.experiment.hplc_file)
    display_to_ex: dict[str, str] = {
        entry.display_name.lower(): ex_id for ex_id, entry in kb.entries.items()
    }
    # Any metabolite with a strictly positive average is a candidate product.
    if hplc.empty:
        return []
    positive = hplc.groupby("metabolite")["value_mM"].mean().reset_index(name="mean_mM")
    observed: list[str] = []
    for m in positive.loc[positive["mean_mM"] > 0, "metabolite"].astype(str):
        ex = display_to_ex.get(m.strip().lower())
        if ex is not None:
            observed.append(ex)
    return sorted(set(observed))


def _run_fit(cfg: Config, model, medium, t_obs, biomass_obs):
    from gemfitcom.kinetics.fit import fit_kinetics

    return fit_kinetics(
        model,
        medium,
        carbon_exchange=cfg.medium.carbon_source.exchange_id,
        t_obs=t_obs,
        biomass_obs=biomass_obs,
        vmax_bounds=cfg.kinetics_fit.vmax_bounds_mmol_per_gDW_per_h,
        km_bounds=cfg.kinetics_fit.km_bounds_mM,
        de_maxiter=cfg.kinetics_fit.de_maxiter,
        de_popsize=cfg.kinetics_fit.de_popsize,
        grid_points=cfg.kinetics_fit.grid_points,
        grid_span=cfg.kinetics_fit.grid_span,
        dt=cfg.simulation.dt,
    )


def _save_model(model, path: Path) -> Path:
    import cobra.io

    path.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.write_sbml_model(model, str(path))
    return path


def _save_fit_grid(fit_result, path: Path) -> Path:
    import numpy as np
    import pandas as pd

    vmax_ax = np.asarray(fit_result.grid_vmax_axis, dtype=float)
    km_ax = np.asarray(fit_result.grid_km_axis, dtype=float)
    r2 = np.asarray(fit_result.grid_r_squared, dtype=float)
    # r2 shape is (len(km), len(vmax)) per FitResult docstring.
    km_mesh, vmax_mesh = np.meshgrid(km_ax, vmax_ax, indexing="ij")
    df = pd.DataFrame(
        {
            "vmax": vmax_mesh.ravel(),
            "km": km_mesh.ravel(),
            "r_squared": r2.ravel(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_gapfill_report(path: Path, report: GapfillReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strain_id": report.strain_id,
        "source": report.source,
        "kb_name": report.kb_name,
        "skipped": report.skipped,
        "outcomes": [
            {
                "exchange_id": o.exchange_id,
                "status": o.status,
                "added_metabolites": list(o.added_metabolites),
                "added_reactions": list(o.added_reactions),
                "skipped_metabolites": list(o.skipped_metabolites),
                "skipped_reactions": list(o.skipped_reactions),
                "message": o.message,
            }
            for o in report.outcomes
        ],
        "warnings": list(report.warnings),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# =========================================================================
# simulate
# =========================================================================


@app.command("simulate")
def simulate_command(
    config: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to community config YAML."
    ),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory."),
) -> None:
    """Run a community simulation (sequential_dfba / micom / fusion)."""
    from gemfitcom.interactions import biomass_panel, exchange_panel
    from gemfitcom.io.config import load_community_config
    from gemfitcom.medium.registry import load_medium

    cfg = load_community_config(config, validate_paths=True)
    output.mkdir(parents=True, exist_ok=True)
    medium = load_medium(cfg.medium)

    strains_data = [_load_strain_data(s) for s in cfg.strains]

    mode = cfg.simulation.mode
    if mode == "sequential_dfba":
        result = _run_sequential(strains_data, medium, cfg)
    elif mode == "micom":
        result = _run_micom(strains_data, medium, cfg)
    elif mode == "fusion":
        result = _run_fusion(strains_data, medium, cfg)
    else:  # pragma: no cover — schema validator already rejects.
        raise typer.BadParameter(f"unsupported mode: {mode!r}")

    # Write shared artifacts. Dynamic modes produce biomass + pool; MICOM
    # is a snapshot (no trajectory).
    base = cfg.name
    panel = exchange_panel(result)
    bpanel = biomass_panel(result)

    paths: dict[str, Path] = {}
    if hasattr(result, "biomass") and hasattr(result, "pool"):
        paths["biomass"] = _write_csv(output / f"{base}_biomass.csv", result.biomass)
        paths["pool"] = _write_csv(output / f"{base}_pool.csv", result.pool)
    paths["exchange_panel"] = _write_csv(output / f"{base}_exchange_panel.csv", panel)
    paths["biomass_panel"] = _write_csv(output / f"{base}_biomass_panel.csv", bpanel)

    typer.echo(f"simulate: {mode} community={base}")
    for k, p in paths.items():
        typer.echo(f"  {k:18s} → {p}")


def _load_strain_data(strain):
    """Resolve (model, mm_params) from a CommunityStrainConfig."""
    from gemfitcom.io.config import load_fitted_params
    from gemfitcom.io.models import load_model

    model = load_model(strain.model_path, strain_name=strain.name)
    mm_params = dict(strain.mm_params)
    if strain.fitted_params_path is not None:
        fp = load_fitted_params(strain.fitted_params_path)
        # Inline mm_params override fitted file entries (caller intent wins).
        merged = dict(fp.mm_params)
        merged.update(mm_params)
        mm_params = merged
    return {
        "name": strain.name,
        "model": model,
        "mm_params": mm_params,
        "initial_biomass": strain.initial_biomass,
    }


def _run_sequential(strains_data, medium, cfg: CommunityConfig):
    from gemfitcom.medium.constraints import apply_medium
    from gemfitcom.simulate import StrainSpec, simulate_sequential_dfba

    specs: list[StrainSpec] = []
    for s in strains_data:
        apply_medium(s["model"], medium, close_others=False, on_missing="ignore")
        specs.append(
            StrainSpec(
                name=s["name"],
                model=s["model"],
                mm_params=s["mm_params"],
                initial_biomass=s["initial_biomass"],
            )
        )
    return simulate_sequential_dfba(
        specs,
        medium,
        t_total=cfg.simulation.total_time_h,
        dt=cfg.simulation.dt,
        save_fluxes=cfg.simulation.save_fluxes,
    )


def _run_micom(strains_data, medium, cfg: CommunityConfig):
    from gemfitcom.simulate import CommunityMember, simulate_micom

    members = [
        CommunityMember(name=s["name"], model=s["model"], abundance=s["initial_biomass"])
        for s in strains_data
    ]
    combined_mm: dict = {}
    for s in strains_data:
        combined_mm.update(s["mm_params"])
    return simulate_micom(
        members,
        medium,
        fraction=cfg.simulation.micom_fraction,
        mm_params=combined_mm or None,
    )


def _run_fusion(strains_data, medium, cfg: CommunityConfig):
    from gemfitcom.simulate import CommunityMember, simulate_fusion_dmicom

    members = [
        CommunityMember(name=s["name"], model=s["model"], abundance=s["initial_biomass"])
        for s in strains_data
    ]
    mm_by_member = {s["name"]: s["mm_params"] for s in strains_data if s["mm_params"]}
    return simulate_fusion_dmicom(
        members,
        medium,
        t_total=cfg.simulation.total_time_h,
        dt=cfg.simulation.dt,
        fraction=cfg.simulation.micom_fraction,
        mm_params_by_member=mm_by_member or None,
        save_fluxes=cfg.simulation.save_fluxes,
    )


def _write_csv(path: Path, df) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# =========================================================================
# interactions
# =========================================================================


@app.command("interactions")
def interactions_command(
    panel: Path = typer.Argument(
        ..., exists=True, readable=True, help="Exchange-flux panel CSV from `simulate`."
    ),
    biomass: Path | None = typer.Option(
        None,
        "--biomass",
        exists=True,
        readable=True,
        help="Optional biomass-panel CSV; weights fluxes for absolute-amount edges.",
    ),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory."),
    include_competition: bool = typer.Option(
        True, "--competition/--no-competition", help="Also compute competition edges."
    ),
    threshold: float = typer.Option(
        0.0, "--threshold", help="Drop edges with weight <= this value from the graph."
    ),
) -> None:
    """Derive cross-feeding / competition edges and a summary graph."""
    import pandas as pd

    from gemfitcom.interactions import (
        competition_edges,
        cross_feeding_edges,
        summary_graph,
    )

    panel_df = pd.read_csv(panel)
    biomass_df = pd.read_csv(biomass) if biomass is not None else None

    output.mkdir(parents=True, exist_ok=True)
    xfeed = cross_feeding_edges(panel_df, biomass=biomass_df)
    xpath = _write_csv(output / "cross_feeding.csv", xfeed)

    paths: dict[str, Path] = {"cross_feeding": xpath}
    if include_competition:
        comp = competition_edges(panel_df, biomass=biomass_df)
        paths["competition"] = _write_csv(output / "competition.csv", comp)

    g = summary_graph(
        panel_df,
        biomass=biomass_df,
        include_competition=include_competition,
        threshold=threshold,
    )
    import networkx as nx

    graph_path = output / "graph.graphml"
    nx.write_graphml(g, graph_path)
    paths["graph"] = graph_path

    typer.echo(
        f"interactions: cross_feeding={len(xfeed)} "
        f"nodes={g.number_of_nodes()} edges={g.number_of_edges()}"
    )
    for k, p in paths.items():
        typer.echo(f"  {k:14s} → {p}")


# =========================================================================
# gapfill
# =========================================================================


@app.command("gapfill")
def gapfill_command(
    model_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to SBML model (.xml / .xml.gz)."
    ),
    source: str = typer.Option(
        ..., "--source", help="Model provenance: curated | agora2 | carveme."
    ),
    observed: str | None = typer.Option(
        None,
        "--observed",
        help="Comma-separated exchange IDs, e.g. 'EX_ac_e,EX_but_e'.",
    ),
    observed_file: Path | None = typer.Option(
        None,
        "--observed-file",
        exists=True,
        readable=True,
        help="YAML/JSON file with a top-level 'observed' list of exchange IDs.",
    ),
    kb: str = typer.Option("scfa", "--kb", help="KB name or path."),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory."),
) -> None:
    """Apply KB-driven gap-fill to a standalone model."""
    if observed is None and observed_file is None:
        raise typer.BadParameter("provide either --observed or --observed-file")

    observed_ids = _collect_observed(observed, observed_file)
    if not observed_ids:
        raise typer.BadParameter("observed product list is empty")

    from gemfitcom.gapfill.knowledge import load_kb
    from gemfitcom.gapfill.run import run_gapfill
    from gemfitcom.io.models import load_model

    model = load_model(model_path, source=source)
    kb_obj = load_kb(kb)
    report = run_gapfill(model, observed_products=observed_ids, kb=kb_obj, source=source)

    output.mkdir(parents=True, exist_ok=True)
    base = Path(model_path).stem
    filled_path = _save_model(model, output / f"{base}_gapfilled.xml")
    report_path = _write_gapfill_report(output / f"{base}_gapfill_report.json", report)

    typer.echo(
        f"gapfill: source={source} kb={kb} skipped={report.skipped} "
        f"added={len(report.products_added)} "
        f"already_present={len(report.products_already_present)} "
        f"no_kb_entry={len(report.products_missing_kb)}"
    )
    typer.echo(f"  model  → {filled_path}")
    typer.echo(f"  report → {report_path}")


def _collect_observed(observed: str | None, observed_file: Path | None) -> list[str]:
    ids: list[str] = []
    if observed is not None:
        ids.extend(x.strip() for x in observed.split(",") if x.strip())
    if observed_file is not None:
        import yaml

        raw = yaml.safe_load(observed_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("observed", [])
        if not isinstance(raw, list):
            raise typer.BadParameter(
                f"{observed_file}: expected a list or a mapping with key 'observed'"
            )
        ids.extend(str(x) for x in raw)
    return sorted(set(ids))


if __name__ == "__main__":
    app()
