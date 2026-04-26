# Methodology

## Kinetics fit (V<sub>max</sub>, K<sub>m</sub>)

GemFitCom estimates Michaelis–Menten parameters by minimizing
$1 - R^2$ between the simulated dFBA biomass trajectory and the observed
OD-derived biomass curve. The R² is computed on the **raw** biomass scale
(not normalized) because normalizing both trajectories by their own max
destroys absolute-scale information and creates a structural identifiability
ridge along which very different `(vmax, km)` pairs all score near 1.0.

The minimization runs in two stages:

1. **Global search** with `scipy.optimize.differential_evolution` over the
   user-supplied `(vmax_bounds, km_bounds)` rectangle. Conservative settings
   (`maxiter=50`, `popsize=15` by default) keep wall time manageable.
2. **Local refinement** on a uniform `grid_points × grid_points` grid
   spanning `±grid_span × optimum` in each dimension. The grid serves
   double duty: it polishes the DE optimum and provides the R² surface
   for the parameter-sensitivity heatmap.

### What identifiability requires

Because uptake at high substrate is approximately V<sub>max</sub>, the
biomass curve only constrains both V<sub>max</sub> and K<sub>m</sub> when
the substrate **actually depletes** within the observation horizon. If
glucose at 5 mM is barely consumed in 6 h, you can recover V<sub>max</sub>
but not K<sub>m</sub>. Plan experiments long enough to see the deceleration
phase.

## Three community-simulation modes

The `simulate` subcommand exposes three integration schemes:

### `sequential_dfba`

The classical dynamic FBA loop: at each time step, every strain solves its
own FBA independently with the current shared pool concentrations as
upper bounds on uptake. Pool updates aggregate per-strain fluxes weighted
by per-strain biomass. Cheap and exposes time-resolved cross-feeding.

### `micom`

A single MICOM optimization: the community FBA finds an allocation of
fluxes that satisfies a cooperative tradeoff between maximum-individual
growth and total community growth, parameterized by `tradeoff_alpha ∈ [0, 1]`.
Lower α favours strict community optimum, higher α approaches each strain's
unconstrained maximum. No time dimension — the result is a snapshot.

### `fusion` (dMICOM)

GemFitCom's contribution: at each dFBA time step, instead of one
independent FBA per strain, we run a MICOM cooperative-tradeoff
optimization across the entire community. This produces dynamic
trajectories that respect cooperativity throughout the simulation —
useful when cross-feeding evolves over time and cooperative effects
matter even early in the curve.

## Gap-fill strategy

For models tagged `model_source: curated`, gap-fill is a no-op. For
`agora2` or `carveme` models, the KB-driven gap-fill compares observed
HPLC products against the model's exchange list:

- For each observed product not currently producible, the KB describes
  the canonical biosynthesis pathway as a set of `(reactions, metabolites)`
  to add.
- Reactions and metabolites are added only if not already present.
- A report records what was added, what was already present, and what
  the KB had no entry for.

The default knowledge base ships seven SCFA pathways (acetate, butyrate,
propionate, formate, lactate-D, lactate-L, succinate) under
`src/gemfitcom/data/gapfill_kb/scfa.yaml`. Custom KBs follow the same
schema and can be passed via `--kb`.
