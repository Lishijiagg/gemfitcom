"""Simulation modes.

Available modes:
    - ``mono``             single-strain dFBA (see :mod:`gemfitcom.kinetics.mono_dfba`)
    - ``sequential_dfba``  per-strain FBA at each step with shared metabolite pool
    - ``micom``            MICOM cooperative-tradeoff steady-state community FBA
    - ``fusion`` (dMICOM)  dynamic MICOM: cooperative-tradeoff QP at each time step
"""

from gemfitcom.simulate.fusion import (
    FusionResult,
    simulate_fusion_dmicom,
)
from gemfitcom.simulate.micom import (
    DEFAULT_FRACTION,
    DEFAULT_UNLIMITED_UPTAKE,
    DEFAULT_UPTAKE,
    CommunityMember,
    MICOMResult,
    simulate_micom,
)
from gemfitcom.simulate.sequential_dfba import (
    DEFAULT_DT,
    SequentialDFBAResult,
    StrainSpec,
    simulate_sequential_dfba,
)

__all__ = [
    "DEFAULT_DT",
    "DEFAULT_FRACTION",
    "DEFAULT_UNLIMITED_UPTAKE",
    "DEFAULT_UPTAKE",
    "CommunityMember",
    "FusionResult",
    "MICOMResult",
    "SequentialDFBAResult",
    "StrainSpec",
    "simulate_fusion_dmicom",
    "simulate_micom",
    "simulate_sequential_dfba",
]
