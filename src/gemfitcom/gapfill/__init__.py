"""Gap-fill: detect and add missing reactions based on observed HPLC products.

Strategy is dispatched on model source (see project design decisions):
    - curated  -> skip gap-fill
    - agora2   -> knowledge-base-driven (a) + (c) hybrid
    - carveme  -> knowledge-base-driven (a) + (c) hybrid

The cobrapy generic ``gapfill`` algorithm is intentionally NOT wrapped
here: auto gap-filling against a universal reaction database too easily
introduces physiologically implausible shortcuts. Instead, a curated
per-product knowledge base (YAML under ``gemfitcom/data/gapfill_kb/``)
supplies the minimal transport + exchange recipe for each observed HPLC
product.
"""

from gemfitcom.gapfill.apply import ApplyError, ApplyResult, apply_entry
from gemfitcom.gapfill.detect import DEFAULT_TOL, can_secrete, missing_products
from gemfitcom.gapfill.knowledge import (
    GapfillKB,
    GapfillKBEntry,
    KBError,
    MetaboliteSpec,
    ReactionSpec,
    clear_custom_kb_registry,
    kb_from_dict,
    kb_from_yaml,
    list_kbs,
    load_kb,
    register_kb,
    unregister_kb,
)
from gemfitcom.gapfill.report import GapfillReport, ProductOutcome, ProductStatus
from gemfitcom.gapfill.run import run_gapfill

__all__ = [
    "DEFAULT_TOL",
    "ApplyError",
    "ApplyResult",
    "GapfillKB",
    "GapfillKBEntry",
    "GapfillReport",
    "KBError",
    "MetaboliteSpec",
    "ProductOutcome",
    "ProductStatus",
    "ReactionSpec",
    "apply_entry",
    "can_secrete",
    "clear_custom_kb_registry",
    "kb_from_dict",
    "kb_from_yaml",
    "list_kbs",
    "load_kb",
    "missing_products",
    "register_kb",
    "run_gapfill",
    "unregister_kb",
]
