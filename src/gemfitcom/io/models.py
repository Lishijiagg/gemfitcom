"""SBML genome-scale metabolic model loader."""

from __future__ import annotations

from pathlib import Path

import cobra


def load_model(
    path: str | Path,
    *,
    strain_name: str | None = None,
    source: str | None = None,
) -> cobra.Model:
    """Load a genome-scale metabolic model from an SBML file.

    Thin wrapper over :func:`cobra.io.read_sbml_model` that attaches optional
    metadata (``strain_name``, ``source``) to the returned model so downstream
    gap-fill can dispatch on model provenance.

    Args:
        path: path to the SBML file (``.xml`` / ``.xml.gz`` / ``.sbml``).
        strain_name: logical strain name. Attached at
            ``model.annotation['strain_name']``; also assigned to ``model.id``
            when the SBML did not provide a meaningful one.
        source: model provenance; typically one of ``curated`` / ``agora2`` /
            ``carveme``. Attached at ``model.annotation['source']``.

    Returns:
        A :class:`cobra.Model`.

    Raises:
        FileNotFoundError: if the SBML file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SBML model not found: {path}")
    model = cobra.io.read_sbml_model(str(path))
    if strain_name is not None:
        model.annotation["strain_name"] = strain_name
        if not model.id or model.id in ("", "default"):
            model.id = strain_name
    if source is not None:
        model.annotation["source"] = source
    return model
