"""Lightweight tqdm wrapper for long-running simulation loops.

Every long-running loop in the package (dFBA time integration, dMICOM QP
sweeps, parameter grid searches) accepts a ``progress`` flag that is
forwarded to :func:`progress_bar`. When ``enabled=False`` (the library
default), the iterable is returned untouched — no tqdm import, no overhead,
no extra output in pytest or programmatic pipelines. When ``enabled=True``
(typically from the CLI or an interactive notebook), the iterable is wrapped
with :mod:`tqdm.auto` so the right display backend is picked up
automatically (TTY, Jupyter, or fallback).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    *,
    enabled: bool = False,
    desc: str | None = None,
    total: int | None = None,
    leave: bool = False,
    unit: str = "it",
) -> Iterable[T]:
    """Wrap ``iterable`` in a tqdm progress bar when ``enabled``.

    Args:
        iterable: Any iterable. Yielded items are returned unchanged.
        enabled: If False (default), ``iterable`` is returned untouched and
            no tqdm import is performed. If True, ``iterable`` is wrapped
            with :func:`tqdm.auto.tqdm`.
        desc: Optional label shown next to the bar.
        total: Optional total iteration count for iterables whose length
            cannot be inferred (generators, ``zip``, etc.). If ``None`` and
            ``iterable`` has ``__len__``, tqdm figures it out.
        leave: Whether to leave the completed bar on screen. Defaults to
            False so nested / successive runs do not pile up in notebooks.
        unit: Per-iteration unit label (``"step"``, ``"strain"``, ...).

    Returns:
        Either ``iterable`` itself (when ``enabled=False``) or a
        ``tqdm``-wrapped iterator.
    """
    if not enabled:
        return iterable

    from tqdm.auto import tqdm

    return tqdm(iterable, desc=desc, total=total, leave=leave, unit=unit)


__all__ = ["progress_bar"]
