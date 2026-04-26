"""Medium registry: resolve medium names to :class:`Medium` instances.

Three lookup sources, searched in order:

1. In-memory custom registrations via :func:`register_medium`.
2. Built-in YAML files shipped under ``gemfitcom/data/media/*.yaml``.
3. Filesystem paths, when the argument contains a path separator or
   ``.yaml`` / ``.yml`` suffix.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from gemfitcom.medium.medium import Medium, medium_from_dict, medium_from_yaml

_BUILTIN_PACKAGE = "gemfitcom.data.media"
_CUSTOM_REGISTRY: dict[str, Medium] = {}


def load_medium(name_or_path: str | Path) -> Medium:
    """Resolve a medium by registered name or filesystem path.

    Args:
        name_or_path: Either a registered / built-in name (e.g. ``"YCFA"``),
            or a path to a ``*.yaml``/``*.yml`` file.

    Raises:
        FileNotFoundError: when a path is given that does not exist.
        KeyError: when a bare name is not registered and not built in.
    """
    if isinstance(name_or_path, Path) or _looks_like_path(str(name_or_path)):
        return medium_from_yaml(name_or_path)

    name = str(name_or_path)
    if name in _CUSTOM_REGISTRY:
        return _CUSTOM_REGISTRY[name]

    builtin = _load_builtin(name)
    if builtin is not None:
        return builtin

    available = sorted({*_CUSTOM_REGISTRY, *_list_builtin()})
    raise KeyError(
        f"medium {name!r} is not registered. Available: {available}. "
        "Pass a file path to load a custom medium."
    )


def list_media() -> list[str]:
    """Return all medium names resolvable by :func:`load_medium` (sorted)."""
    return sorted({*_CUSTOM_REGISTRY, *_list_builtin()})


def register_medium(name: str, source: str | Path | Medium | dict) -> Medium:
    """Register a medium under ``name`` for later :func:`load_medium` calls.

    Args:
        name: Registration key; overwrites any existing custom entry.
        source: One of
            * a :class:`Medium` instance,
            * a parsed dict (e.g. from ``yaml.safe_load``),
            * a filesystem path to a YAML file.

    Returns:
        The registered :class:`Medium`.
    """
    if isinstance(source, Medium):
        medium = source
    elif isinstance(source, dict):
        medium = medium_from_dict(source)
    else:
        medium = medium_from_yaml(source)
    _CUSTOM_REGISTRY[name] = medium
    return medium


def unregister_medium(name: str) -> None:
    """Remove a previously :func:`register_medium`-ed entry. No-op if absent."""
    _CUSTOM_REGISTRY.pop(name, None)


def clear_custom_registry() -> None:
    """Drop all custom registrations (does not affect built-ins)."""
    _CUSTOM_REGISTRY.clear()


def _looks_like_path(s: str) -> bool:
    return any(sep in s for sep in ("/", "\\")) or s.endswith((".yaml", ".yml"))


def _load_builtin(name: str) -> Medium | None:
    for suffix in (".yaml", ".yml"):
        filename = f"{name}{suffix}"
        try:
            ref = resources.files(_BUILTIN_PACKAGE).joinpath(filename)
        except (ModuleNotFoundError, FileNotFoundError):
            return None
        if ref.is_file():
            with resources.as_file(ref) as path:
                return medium_from_yaml(path)
    return None


def _list_builtin() -> list[str]:
    try:
        entries = list(resources.files(_BUILTIN_PACKAGE).iterdir())
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    names: list[str] = []
    for entry in entries:
        entry_name = entry.name
        if entry_name.endswith((".yaml", ".yml")):
            names.append(entry_name.rsplit(".", 1)[0])
    return names
