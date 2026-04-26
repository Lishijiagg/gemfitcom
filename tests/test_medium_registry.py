"""Tests for the medium registry and YAML loader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from gemfitcom.medium import (
    Medium,
    MediumError,
    clear_custom_registry,
    list_media,
    load_medium,
    medium_from_dict,
    register_medium,
    unregister_medium,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    clear_custom_registry()
    yield
    clear_custom_registry()


def test_builtin_ycfa_loads() -> None:
    m = load_medium("YCFA")
    assert isinstance(m, Medium)
    assert m.name == "YCFA"
    assert "EX_glc__D_e" in m.pool_components
    assert m.pool_components["EX_glc__D_e"] == pytest.approx(5.0)
    assert "EX_h2o_e" in m.unlimited_components
    assert not (set(m.pool_components) & m.unlimited_components)


def test_list_media_includes_builtin() -> None:
    assert "YCFA" in list_media()


def test_load_medium_unknown_name_lists_available() -> None:
    with pytest.raises(KeyError, match="not registered"):
        load_medium("NoSuchMedium123")


def test_medium_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(MediumError, match="must be a mapping"):
        medium_from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_medium_from_dict_rejects_missing_name() -> None:
    with pytest.raises(MediumError, match="'name'"):
        medium_from_dict({"pool_components": {}})


def test_medium_rejects_overlap_between_pool_and_unlimited() -> None:
    with pytest.raises(MediumError, match="overlap"):
        medium_from_dict(
            {
                "name": "BadOverlap",
                "pool_components": {"EX_glc__D_e": 5.0},
                "unlimited_components": ["EX_glc__D_e"],
            }
        )


def test_medium_rejects_negative_concentration() -> None:
    with pytest.raises(MediumError, match=">= 0"):
        medium_from_dict(
            {
                "name": "Bad",
                "pool_components": {"EX_glc__D_e": -1.0},
            }
        )


def test_medium_rejects_malformed_exchange_id() -> None:
    with pytest.raises(MediumError, match="exchange reaction ID"):
        medium_from_dict(
            {
                "name": "Bad",
                "pool_components": {"glucose": 5.0},
            }
        )


def test_medium_rejects_non_numeric_concentration() -> None:
    with pytest.raises(MediumError, match="not a number"):
        medium_from_dict(
            {
                "name": "Bad",
                "pool_components": {"EX_glc__D_e": "five"},
            }
        )


def test_register_and_load_custom_medium_from_dict() -> None:
    data = {
        "name": "Minimal",
        "pool_components": {"EX_glc__D_e": 5.0},
        "unlimited_components": ["EX_h2o_e"],
    }
    register_medium("Minimal", data)
    m = load_medium("Minimal")
    assert m.name == "Minimal"
    assert "Minimal" in list_media()

    unregister_medium("Minimal")
    assert "Minimal" not in list_media()


def test_load_medium_from_path(tmp_path: Path) -> None:
    yaml_text = (
        "name: FromFile\n"
        "pool_components:\n"
        "  EX_glc__D_e: 2.0\n"
        "unlimited_components:\n"
        "  - EX_h2o_e\n"
    )
    p = tmp_path / "custom.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    m = load_medium(p)
    assert m.name == "FromFile"
    assert m.pool_components["EX_glc__D_e"] == pytest.approx(2.0)


def test_load_medium_path_string_with_slash(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("name: X\npool_components: {EX_glc__D_e: 1.0}\n", encoding="utf-8")
    m = load_medium(str(p))
    assert m.name == "X"


def test_load_medium_path_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_medium(tmp_path / "nope.yaml")


def test_custom_registration_shadows_builtin() -> None:
    replacement = {"name": "YCFA", "pool_components": {"EX_glc__D_e": 99.0}}
    register_medium("YCFA", replacement)
    m = load_medium("YCFA")
    assert m.pool_components["EX_glc__D_e"] == pytest.approx(99.0)


def test_register_medium_accepts_medium_instance() -> None:
    m = Medium(
        name="InMem",
        pool_components={"EX_glc__D_e": 3.0},
        unlimited_components=frozenset({"EX_h2o_e"}),
    )
    register_medium("InMem", m)
    loaded = load_medium("InMem")
    assert loaded is m


def test_extra_top_level_fields_land_in_metadata() -> None:
    m = medium_from_dict(
        {
            "name": "X",
            "pool_components": {"EX_glc__D_e": 1.0},
            "source": "paper 2020",
        }
    )
    assert m.metadata.get("source") == "paper 2020"
