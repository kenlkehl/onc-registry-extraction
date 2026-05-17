"""Helpers for locating vendored registry reference data."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent


def reference_data_dir(name: str) -> Path:
    """Return the vendored reference-data directory for *name*.

    Source checkouts keep reference data at the repository root. Built wheels
    can carry the same files under ``onc_registry_pipeline/resources``.
    """
    candidates = [
        PACKAGE_ROOT / "resources" / name,
        SOURCE_ROOT / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def default_data_dict_dir() -> Path:
    """Return the default NAACCR data dictionary directory."""
    return reference_data_dir("NAACCRDataItems")


def default_seer_manuals_dir() -> Path:
    """Return the default SEER/NAACCR manuals directory."""
    return reference_data_dir("SEERManuals")


def default_data_items_csv() -> Path:
    return default_data_dict_dir() / "DataItems.csv"


def default_code_list_csv() -> Path:
    return default_data_dict_dir() / "CodeList.csv"


def default_alternate_names_csv() -> Path:
    return default_data_dict_dir() / "AlternateNames.csv"


def resolve_reference_path(path: str | Path, reference_dir_name: str) -> Path:
    """Resolve a reference-data path without tying defaults to ``cwd``.

    Relative paths still work normally when they exist from the current working
    directory. If a relative path starts with a known vendored reference-data
    directory name and is not present in ``cwd``, it falls back to the vendored
    package/repo copy.
    """
    resolved = Path(path).expanduser()
    if resolved.is_absolute() or resolved.exists():
        return resolved

    parts = resolved.parts
    if parts and parts[0] == reference_dir_name:
        return reference_data_dir(reference_dir_name).joinpath(*parts[1:])
    return resolved
