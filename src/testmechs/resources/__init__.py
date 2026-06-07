"""Packaged paper reproduction resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from .._json_io import write_strict_json_atomic as _write_strict_json_atomic

_RESOURCE_CATEGORIES = ("fixture", "empirical_statistic", "monte_carlo_table")
_RESOURCE_SUMMARY_KEYS = {
    "package_version",
    "manifest_sha256",
    "resource_count",
    "fixture_count",
    "empirical_statistic_count",
    "monte_carlo_table_count",
}
_RESOURCE_SUMMARY_COUNT_KEYS = {
    "resource_count",
    "fixture_count",
    "empirical_statistic_count",
    "monte_carlo_table_count",
}
_SOURCE_ANCHOR_PREFIXES = {
    "fixture": "tests/python/fixtures/inputs/",
    "empirical_statistic": "manuscript/sources/arxiv-2404.11739v3/Statistics/",
    "monte_carlo_table": "manuscript/sources/arxiv-2404.11739v3/tables/",
}
_PACKAGE_PATH_PREFIXES = {
    "fixture": "testmechs/resources/fixtures/",
    "empirical_statistic": "testmechs/resources/statistics/",
    "monte_carlo_table": "testmechs/resources/tables/",
}


def _package_version() -> str:
    try:
        return version("testmechs")
    except PackageNotFoundError:
        return "0.1.0"


@dataclass(frozen=True)
class PaperReproductionResource:
    """Machine-readable metadata for one packaged paper reproduction resource."""

    category: str
    name: str
    package_path: str
    byte_count: int
    sha256: str
    source_anchor: str

    def __post_init__(self) -> None:
        _validate_resource_record_fields(
            category=self.category,
            name=self.name,
            package_path=self.package_path,
            byte_count=self.byte_count,
            sha256=self.sha256,
            source_anchor=self.source_anchor,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "package_path": self.package_path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "source_anchor": self.source_anchor,
        }


@dataclass(frozen=True)
class PaperReproductionResourceManifestPacket:
    """Validated strict-JSON packet for packaged paper reproduction resources."""

    resources: tuple[PaperReproductionResource, ...]
    summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.resources, tuple) or not all(
            isinstance(resource, PaperReproductionResource)
            for resource in self.resources
        ):
            raise ValueError(
                "resource manifest packet resources must be a tuple of "
                "PaperReproductionResource objects."
            )
        if not isinstance(self.summary, Mapping):
            raise ValueError("resource manifest packet summary must be a mapping.")
        summary = dict(self.summary)
        _validate_resource_manifest_summary(summary, self.resources)
        _validate_resource_manifest_records(self.resources)
        object.__setattr__(self, "summary", MappingProxyType(summary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "resources": [resource.to_dict() for resource in self.resources],
        }


def paper_reproduction_resource_manifest() -> tuple[PaperReproductionResource, ...]:
    """Return metadata for packaged fixture, statistic, and table resources."""

    package_root = files(__name__)
    specs = (
        (
            "fixture",
            "fixtures",
            ".csv",
            "tests/python/fixtures/inputs",
        ),
        (
            "empirical_statistic",
            "statistics",
            ".tex",
            "manuscript/sources/arxiv-2404.11739v3/Statistics",
        ),
        (
            "monte_carlo_table",
            "tables",
            ".tex",
            "manuscript/sources/arxiv-2404.11739v3/tables",
        ),
    )
    resources: list[PaperReproductionResource] = []
    for category, directory, suffix, source_root in specs:
        resource_dir = package_root / directory
        for resource in sorted(resource_dir.iterdir(), key=lambda item: item.name):
            if not resource.is_file() or not resource.name.endswith(suffix):
                continue
            payload = resource.read_bytes()
            resources.append(
                PaperReproductionResource(
                    category=category,
                    name=resource.name,
                    package_path=f"testmechs/resources/{directory}/{resource.name}",
                    byte_count=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    source_anchor=f"{source_root}/{resource.name}",
                )
            )
    return tuple(resources)


def paper_reproduction_resource_manifest_frame() -> pd.DataFrame:
    """Return the packaged paper reproduction resource manifest as a DataFrame."""

    return pd.DataFrame(
        [resource.to_dict() for resource in paper_reproduction_resource_manifest()],
        dtype=object,
    )


def _resource_manifest_digest(resource_records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        resource_records,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paper_reproduction_resource_manifest_payload() -> dict[str, Any]:
    return paper_reproduction_resource_manifest_packet().to_dict()


def paper_reproduction_resource_manifest_packet() -> PaperReproductionResourceManifestPacket:
    """Return a typed strict-JSON packet for packaged reproduction resources."""

    resources = paper_reproduction_resource_manifest()
    resource_records = [resource.to_dict() for resource in resources]
    counts = {
        category: sum(resource.category == category for resource in resources)
        for category in _RESOURCE_CATEGORIES
    }
    return PaperReproductionResourceManifestPacket(
        resources=resources,
        summary={
            "package_version": _package_version(),
            "manifest_sha256": _resource_manifest_digest(resource_records),
            "resource_count": len(resources),
            "fixture_count": counts["fixture"],
            "empirical_statistic_count": counts["empirical_statistic"],
            "monte_carlo_table_count": counts["monte_carlo_table"],
        },
    )


def write_paper_reproduction_resource_manifest_json(
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the packaged paper reproduction resource manifest as strict JSON."""

    if not isinstance(overwrite, bool):
        raise ValueError("resource manifest overwrite must be boolean.")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
        )
    payload = _paper_reproduction_resource_manifest_payload()
    _write_strict_json_atomic(path, payload)
    return payload


def _require_resource_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"resource manifest field {field_name} must be an object.")
    return value


def _require_manifest_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"resource manifest {field_name} must be an integer.")
    return value


def _validate_resource_record_fields(
    *,
    category: Any,
    name: Any,
    package_path: Any,
    byte_count: Any,
    sha256: Any,
    source_anchor: Any,
) -> int:
    if not all(
        isinstance(value, str)
        for value in (category, name, package_path, sha256, source_anchor)
    ):
        raise ValueError("resource manifest string fields must be strings.")
    if category not in _RESOURCE_CATEGORIES:
        raise ValueError(f"resource manifest category is not supported: {category!r}")
    if not name:
        raise ValueError("resource manifest name fields must be nonempty strings.")
    if "/" in name or "\\" in name:
        raise ValueError("resource manifest name fields must be file names.")
    byte_count = _require_manifest_int(byte_count, "byte_count")
    if byte_count <= 0:
        raise ValueError("resource manifest byte_count fields must be positive.")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("resource manifest sha256 fields must be lowercase hex digests.")
    expected_package_path = f"{_PACKAGE_PATH_PREFIXES[category]}{name}"
    if package_path != expected_package_path:
        raise ValueError(
            "resource manifest package_path must match its category and name exactly."
        )
    expected_source_anchor = f"{_SOURCE_ANCHOR_PREFIXES[category]}{name}"
    if source_anchor != expected_source_anchor:
        raise ValueError(
            "resource manifest source_anchor must match its category and name exactly."
        )
    return byte_count


def _load_resource_record(value: Any, index: int) -> PaperReproductionResource:
    record = _require_resource_object(value, f"resources[{index}]")
    required_keys = {
        "category",
        "name",
        "package_path",
        "byte_count",
        "sha256",
        "source_anchor",
    }
    if set(record) != required_keys:
        raise ValueError(
            "resource manifest record must contain category, name, package_path, "
            "byte_count, sha256, and source_anchor."
        )
    category = record["category"]
    name = record["name"]
    package_path = record["package_path"]
    byte_count = record["byte_count"]
    sha256 = record["sha256"]
    source_anchor = record["source_anchor"]
    byte_count = _validate_resource_record_fields(
        category=category,
        name=name,
        package_path=package_path,
        byte_count=byte_count,
        sha256=sha256,
        source_anchor=source_anchor,
    )
    return PaperReproductionResource(
        category=category,
        name=name,
        package_path=package_path,
        byte_count=byte_count,
        sha256=sha256,
        source_anchor=source_anchor,
    )


def _validate_resource_manifest_summary(
    summary: dict[str, Any],
    resources: tuple[PaperReproductionResource, ...],
) -> None:
    if set(summary) != _RESOURCE_SUMMARY_KEYS:
        raise ValueError(
            "resource manifest summary must contain package_version, manifest_sha256, "
            "resource_count, fixture_count, empirical_statistic_count, and "
            "monte_carlo_table_count."
        )
    package_version = summary["package_version"]
    if not isinstance(package_version, str) or not package_version:
        raise ValueError("resource manifest package_version must be a nonempty string.")
    if package_version != _package_version():
        raise ValueError(
            "resource manifest package_version does not match the installed package."
        )
    manifest_sha256 = summary["manifest_sha256"]
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise ValueError(
            "resource manifest manifest_sha256 must be a lowercase hex digest."
        )
    observed_manifest_sha256 = _resource_manifest_digest(
        [resource.to_dict() for resource in resources]
    )
    if manifest_sha256 != observed_manifest_sha256:
        raise ValueError(
            "resource manifest manifest_sha256 does not match resource records."
        )
    expected_counts = {
        "resource_count": len(resources),
        "fixture_count": sum(resource.category == "fixture" for resource in resources),
        "empirical_statistic_count": sum(
            resource.category == "empirical_statistic" for resource in resources
        ),
        "monte_carlo_table_count": sum(
            resource.category == "monte_carlo_table" for resource in resources
        ),
    }
    observed_counts = {
        key: _require_manifest_int(summary[key], key)
        for key in _RESOURCE_SUMMARY_COUNT_KEYS
    }
    if observed_counts != expected_counts:
        raise ValueError(
            "resource manifest summary counts do not match resource records."
        )


def _validate_resource_manifest_records(
    resources: tuple[PaperReproductionResource, ...],
) -> None:
    package_paths = [resource.package_path for resource in resources]
    source_anchors = [resource.source_anchor for resource in resources]
    if len(set(package_paths)) != len(package_paths):
        raise ValueError("resource manifest package_path values must be unique.")
    if len(set(source_anchors)) != len(source_anchors):
        raise ValueError("resource manifest source_anchor values must be unique.")

    current_records = tuple(
        resource.to_dict() for resource in paper_reproduction_resource_manifest()
    )
    loaded_records = tuple(resource.to_dict() for resource in resources)
    if loaded_records != current_records:
        raise ValueError(
            "resource manifest records do not match packaged reproduction resources in order."
        )


def load_paper_reproduction_resource_manifest_packet_json(
    path_like: str | Path,
) -> PaperReproductionResourceManifestPacket:
    """Load a validated strict-JSON packaged resource manifest packet."""

    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(
            f"paper reproduction resource manifest JSON file does not exist: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_resource_manifest_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"paper reproduction resource manifest must be valid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"summary", "resources"}:
        raise ValueError(
            "paper reproduction resource manifest JSON must contain only summary and resources."
        )
    summary = _require_resource_object(payload["summary"], "summary")
    resources = payload["resources"]
    if not isinstance(resources, list):
        raise ValueError("resource manifest resources field must be a list.")
    loaded_resources = tuple(
        _load_resource_record(resource, index)
        for index, resource in enumerate(resources)
    )
    _validate_resource_manifest_summary(summary, loaded_resources)
    _validate_resource_manifest_records(loaded_resources)
    return PaperReproductionResourceManifestPacket(
        resources=loaded_resources,
        summary=dict(summary),
    )


def _reject_resource_manifest_json_constant(value: str) -> None:
    raise ValueError(
        f"paper reproduction resource manifest must be strict JSON; found {value}."
    )


def load_paper_reproduction_resource_manifest_json(
    path_like: str | Path,
) -> tuple[PaperReproductionResource, ...]:
    """Load the resources from a saved strict-JSON manifest packet."""

    return load_paper_reproduction_resource_manifest_packet_json(path_like).resources


__all__ = [
    "PaperReproductionResource",
    "PaperReproductionResourceManifestPacket",
    "load_paper_reproduction_resource_manifest_json",
    "load_paper_reproduction_resource_manifest_packet_json",
    "paper_reproduction_resource_manifest",
    "paper_reproduction_resource_manifest_frame",
    "paper_reproduction_resource_manifest_packet",
    "write_paper_reproduction_resource_manifest_json",
]
