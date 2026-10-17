"""Local ingestion checks for supported LeRobot dataset formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

LEROBOT_V2_STATISTICS_FILES = {
    "v2.0": "meta/stats.json",
    "v2.1": "meta/episodes_stats.jsonl",
}


class LocalDatasetValidationError(ValueError):
    """The local directory isn't a canonical upload candidate."""


def _reject_unmaterialized_acquisition(root_path: Path) -> None:
    acquisition = root_path / "acquisition-manifest.json"
    if not acquisition.exists():
        return
    try:
        source_manifest = json.loads(acquisition.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalDatasetValidationError("invalid acquisition-manifest.json") from exc
    upload_contract = source_manifest.get("upload_contract", {})
    if isinstance(upload_contract, dict) and upload_contract.get("source_is_upload_ready") is False:
        raise LocalDatasetValidationError(
            "this is acquisition source material, not a materialized LeRobot dataset"
        )


def detect_lerobot_format(root: str | Path) -> Literal["lerobot_v2", "lerobot_v3"]:
    """Read the dataset's authoritative metadata instead of guessing a default."""

    root_path = Path(root).expanduser().resolve(strict=True)
    _reject_unmaterialized_acquisition(root_path)
    info_path = root_path / "meta" / "info.json"
    if not info_path.is_file():
        raise LocalDatasetValidationError("missing meta/info.json")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalDatasetValidationError("meta/info.json is not valid JSON") from exc
    if not isinstance(info, dict):
        raise LocalDatasetValidationError("meta/info.json must contain an object")
    version = info.get("codebase_version")
    if isinstance(version, str) and version.lstrip("v").startswith("2."):
        return "lerobot_v2"
    if isinstance(version, str) and version.lstrip("v").startswith("3."):
        return "lerobot_v3"
    raise LocalDatasetValidationError(f"unsupported LeRobot codebase_version: {version!r}")


def validate_lerobot_root(
    root: str | Path,
    format: Literal["lerobot_v2", "lerobot_v3"],
) -> dict[str, object]:
    """Validate the declared local format boundary before manifest creation."""

    root_path = Path(root).expanduser().resolve(strict=True)
    _reject_unmaterialized_acquisition(root_path)
    info_path = root_path / "meta" / "info.json"
    if not info_path.is_file():
        raise LocalDatasetValidationError("missing meta/info.json")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalDatasetValidationError("meta/info.json is not valid JSON") from exc
    if not isinstance(info, dict):
        raise LocalDatasetValidationError("meta/info.json must contain an object")
    version = info.get("codebase_version")
    expected_major = "2." if format == "lerobot_v2" else "3."
    if not isinstance(version, str) or not version.lstrip("v").startswith(expected_major):
        raise LocalDatasetValidationError(
            f"expected {format} codebase_version, got {version!r}; relabeling is forbidden"
        )
    if format == "lerobot_v2":
        if version not in LEROBOT_V2_STATISTICS_FILES:
            raise LocalDatasetValidationError("supported LeRobot v2 revisions are v2.0 and v2.1")
        required = [
            root_path / "meta" / "tasks.jsonl",
            root_path / "meta" / "episodes.jsonl",
            root_path / LEROBOT_V2_STATISTICS_FILES[version],
            root_path / "data",
        ]
        missing = [str(path.relative_to(root_path)) for path in required if not path.exists()]
        if missing:
            raise LocalDatasetValidationError(f"incomplete LeRobot v2 artifact: missing {missing}")
    return info


def validate_lerobot_v3_root(root: str | Path) -> dict[str, object]:
    return validate_lerobot_root(root, "lerobot_v3")
