from __future__ import annotations

import json
from pathlib import Path

import pytest

from tenon_pi.datasets import (
    LocalDatasetValidationError,
    detect_lerobot_format,
    validate_lerobot_v3_root,
)
from tenon_pi.libero import write_libero10_acquisition_manifest


def test_accepts_declared_lerobot_v3_root(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    assert validate_lerobot_v3_root(tmp_path)["codebase_version"] == "v3.0"


@pytest.mark.parametrize(("version", "expected"), [("v2.1", "lerobot_v2"), ("v3.0", "lerobot_v3")])
def test_detects_authoritative_format_without_a_default(
    tmp_path: Path, version: str, expected: str
):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": version}))
    assert detect_lerobot_format(tmp_path) == expected


def test_rejects_v2_relabeling(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.0"}))
    with pytest.raises(LocalDatasetValidationError, match="relabeling is forbidden"):
        validate_lerobot_v3_root(tmp_path)


def test_rejects_acquisition_source_until_materialized(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    write_libero10_acquisition_manifest(tmp_path / "acquisition-manifest.json")
    with pytest.raises(LocalDatasetValidationError, match="not a materialized"):
        validate_lerobot_v3_root(tmp_path)


@pytest.mark.parametrize("version", ["v2.0", "v2.1"])
def test_v2_statistics_requirement_follows_revision(tmp_path, version):
    from tenon_pi.datasets import validate_lerobot_root

    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"codebase_version": version}))
    for name in ("tasks.jsonl", "episodes.jsonl"):
        (tmp_path / "meta" / name).write_text("{}\n")
    required = "stats.json" if version == "v2.0" else "episodes_stats.jsonl"
    other = "episodes_stats.jsonl" if version == "v2.0" else "stats.json"
    (tmp_path / "meta" / other).write_text("{}\n")
    with pytest.raises(LocalDatasetValidationError, match=required):
        validate_lerobot_root(tmp_path, "lerobot_v2")
    (tmp_path / "meta" / required).write_text("{}\n")
    (tmp_path / "meta" / other).unlink()
    assert validate_lerobot_root(tmp_path, "lerobot_v2")["codebase_version"] == version
