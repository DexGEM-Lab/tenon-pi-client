from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tenon_pi.libero import (
    LIBERO_LICENSE,
    LIBERO_REVISION,
    SourceFile,
    libero10_acquisition_manifest,
    verify_source_file,
    write_libero10_acquisition_manifest,
)


def test_exact_libero10_subset_manifest():
    manifest = libero10_acquisition_manifest()
    episodes = [file for file in manifest.files if file.role == "episode"]
    assert manifest.source_revision == LIBERO_REVISION
    assert manifest.source_license == LIBERO_LICENSE == "CC-BY-4.0"
    assert manifest.selected_episode_indices == [0, 1, 2, 6, 7, 8, 10, 12, 14, 27]
    assert manifest.total_episode_payload_bytes == 348_445_346
    assert manifest.total_frames == 2_806
    assert len(episodes) == 10
    assert {episode.task_index for episode in episodes} == set(range(10))
    assert len({episode.sha256 for episode in episodes}) == 10
    assert all(len(episode.sha256) == 64 for episode in episodes)


def test_manifest_forbids_resampling_and_requires_materialization(tmp_path: Path):
    destination = tmp_path / "acquisition.json"
    manifest = write_libero10_acquisition_manifest(destination)
    serialized = json.loads(destination.read_text())
    assert serialized["canonical_manifest_sha256"] == manifest.canonical_manifest_sha256
    assert manifest.conversion_contract["implicit_resampling"] == "forbidden"
    assert manifest.upload_contract["recommended_demo_format"] == "lerobot_v2"
    assert manifest.upload_contract["optional_canonical_format"] == "lerobot_v3"
    assert manifest.upload_contract["source_is_upload_ready"] is False
    assert manifest.upload_contract["client_must_not_upload_unrenumbered_source"] is True
    assert manifest.upload_contract["compatibility_profile"] == "tenon-native-v1"
    assert "renumbered episode/task/global" in manifest.upload_contract["required_handoff"]


def test_verify_source_file_checks_size_and_sha(tmp_path: Path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"fixture")
    source = SourceFile(
        relative_path="sample.bin",
        size_bytes=7,
        sha256=hashlib.sha256(b"fixture").hexdigest(),
        role="upstream_metadata",
    )
    verify_source_file(path, source)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_source_file(path, source)
