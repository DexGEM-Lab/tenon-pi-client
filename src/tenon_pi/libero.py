"""Pinned, lossless acquisition manifest for the ten-episode LIBERO demo subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from .errors import TenonTransportError

LIBERO_REPO_ID = "physical-intelligence/libero"
LIBERO_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
LIBERO_LICENSE = "CC-BY-4.0"
_RESOLVE_BASE = f"https://huggingface.co/datasets/{LIBERO_REPO_ID}/resolve/{LIBERO_REVISION}"


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    size_bytes: int
    sha256: str
    role: Literal["upstream_metadata", "episode"]
    source_episode_index: int | None = None
    task_index: int | None = None
    frames: int | None = None
    task: str | None = None

    @property
    def url(self) -> str:
        return f"{_RESOLVE_BASE}/{self.relative_path}"


class AcquisitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal["libero10-mini-acquisition-v1"]
    source_repo_id: str
    source_revision: str
    source_license: str
    source_format: Literal["lerobot_v2"]
    selected_suite: Literal["libero_10"]
    selected_episode_indices: list[int]
    total_episode_payload_bytes: int
    total_frames: int
    temporal_provenance: dict[str, str]
    conversion_contract: dict[str, str]
    upload_contract: dict[str, str | bool]
    attribution: list[str]
    files: list[SourceFile]
    canonical_manifest_sha256: str


_METADATA = (
    SourceFile(
        relative_path="README.md",
        size_bytes=3157,
        sha256="b178f7bb726feb0bbe82a6836895a962b91f957ae328ae3340a269c93b5e9b6e",
        role="upstream_metadata",
    ),
    SourceFile(
        relative_path="meta/info.json",
        size_bytes=2131,
        sha256="77c0da93c67960bac803a8ea25dabf380d752c26c1d91c54151121dd10e7d82f",
        role="upstream_metadata",
    ),
    SourceFile(
        relative_path="meta/tasks.jsonl",
        size_bytes=3375,
        sha256="e72ef3cf7ca7a3abe70b3e142a0d1acf2719d2a3f27598869092cc4f1762bb99",
        role="upstream_metadata",
    ),
    SourceFile(
        relative_path="meta/episodes.jsonl",
        size_bytes=180384,
        sha256="0479a2ced6117700e4638a1da54dccd9f60ae5f47485ea6d91390c869befcf08",
        role="upstream_metadata",
    ),
    SourceFile(
        relative_path="meta/stats.json",
        size_bytes=5855,
        sha256="1fe6c2d548a7998394488ae68bc2be46a51c8cd3798af6ab74fff79f8b20e293",
        role="upstream_metadata",
    ),
)

_EPISODES = (
    (
        0,
        0,
        214,
        28_009_331,
        "1dff46a888aac51da3c4fedb6c6fe596df16f8cba47dd10d0e0c0ca565fd1d65",
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
    ),
    (
        1,
        1,
        284,
        37_414_512,
        "6c416803bba7649172e6b3c0cb2c579a570fcb234e41249f7589c52809971916",
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    ),
    (
        2,
        2,
        345,
        35_350_193,
        "e590ad6f1f13e894d6e3a250ba54346b16803e873403061a7f0978903830d53d",
        "put the yellow and white mug in the microwave and close it",
    ),
    (
        6,
        3,
        279,
        33_946_834,
        "202f9eb18c0d4d0ff2fb398d1423d9a3c947801b6027c631f095a007017d1db2",
        "turn on the stove and put the moka pot on it",
    ),
    (
        7,
        4,
        272,
        36_515_760,
        "a8e0b399a4a625236792ed4e989f98406e1c4010469b336fc71d24bf5cf813ad",
        "put both the alphabet soup and the cream cheese box in the basket",
    ),
    (
        8,
        5,
        259,
        38_264_721,
        "746602605cf5b61ee0a0830094e452453d0e068283346b76eb66f3d201cdcd24",
        "put both the alphabet soup and the tomato sauce in the basket",
    ),
    (
        10,
        6,
        383,
        42_209_473,
        "030c49e6d1907c8092e5ed2e6d7472e8ce00481e7ba442d582831f8163d0ee3a",
        "put both moka pots on the stove",
    ),
    (
        12,
        7,
        279,
        38_292_394,
        "52df2f981836316f361059970b096c45bea45e59c54152343189db68688a349d",
        "put both the cream cheese box and the butter in the basket",
    ),
    (
        14,
        8,
        257,
        30_907_484,
        "8cec14466bc6a51dd3fb3355a0387e4c29af12355bfc2b4a3e5c9cf52a5d653b",
        "put the black bowl in the bottom drawer of the cabinet and close it",
    ),
    (
        27,
        9,
        234,
        27_534_644,
        "46e23143129f9b7e17c9e496e250ef0d1370b296c67b23e8d4dac7d79687b274",
        "pick up the book and place it in the back compartment of the caddy",
    ),
)


def _episode_files() -> tuple[SourceFile, ...]:
    return tuple(
        SourceFile(
            relative_path=f"data/chunk-000/episode_{episode_index:06d}.parquet",
            size_bytes=size_bytes,
            sha256=sha256,
            role="episode",
            source_episode_index=episode_index,
            task_index=task_index,
            frames=frames,
            task=task,
        )
        for episode_index, task_index, frames, size_bytes, sha256, task in _EPISODES
    )


def _canonical_digest(data: dict[str, object]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def libero10_acquisition_manifest() -> AcquisitionManifest:
    files = [*_METADATA, *_episode_files()]
    payload: dict[str, object] = {
        "manifest_version": "libero10-mini-acquisition-v1",
        "source_repo_id": LIBERO_REPO_ID,
        "source_revision": LIBERO_REVISION,
        "source_license": LIBERO_LICENSE,
        "source_format": "lerobot_v2",
        "selected_suite": "libero_10",
        "selected_episode_indices": [row[0] for row in _EPISODES],
        "total_episode_payload_bytes": sum(row[3] for row in _EPISODES),
        "total_frames": sum(row[2] for row in _EPISODES),
        "temporal_provenance": {
            "source_environment_controller_hz": "20",
            "stored_fps": "10",
            "stored_timestamp": "synthetic frame_index / 10",
            "prior_filtering": "irregular no-op transition removal",
            "elapsed_time_recoverable": "false",
        },
        "conversion_contract": {
            "row_order": "preserve exactly within every source episode",
            "state_action_values": "preserve exactly; no interpolation",
            "implicit_resampling": "forbidden",
            "episode_selection": "one immutable source episode per LIBERO-10 task",
        },
        "upload_contract": {
            "recommended_demo_format": "lerobot_v2",
            "optional_canonical_format": "lerobot_v3",
            "source_is_upload_ready": False,
            "required_handoff": (
                "materialize the selected episodes with renumbered episode/task/global indices; "
                "preserve images, state, actions, timestamps, and frame order"
            ),
            "client_must_not_upload_unrenumbered_source": True,
            "compatibility_profile": "tenon-native-v1",
        },
        "attribution": [
            "LIBERO: Liu et al., arXiv:2306.03310",
            "Converted dataset: physical-intelligence/libero",
            "Modified RLDS provenance: openvla/modified_libero_rlds",
        ],
        "files": [file.model_dump(mode="json") for file in files],
    }
    return AcquisitionManifest.model_validate(
        {**payload, "canonical_manifest_sha256": _canonical_digest(payload)}
    )


def write_libero10_acquisition_manifest(path: str | Path) -> AcquisitionManifest:
    manifest = libero10_acquisition_manifest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_source_file(path: Path, source: SourceFile) -> None:
    if path.stat().st_size != source.size_bytes:
        raise ValueError(f"size mismatch for {source.relative_path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != source.sha256:
        raise ValueError(f"SHA-256 mismatch for {source.relative_path}")


def acquire_libero10_source(
    destination: str | Path,
    *,
    client: httpx.Client | None = None,
) -> AcquisitionManifest:
    """Download and verify the pinned source blobs without transforming them.

    The result remains LeRobot v2 source material and MUST NOT be uploaded as a
    ``lerobot_v3`` dataset version. See the generated handoff contract.
    """

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    manifest = write_libero10_acquisition_manifest(root / "acquisition-manifest.json")
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    try:
        for source in manifest.files:
            target = root / source.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                verify_source_file(target, source)
                continue
            partial = target.with_name(f".{target.name}.part")
            digest = hashlib.sha256()
            size = 0
            try:
                with http.stream("GET", source.url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as output:
                        for block in response.iter_bytes():
                            output.write(block)
                            digest.update(block)
                            size += len(block)
            except (httpx.HTTPError, OSError) as exc:
                partial.unlink(missing_ok=True)
                raise TenonTransportError(
                    f"failed to acquire pinned LIBERO source {source.relative_path}: {exc}"
                ) from exc
            if size != source.size_bytes or digest.hexdigest() != source.sha256:
                partial.unlink(missing_ok=True)
                raise ValueError(f"download verification failed for {source.relative_path}")
            partial.replace(target)
    finally:
        if owns_client:
            http.close()
    return manifest
