"""Offline discovery and an explicit lossless JSONL-to-LeRobot PNG import path."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .preparation import (
    InputSpace,
    PreparationError,
    _load_episodes,
    _pyarrow,
    canonical_json_digest,
    dataset_file_manifest,
    file_sha256,
)
from .robot_profile import RobotProfile


def inspect_data(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve(strict=True)
    if (root / "preparation-manifest.json").is_file():
        from .preparation import verify_preparation

        return {
            "kind": "prepared_bundle",
            "verified": True,
            **verify_preparation(root).json_value(),
            "next": "upload the whole prepared root",
        }
    info_path = root / "meta/info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        version = info.get("codebase_version")
        return {
            "kind": "lerobot",
            "codebase_version": version,
            "fps": info.get("fps"),
            "features": info.get("features"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "verified": False,
            "next": "validate explicit profile/schema and source semantics before upload",
        }
    files = dataset_file_manifest(root)
    return {
        "kind": "unrecognized",
        "verified": False,
        "files": files,
        "supported_import": (
            "explicit rows.jsonl with state/action/timestamps/episode IDs and PNG paths"
        ),
        "next": "inspect source metadata; supply profile and conversion plan; never infer units",
    }


def import_jsonl(
    source: str | Path, output: str | Path, *, profile: RobotProfile, input_space: InputSpace
) -> Path:
    """Import ordered explicit records, without resampling, clipping or image encoding.

    Each line has exactly episode_index, frame_index, timestamp, state, action,
    task, images (camera ID -> source-relative PNG path). Source pixels are copied
    as bytes; the source JSONL and image digests are recorded for provenance.
    """
    if input_space not in {"native", "canonical"}:
        raise ValueError("input_space must be native or canonical")
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if destination.exists() or source.parent == destination or source.parent in destination.parents:
        raise ValueError("output must be new and outside the source directory")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if not rows:
        raise PreparationError("JSONL must contain nonempty ordered records")
    keys = {"episode_index", "frame_index", "timestamp", "state", "action", "task", "images"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != keys:
            raise PreparationError(f"each JSONL record requires exactly {sorted(keys)}")
        if (
            type(row["episode_index"]) is not int
            or type(row["frame_index"]) is not int
            or type(row["timestamp"]) not in (int, float)
            or not np.isfinite(row["timestamp"])
            or not isinstance(row["task"], str)
            or not row["task"]
        ):
            raise PreparationError(
                "explicit integer indices, finite time and nonempty task required"
            )
        for key in ("state", "action"):
            values = row[key]
            if not isinstance(values, list) or any(type(v) not in (int, float) for v in values):
                raise PreparationError("state/action must be numeric vectors, not inferred fields")
            original = np.asarray(values, dtype=np.float64)
            cast = np.asarray(values, dtype=np.float32)
            if not np.isfinite(original).all() or not np.array_equal(
                original, cast.astype(np.float64)
            ):
                raise PreparationError(
                    "source vectors must be exactly float32-representable; "
                    "explicit precision conversion is required"
                )
        if not isinstance(row["images"], dict) or set(row["images"]) != {
            c.id for c in profile.cameras
        }:
            raise PreparationError("image camera IDs must exactly match the supplied profile")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    pa, pq = _pyarrow()
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    features = {}
    for key, mapping in (("observation.state", profile.state), ("action", profile.action)):
        channels = mapping.native if input_space == "native" else mapping.canonical
        features[key] = dict(
            dtype="float32", shape=[len(channels)], names=[c.name for c in channels]
        )
    features.update(
        {
            "timestamp": dict(dtype="float64", shape=[1]),
            **{
                k: dict(dtype="int64", shape=[1])
                for k in ("frame_index", "episode_index", "index", "task_index")
            },
        }
    )
    features.update(
        {
            f"observation.images.{c.id}": dict(dtype="image", shape=[c.height, c.width, 3])
            for c in profile.cameras
        }
    )
    tasks: list[str] = []
    episodes: list[dict[str, Any]] = []
    source_images: dict[str, str] = {}
    source_digest = file_sha256(source)
    try:
        (stage / "meta").mkdir()
        (stage / "data/chunk-000").mkdir(parents=True)
        cursor = 0
        while cursor < len(rows):
            episode = len(episodes)
            subset = []
            episode_tasks = []
            while cursor < len(rows) and rows[cursor]["episode_index"] == episode:
                row = rows[cursor]
                if row["frame_index"] != len(subset):
                    raise PreparationError("frame indices must start at zero and be consecutive")
                if row["task"] not in tasks:
                    tasks.append(row["task"])
                if row["task"] not in episode_tasks:
                    episode_tasks.append(row["task"])
                record = {
                    "observation.state": row["state"],
                    "action": row["action"],
                    "timestamp": row["timestamp"],
                    "frame_index": row["frame_index"],
                    "episode_index": episode,
                    "index": cursor,
                    "task_index": tasks.index(row["task"]),
                }
                for camera, relative in row["images"].items():
                    if (
                        not isinstance(relative, str)
                        or Path(relative).is_absolute()
                        or ".." in Path(relative).parts
                    ):
                        raise PreparationError(
                            "images must be source-relative paths without traversal"
                        )
                    path = source.parent / relative
                    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
                        raise PreparationError(
                            "image source must be an existing regular non-symlink file"
                        )
                    digest = file_sha256(path)
                    if relative in source_images and source_images[relative] != digest:
                        raise PreparationError("source image changed during import")
                    source_images[relative] = digest
                    record[f"observation.images.{camera}"] = {
                        "bytes": path.read_bytes(),
                        "path": None,
                    }
                subset.append(record)
                cursor += 1
            if not subset:
                raise PreparationError("episodes must be contiguous from zero in source row order")
            fields = [
                pa.field(
                    k,
                    image_type
                    if f["dtype"] == "image"
                    else pa.list_(pa.float32(), f["shape"][0])
                    if k in {"action", "observation.state"}
                    else getattr(pa, f["dtype"])(),
                )
                for k, f in features.items()
            ]
            from .dataset_metadata import arrow_feature_metadata

            table = pa.Table.from_pylist(
                subset, schema=pa.schema(fields, metadata=arrow_feature_metadata(features))
            )
            pq.write_table(
                table, stage / f"data/chunk-000/episode_{episode:06d}.parquet", compression="snappy"
            )
            episodes.append(dict(episode_index=episode, length=len(subset), tasks=episode_tasks))
        info = dict(
            codebase_version="v2.1",
            fps=profile.controller_hz,
            features=features,
            total_episodes=len(episodes),
            total_frames=len(rows),
            total_tasks=len(tasks),
            total_videos=0,
            total_chunks=1,
            chunks_size=1000,
            splits={"train": f"0:{len(episodes)}"},
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        (stage / "meta/info.json").write_text(json.dumps(info) + "\n")
        (stage / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
        (stage / "meta/tasks.jsonl").write_text(
            "".join(json.dumps(dict(task_index=i, task=t)) + "\n" for i, t in enumerate(tasks))
        )
        from .dataset_metadata import write_v2_metadata_statistics

        write_v2_metadata_statistics(stage, info)
        _load_episodes(stage, profile.descriptor(), space=input_space)
        if source_digest != file_sha256(source) or any(
            file_sha256(source.parent / p) != h for p, h in source_images.items()
        ):
            raise PreparationError("source changed during import")
        provenance = dict(
            version="tenon-jsonl-import-v1",
            source_jsonl_sha256=source_digest,
            source_images=source_images,
            rows=len(rows),
            input_space=input_space,
            profile_digest=profile.digest(),
            image_transcoding=False,
            implicit_resampling=False,
            dataset_content_digest=canonical_json_digest(dataset_file_manifest(stage)),
        )
        (stage / "import-manifest.json").write_text(
            json.dumps(provenance, sort_keys=True, indent=2) + "\n"
        )
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination
