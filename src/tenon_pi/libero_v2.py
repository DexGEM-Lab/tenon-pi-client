"""Lossless native LeRobot v2 materialization for the pinned LIBERO subset."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .libero import AcquisitionManifest, libero10_acquisition_manifest, verify_source_file

_PRESERVED_COLUMNS = (
    "image",
    "wrist_image",
    "state",
    "actions",
    "timestamp",
    "frame_index",
)


@dataclass
class _RunningStats:
    size: int

    def __post_init__(self) -> None:
        import numpy as np

        self.count = 0
        self.sum = np.zeros(self.size, dtype=np.float64)
        self.sum_sq = np.zeros(self.size, dtype=np.float64)
        self.minimum = np.full(self.size, np.inf, dtype=np.float64)
        self.maximum = np.full(self.size, -np.inf, dtype=np.float64)

    def add(self, values: Any) -> None:
        import numpy as np

        array = np.asarray(values, dtype=np.float64).reshape(-1, self.size)
        if array.size == 0:
            return
        self.count += array.shape[0]
        self.sum += array.sum(axis=0)
        self.sum_sq += np.square(array).sum(axis=0)
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))

    def result(self) -> dict[str, list[float]]:
        import numpy as np

        if self.count == 0:
            raise ValueError("cannot compute stats for an empty feature")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0)
        return {
            "mean": mean.astype(np.float32).tolist(),
            "std": np.sqrt(variance).astype(np.float32).tolist(),
            "max": self.maximum.astype(np.float32).tolist(),
            "min": self.minimum.astype(np.float32).tolist(),
        }


def _imports() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LIBERO materialization requires: pip install 'tenon-pi-client[libero]'"
        ) from exc
    return np, pa, pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _decoded_image(cell: dict[str, Any], source_root: Path) -> Any:
    import numpy as np
    from PIL import Image

    data = cell.get("bytes")
    if data is None:
        relative = cell.get("path")
        if not isinstance(relative, str):
            raise ValueError("image cell has neither bytes nor path")
        data = (source_root / relative).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _add_image_stats(stats: _RunningStats, column: Any, source_root: Path) -> None:
    import numpy as np

    for cell in column.to_pylist():
        stats.add(_decoded_image(cell, source_root).astype(np.float32).reshape(-1, 3) / 255.0)


def _image_stats_shape(result: dict[str, list[float]]) -> dict[str, list[list[list[float]]]]:
    return {key: [[[value]] for value in values] for key, values in result.items()}


def _values(table: Any, name: str) -> Any:
    import numpy as np

    if name in {"state", "actions"}:
        return np.asarray(table[name].to_pylist(), dtype=np.float32)
    return np.asarray(table[name].to_numpy()).reshape(-1, 1)


def _replace_indices(table: Any, episode: int, task: int, start: int) -> Any:
    _, pa, _ = _imports()
    replacements = {
        "episode_index": pa.array([episode] * table.num_rows, type=pa.int64()),
        "task_index": pa.array([task] * table.num_rows, type=pa.int64()),
        "index": pa.array(range(start, start + table.num_rows), type=pa.int64()),
    }
    for name, values in replacements.items():
        position = table.schema.get_field_index(name)
        if position < 0:
            raise ValueError(f"source episode is missing {name}")
        table = table.set_column(position, name, values)
    return table


def _verify_preserved(source: Any, output: Any) -> None:
    if source.num_rows != output.num_rows:
        raise ValueError("row count changed")
    for name in _PRESERVED_COLUMNS:
        if not source[name].equals(output[name]):
            raise ValueError(f"preserved column changed: {name}")


def materialize_lerobot_v2_subset(
    source_root: str | Path,
    output_root: str | Path,
    manifest: AcquisitionManifest,
) -> Path:
    """Create a standalone renumbered v2 subset with exact retained content."""

    np, _, pq = _imports()
    source = Path(source_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must not already exist: {output}")
    (output / "data" / "chunk-000").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)

    info_entry = next(item for item in manifest.files if item.relative_path == "meta/info.json")
    verify_source_file(source / info_entry.relative_path, info_entry)
    info = json.loads((source / info_entry.relative_path).read_text())
    episodes = [item for item in manifest.files if item.role == "episode"]
    state_stats, action_stats = _RunningStats(8), _RunningStats(7)
    image_stats, wrist_stats = _RunningStats(3), _RunningStats(3)
    scalar_stats = {
        name: _RunningStats(1)
        for name in ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    }
    episode_meta: list[dict[str, Any]] = []
    task_meta: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    global_index = 0

    for output_episode, item in enumerate(episodes):
        if item.frames is None or item.task is None or item.source_episode_index is None:
            raise ValueError("episode manifest lacks provenance")
        input_file = source / item.relative_path
        verify_source_file(input_file, item)
        source_table = pq.read_table(input_file)
        if source_table.num_rows != item.frames:
            raise ValueError("source frame count mismatch")
        expected_frame = np.arange(item.frames, dtype=np.int64)
        if not np.array_equal(source_table["frame_index"].to_numpy(), expected_frame):
            raise ValueError("source frame_index isn't contiguous")
        table = _replace_indices(source_table, output_episode, output_episode, global_index)
        output_file = output / "data" / "chunk-000" / f"episode_{output_episode:06d}.parquet"
        pq.write_table(table, output_file, compression="snappy")
        reloaded = pq.read_table(output_file)
        _verify_preserved(source_table, reloaded)
        if reloaded["episode_index"].to_pylist() != [output_episode] * item.frames:
            raise ValueError("episode index rewrite failed")
        if reloaded["index"].to_pylist() != list(range(global_index, global_index + item.frames)):
            raise ValueError("global index rewrite failed")

        state_stats.add(_values(reloaded, "state"))
        action_stats.add(_values(reloaded, "actions"))
        _add_image_stats(image_stats, reloaded["image"], source)
        _add_image_stats(wrist_stats, reloaded["wrist_image"], source)
        for name, running_stats in scalar_stats.items():
            running_stats.add(_values(reloaded, name))
        episode_meta.append(
            {"episode_index": output_episode, "tasks": [item.task], "length": item.frames}
        )
        task_meta.append({"task_index": output_episode, "task": item.task})
        provenance.append(
            {
                "source_episode_index": item.source_episode_index,
                "output_episode_index": output_episode,
                "source_sha256": item.sha256,
                "output_sha256": _sha256(output_file),
                "frames": item.frames,
                "preserved_columns": list(_PRESERVED_COLUMNS),
            }
        )
        global_index += item.frames

    if global_index != manifest.total_frames:
        raise ValueError("materialized frame total mismatch")
    info.update(
        {
            "codebase_version": "v2.0",
            "total_episodes": len(episodes),
            "total_frames": global_index,
            "total_tasks": len(episodes),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 10,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
        }
    )
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (output / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in task_meta)
    )
    (output / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_meta)
    )
    output_stats: dict[str, Any] = {
        "image": _image_stats_shape(image_stats.result()),
        "wrist_image": _image_stats_shape(wrist_stats.result()),
        "state": state_stats.result(),
        "actions": action_stats.result(),
        **{name: value.result() for name, value in scalar_stats.items()},
    }
    (output / "meta" / "stats.json").write_text(
        json.dumps(output_stats, indent=4, allow_nan=False) + "\n"
    )
    (output / "meta" / "source_provenance.json").write_text(
        json.dumps(
            {
                "manifest_version": "tenon-libero10-mini-v2-v1",
                "source_repo_id": manifest.source_repo_id,
                "source_revision": manifest.source_revision,
                "source_license": manifest.source_license,
                "acquisition_manifest_sha256": manifest.canonical_manifest_sha256,
                "implicit_resampling": False,
                "stored_fps": 10,
                "rewritten_fields": ["episode_index", "task_index", "index"],
                "episodes": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "README.md").write_text(
        "# Tenon LIBERO-10 mini v2\n\n"
        f"Losslessly materialized from `{manifest.source_repo_id}@{manifest.source_revision}`; "
        "one retained episode per LIBERO-10 task; no resampling.\n\n"
        "License: CC-BY-4.0. Cite LIBERO: Liu et al., arXiv:2306.03310.\n"
    )
    return output


def materialize_libero10_v2(source_root: str | Path, output_root: str | Path) -> Path:
    source = Path(source_root)
    manifest_path = source / "acquisition-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source directory is missing acquisition-manifest.json")
    observed = AcquisitionManifest.model_validate_json(manifest_path.read_text())
    expected = libero10_acquisition_manifest()
    if observed.canonical_manifest_sha256 != expected.canonical_manifest_sha256:
        raise ValueError("acquisition manifest doesn't match the pinned selection")
    return materialize_lerobot_v2_subset(source, output_root, observed)
