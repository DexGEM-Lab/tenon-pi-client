"""Executable, data-only profile-driven canonicalization and normalization.

This module prepares immutable LeRobot v2/v3 data and profile assets. It contains
no model, training, server, hardware-command, or GPU implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .normalization import (
    OPENPI_NORMALIZE_SOURCE,
    OPENPI_NORMALIZE_SOURCE_SHA256,
    NormalizationStats,
    RunningStats,
    serialize_normalization,
)
from .robot_profile import RobotProfile

PROFILE_ASSET_ID = "robot-profile-v1"
PREPARATION_VERSION = "tenon-preparation-v2"
NORMALIZATION_VERSION = "tenon-normalization-v2"
OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"

InputSpace = Literal["native", "canonical"]
DatasetFormatName = Literal["lerobot_v2", "lerobot_v3"]
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_ID = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+")
_REQUIRED_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def _profile(descriptor: dict[str, Any]) -> RobotProfile:
    return RobotProfile.model_validate(descriptor)


def _camera_columns(descriptor: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"observation.images.{c.id}" for c in _profile(descriptor).cameras)


class PreparationError(ValueError):
    """The input cannot be prepared without guessing or changing semantics."""


@dataclass(frozen=True)
class Episode:
    episode_index: int
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class PreparationResult:
    root: Path
    dataset: Path
    assets: Path
    schema: Path
    dataset_content_digest: str
    transform_digest: str
    normalization_digest: str
    dataset_repo_id: str
    rows: int
    format: DatasetFormatName = "lerobot_v2"

    def json_value(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "dataset": str(self.dataset),
            "assets": str(self.assets),
            "schema": str(self.schema),
            "dataset_content_digest": self.dataset_content_digest,
            "transform_digest": self.transform_digest,
            "normalization_digest": self.normalization_digest,
            "dataset_repo_id": self.dataset_repo_id,
            "rows": self.rows,
            "format": self.format,
        }


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised by fresh-install boundary
        raise RuntimeError(
            "Robot preparation requires: pip install 'tenon-pi-client[data]'"
        ) from exc
    return pa, pq


def canonical_json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def dataset_file_manifest(root: str | Path) -> list[dict[str, object]]:
    """Return the sorted sorted immutable tree manifest."""

    directory = Path(root).expanduser().resolve(strict=True)
    result: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PreparationError(f"dataset symlinks are forbidden: {path.relative_to(directory)}")
        if path.is_file():
            result.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    if not result:
        raise PreparationError("dataset tree is empty")
    return result


def _json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{label} must be readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must contain a JSON object")
    return value


def validate_repo_id(value: object) -> str:
    """Validate logical normalization identity, never a cache or filesystem path."""

    if (
        not isinstance(value, str)
        or _REPO_ID.fullmatch(value) is None
        or value.split("/", 1)[1] in {".", ".."}
    ):
        raise PreparationError(
            "Robot repo ID must be exactly namespace/name with safe ASCII segments"
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PreparationError(f"{label} must be an actual sha256:<64 lowercase hex> digest")
    return value


def build_transform_descriptor(profile: RobotProfile | dict[str, Any]) -> dict[str, Any]:
    return RobotProfile.model_validate(profile).descriptor()


def _dataset_format(root: Path) -> DatasetFormatName:
    info = _json_object(root / "meta/info.json", "LeRobot meta/info.json")
    if "tenon_view" in info or any(root.rglob("*view-manifest.json")):
        raise PreparationError("a derived view is not a source dataset; never upload a view")
    version = info.get("codebase_version")
    if isinstance(version, str) and version.lstrip("v").startswith("2."):
        return "lerobot_v2"
    if version == "v3.0":
        return "lerobot_v3"
    raise PreparationError(f"unsupported Robot codebase_version {version!r}; v2.x/v3.0 only")


def _v2_info(root: Path, input_space: InputSpace, descriptor: dict[str, Any]) -> dict[str, Any]:
    profile = _profile(descriptor)
    info_path = root / "meta" / "info.json"
    info = _json_object(info_path, "LeRobot meta/info.json")
    version = info.get("codebase_version")
    from .dataset_metadata import required_statistics_file

    statistics_file = required_statistics_file(version)
    if info.get("fps") != profile.controller_hz:
        raise PreparationError("Robot data must declare the profile fps; no implicit resampling")
    if info.get("total_videos") not in (None, 0):
        raise PreparationError(
            "Robot native preparation is PNG-only; video materialization is separate work"
        )
    for relative in ("meta/tasks.jsonl", "meta/episodes.jsonl", statistics_file):
        if not (root / relative).is_file():
            raise PreparationError(f"incomplete LeRobot v2 dataset: missing {relative}")
    features = info.get("features")
    if not isinstance(features, dict):
        raise PreparationError("meta/info.json features must be an object")
    for key, mapping in (("observation.state", profile.state), ("action", profile.action)):
        expected_channels = mapping.native if input_space == "native" else mapping.canonical
        expected_width = len(expected_channels)
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("shape") != [expected_width]:
            raise PreparationError(f"{key} must declare shape [{expected_width}] for {input_space}")
        if feature.get("names") != [c.name for c in expected_channels]:
            raise PreparationError(f"{key} names must exactly match the supplied profile order")
        if feature.get("dtype") != "float32":
            raise PreparationError(f"{key} must declare dtype=float32")
    for camera in profile.cameras:
        camera_id, key = camera.id, f"observation.images.{camera.id}"
        feature = features.get(key)
        if (
            not isinstance(feature, dict)
            or feature.get("shape") != [camera.height, camera.width, 3]
            or feature.get("dtype") != "image"
        ):
            raise PreparationError(
                f"camera {camera_id} must declare PNG image shape "
                f"{[camera.height, camera.width, 3]}; "
                "video materialization/transcoding is separate work"
            )
    return info


def _parquet_files(root: Path) -> list[Path]:
    values = sorted((root / "data").glob("**/*.parquet"))
    if not values:
        raise PreparationError("LeRobot v2 dataset contains no data parquet files")
    return values


def _metadata_tasks(root: Path) -> set[int]:
    result: set[int] = set()
    for line_number, line in enumerate(
        (root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            item = json.loads(line)
            task_index = int(item["task_index"])
            task = item["task"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PreparationError(f"invalid meta/tasks.jsonl line {line_number}") from exc
        if task_index < 0 or task_index in result or not isinstance(task, str) or not task:
            raise PreparationError(
                "task metadata must have unique nonnegative indices and nonempty text"
            )
        result.add(task_index)
    if not result:
        raise PreparationError("meta/tasks.jsonl is empty")
    return result


def _metadata_episodes(root: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for line_number, line in enumerate(
        (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            item = json.loads(line)
            episode = int(item["episode_index"])
            length = int(item["length"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PreparationError(f"invalid meta/episodes.jsonl line {line_number}") from exc
        if episode in result or episode < 0 or length < 1:
            raise PreparationError("episode metadata must have unique nonnegative indices/lengths")
        result[episode] = length
    if not result:
        raise PreparationError("meta/episodes.jsonl is empty")
    return result


def _validate_png_images(
    root: Path, table: Any, relative: Path, descriptor: dict[str, Any]
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Robot preparation requires: pip install 'tenon-pi-client[data]'"
        ) from exc
    import io

    for capture in _profile(descriptor).cameras:
        camera, column_name = capture.id, f"observation.images.{capture.id}"
        for row, cell in enumerate(table[column_name].to_pylist()):
            if not isinstance(cell, dict):
                raise PreparationError(
                    f"{relative}:{row} camera {camera} must be a LeRobot image cell"
                )
            encoded = cell.get("bytes")
            if encoded is None:
                image_path = cell.get("path")
                if not isinstance(image_path, str) or not image_path:
                    raise PreparationError(
                        f"{relative}:{row} camera {camera} has neither bytes nor path"
                    )
                candidate = (root / image_path).resolve()
                if root not in candidate.parents or not candidate.is_file():
                    raise PreparationError(
                        f"{relative}:{row} camera {camera} path escapes/misses dataset"
                    )
                encoded = candidate.read_bytes()
            if not isinstance(encoded, bytes):
                raise PreparationError(f"{relative}:{row} camera {camera} image bytes are invalid")
            try:
                with Image.open(io.BytesIO(encoded)) as image:
                    image.load()
                    if (
                        image.format != "PNG"
                        or image.size != (capture.width, capture.height)
                        or image.mode != "RGB"
                    ):
                        raise PreparationError(
                            f"{relative}:{row} camera {camera} must be RGB PNG "
                            "with the declared profile dimensions"
                        )
            except PreparationError:
                raise
            except Exception as exc:
                raise PreparationError(
                    f"{relative}:{row} camera {camera} is not a valid PNG"
                ) from exc


def _validate_vectors(
    values: np.ndarray, *, space: InputSpace, descriptor: dict[str, Any], label: str
) -> None:
    profile = _profile(descriptor)
    mapping = profile.state if label == "observation.state" else profile.action
    width = len(mapping.native if space == "native" else mapping.canonical)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise PreparationError(f"{label} must contain finite [{width}] vectors")
    lower, upper = mapping.bounds(space)
    if space == "canonical":
        for i, c in enumerate(mapping.canonical):
            if c.native_index is None and np.any(values[:, i] != np.float32(c.constant)):
                raise PreparationError(f"{label} virtual coordinate differs from profile")
    if np.any(values < lower - 1e-6) or np.any(values > upper + 1e-6):
        raise PreparationError(
            f"{label} exceeds supplied calibration bounds; preparation never clips"
        )


def _load_episodes(
    dataset_root: str | Path,
    descriptor: dict[str, Any],
    *,
    space: InputSpace = "canonical",
    validate_images: bool = True,
) -> list[Episode]:
    profile = _profile(descriptor)
    root = Path(dataset_root).expanduser().resolve(strict=True)
    if _dataset_format(root) == "lerobot_v3":
        if space != "canonical":
            raise PreparationError("v3 requires canonical; native v3 is unsupported")
        from .dataset_v3 import load_v3_episodes

        return load_v3_episodes(root, descriptor)
    info = _v2_info(root, space, descriptor)
    _, pq = _pyarrow()
    metadata = _metadata_episodes(root)
    task_metadata = _metadata_tasks(root)
    episodes: list[Episode] = []
    next_global_index = 0
    for expected_episode, path in enumerate(_parquet_files(root)):
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        missing = set((*_REQUIRED_COLUMNS, *_camera_columns(descriptor))) - names
        if missing:
            raise PreparationError(
                f"{path.relative_to(root)} lacks required columns: {sorted(missing)}"
            )
        if validate_images:
            image_table = pq.read_table(path, columns=list(_camera_columns(descriptor)))
            _validate_png_images(root, image_table, path.relative_to(root), descriptor)
        table = pq.read_table(path, columns=list(_REQUIRED_COLUMNS))
        if table.num_rows < 1:
            raise PreparationError(f"empty episode parquet: {path.relative_to(root)}")
        try:
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            episode_indices = np.asarray(
                table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64
            )
            frame_indices = np.asarray(
                table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64
            )
            global_indices = np.asarray(
                table["index"].to_numpy(zero_copy_only=False), dtype=np.int64
            )
            task_indices = np.asarray(
                table["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64
            )
            timestamps = np.asarray(
                table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise PreparationError(
                f"invalid vector/index columns in {path.relative_to(root)}"
            ) from exc
        _validate_vectors(states, space=space, descriptor=descriptor, label="observation.state")
        _validate_vectors(actions, space=space, descriptor=descriptor, label="action")
        rows = table.num_rows
        if not np.array_equal(episode_indices, np.full(rows, expected_episode)):
            raise PreparationError("episode parquets must be sorted contiguous indices from zero")
        if not np.array_equal(frame_indices, np.arange(rows)):
            raise PreparationError("frame_index must start at zero and be contiguous per episode")
        expected_global = np.arange(next_global_index, next_global_index + rows)
        if not np.array_equal(global_indices, expected_global):
            raise PreparationError("global index must preserve contiguous dataset row order")
        if not set(task_indices.tolist()).issubset(task_metadata):
            raise PreparationError("row task_index is absent from tasks metadata")
        if not np.isfinite(timestamps).all() or (
            rows > 1
            # Pinned hf_transform_to_torch tensorizes Python timestamps as
            # float32 before check_timestamps_sync, even for float64 Arrow data.
            and not np.all(
                np.abs(np.diff(timestamps.astype(np.float32)) - 1.0 / profile.controller_hz) <= 1e-4
            )
        ):
            raise PreparationError(
                "timestamps must be finite consecutive profile-rate values within 100 us "
                "in the pinned reader; never resample"
            )
        if metadata.get(expected_episode) != rows:
            raise PreparationError("episode parquet length disagrees with episodes metadata")
        episodes.append(
            Episode(
                episode_index=expected_episode,
                states=states,
                actions=actions,
                timestamps=timestamps,
            )
        )
        next_global_index += rows
    if set(metadata) != set(range(len(episodes))):
        raise PreparationError("episode metadata must exactly match sorted parquet episodes")
    if (
        info.get("total_episodes") != len(episodes)
        or info.get("total_frames") != next_global_index
        or info.get("total_tasks") != len(task_metadata)
    ):
        raise PreparationError("info.json episode/frame/task totals disagree with data")
    if validate_images:
        from .dataset_metadata import validate_v2_metadata_statistics

        validate_v2_metadata_statistics(root, info)
    return episodes


def iter_horizon_samples(
    dataset_root: str | Path,
    descriptor: dict[str, Any],
    *,
    validate_images: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield canonical rows with same-episode absolute horizon target windows."""

    profile = _profile(descriptor)
    for episode in _load_episodes(
        dataset_root,
        descriptor,
        space="canonical",
        validate_images=validate_images,
    ):
        for frame_index in range(len(episode.states)):
            targets = episode.actions[frame_index : frame_index + profile.action_horizon_steps]
            if len(targets) < profile.action_horizon_steps:
                targets = np.concatenate(
                    [
                        targets,
                        np.repeat(
                            episode.actions[-1][None],
                            profile.action_horizon_steps - len(targets),
                            axis=0,
                        ),
                    ],
                    axis=0,
                )
            yield {
                "observation.state": episode.states[frame_index].copy(),
                "action": targets.astype(np.float32, copy=True),
                "episode_index": episode.episode_index,
                "frame_index": frame_index,
                "timestamp": float(episode.timestamps[frame_index]),
            }


def compute_normalization(
    dataset_root: str | Path,
    descriptor: dict[str, Any],
    *,
    validate_images: bool = True,
) -> tuple[bytes, int]:
    """Compute pinned OpenPI-compatible state/action statistics from every row."""

    profile = _profile(descriptor)
    state_stats = RunningStats()
    action_stats = RunningStats()
    samples = iter_horizon_samples(dataset_root, descriptor, validate_images=validate_images)
    rows = 0
    for sample in samples:
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        targets = np.asarray(sample["action"], dtype=np.float32)
        state_stats.update(np.asarray(state, dtype=np.float64))
        action_stats.update(np.asarray(profile.model_actions(state, targets), dtype=np.float64))
        rows += 1
    stats: dict[str, NormalizationStats] = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    return serialize_normalization(stats), rows


def _materialize_native(source: Path, destination: Path, descriptor: dict[str, Any]) -> None:
    profile = _profile(descriptor)
    pa, pq = _pyarrow()
    if _dataset_format(source) != "lerobot_v2":
        raise PreparationError("v3 requires canonical; native v3 is unsupported")
    _v2_info(source, "native", descriptor)
    _load_episodes(source, descriptor, space="native")
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    for source_path in _parquet_files(source):
        relative = source_path.relative_to(source)
        output_path = destination / relative
        source_table = pq.read_table(source_path)
        states = np.asarray(source_table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(source_table["action"].to_pylist(), dtype=np.float32)
        output_states = np.stack([profile.state.to_canonical(row) for row in states])
        output_actions = np.stack([profile.action.to_canonical(row) for row in actions])
        table = source_table
        for name, values in (
            ("observation.state", output_states),
            ("action", output_actions),
        ):
            vector_type = pa.list_(pa.float32(), values.shape[1])
            position = table.schema.get_field_index(name)
            old_field = table.schema.field(position)
            field = pa.field(
                name,
                vector_type,
                nullable=old_field.nullable,
                metadata=old_field.metadata,
            )
            array = pa.array(values.tolist(), type=vector_type)
            table = table.set_column(position, field, array)
        from .dataset_metadata import canonical_arrow_vectors

        table = canonical_arrow_vectors(
            table,
            {
                "observation.state": len(profile.state.canonical),
                "action": len(profile.action.canonical),
            },
        )
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, output_path)
        reloaded = pq.read_table(output_path)
        if reloaded.num_rows != source_table.num_rows:
            raise PreparationError("native materialization changed row count")
        for name in source_table.column_names:
            if name in {"observation.state", "action"}:
                continue
            if not source_table[name].equals(reloaded[name]):
                raise PreparationError(f"native materialization changed preserved column {name}")
    info_path = destination / "meta" / "info.json"
    info = _json_object(info_path, "LeRobot meta/info.json")
    for name, mapping in (("observation.state", profile.state), ("action", profile.action)):
        info["features"][name]["shape"] = [len(mapping.canonical)]
        info["features"][name]["dtype"] = "float32"
        info["features"][name]["names"] = [c.name for c in mapping.canonical]
    info_path.write_text(json.dumps(info, indent=4, allow_nan=False) + "\n", encoding="utf-8")
    from .dataset_metadata import write_v2_metadata_statistics

    write_v2_metadata_statistics(destination, info)


def _copy_canonical(source: Path, destination: Path, descriptor: dict[str, Any]) -> None:
    _load_episodes(source, descriptor, space="canonical")
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    if dataset_file_manifest(source) != dataset_file_manifest(destination):
        raise PreparationError("canonical copy changed dataset bytes")


def _normalization_payload(content: bytes, descriptor: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise TypeError("normalization payload is not an object")
        stats = payload["norm_stats"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PreparationError("norm_stats.json has invalid OpenPI structure") from exc
    if not isinstance(stats, dict) or set(stats) != {"state", "actions"}:
        raise PreparationError("normalization must contain state and actions")
    profile = _profile(descriptor)
    for name, width in (
        ("state", len(profile.state.canonical)),
        ("actions", len(profile.action.canonical)),
    ):
        value = stats[name]
        if not isinstance(value, dict) or set(value) != {"mean", "std", "q01", "q99"}:
            raise PreparationError(f"normalization {name} fields are invalid")
        for statistic in ("mean", "std", "q01", "q99"):
            array = np.asarray(value[statistic], dtype=np.float64)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise PreparationError(f"normalization {name}.{statistic} must be finite canonical")
        if np.any(np.asarray(value["std"]) < 0) or np.any(
            np.asarray(value["q01"]) > np.asarray(value["q99"])
        ):
            raise PreparationError(f"normalization {name} ranges are invalid")
    return payload


def build_robot_schema(normalization: bytes, descriptor: dict[str, Any]) -> dict[str, Any]:
    _normalization_payload(normalization, descriptor)
    profile = _profile(descriptor)
    if descriptor != profile.descriptor():
        raise PreparationError("descriptor must be canonical validated profile data")
    schema = profile.robot_schema("sha256:" + hashlib.sha256(normalization).hexdigest())
    # Stable lists avoid process-dependent JSON set ordering.
    return schema.model_dump(mode="json")


def _write_json(path: Path, value: object, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=sort_keys, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _v3_normalization_contract(profile: RobotProfile) -> dict[str, Any]:
    return {
        "dataset_format": "lerobot_v3",
        "episode_tail_padding": "repeat_last_action",
        "statistics_schedule": {
            "order": "dataset_row_order",
            "transformed_dtype": "float32",
            "update_dtype": "float64",
            "per_row_updates": [
                ["state", [1, len(profile.state.canonical)]],
                ["actions", [profile.action_horizon_steps, len(profile.action.canonical)]],
            ],
        },
    }


def prepare_dataset(
    source: str | Path,
    output: str | Path,
    *,
    input_space: InputSpace,
    dataset_repo_id: str,
    profile: RobotProfile | dict[str, Any],
) -> PreparationResult:
    """Prepare canonical data, exact normalization assets, and a bound RobotSchema."""

    source_root = Path(source).expanduser().resolve(strict=True)
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output directory must not already exist: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise PreparationError("output must not be inside the source dataset")
    logical_repo_id = validate_repo_id(dataset_repo_id)
    source_files = dataset_file_manifest(source_root)
    descriptor = build_transform_descriptor(profile)
    profile = _profile(descriptor)
    dataset_format = _dataset_format(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent)))
    try:
        dataset = stage / "dataset"
        if input_space == "native":
            _materialize_native(source_root, dataset, descriptor)
        elif input_space == "canonical":
            _copy_canonical(source_root, dataset, descriptor)
        else:  # pragma: no cover - Literal/type boundary
            raise PreparationError(f"unsupported Robot input space: {input_space}")
        dataset_files = dataset_file_manifest(dataset)
        dataset_digest = canonical_json_digest(dataset_files)
        normalization, rows = compute_normalization(dataset, descriptor, validate_images=False)
        assets = stage / "assets" / PROFILE_ASSET_ID
        assets.mkdir(parents=True)
        norm_path = assets / "norm_stats.json"
        norm_path.write_bytes(normalization)
        normalization_digest = file_sha256(norm_path)
        transform_digest = canonical_json_digest(descriptor)
        normalization_manifest = {
            "version": NORMALIZATION_VERSION,
            "dataset_repo_id": logical_repo_id,
            "dataset_content_digest": dataset_digest,
            "dataset_files": dataset_files,
            "transform_digest": transform_digest,
            "normalization_digest": normalization_digest,
            "openpi_revision": OPENPI_REVISION,
            "state_count": rows,
            "action_count": rows * profile.action_horizon_steps,
            "action_horizon_steps": profile.action_horizon_steps,
            "state_width": len(profile.state.canonical),
            "action_width": len(profile.action.canonical),
            "normalization": "quantile",
            "sampling": "all_rows_in_episode_order",
        }
        if dataset_format == "lerobot_v3":
            normalization_manifest.update(_v3_normalization_contract(profile))
        _write_json(assets / "normalization-manifest.json", normalization_manifest)
        _write_json(assets / "transform.json", descriptor)
        schema = build_robot_schema(normalization, descriptor)
        _write_json(stage / "robot-schema.json", schema)
        preparation_manifest = {
            "version": PREPARATION_VERSION,
            "dataset_repo_id": logical_repo_id,
            "input_space": input_space,
            "source_dataset_content_digest": canonical_json_digest(source_files),
            "source_dataset_files": source_files,
            "dataset_content_digest": dataset_digest,
            "dataset_files": dataset_files,
            "transform_digest": transform_digest,
            "normalization_digest": normalization_digest,
            "robot_schema_digest": canonical_json_digest(schema),
            "rows": rows,
            "canonicalization": (
                "profile_native_to_canonical"
                if input_space == "native"
                else "validated_identity_canonical"
            ),
            "implicit_resampling": False,
            "image_transcoding": False,
            "horizon": profile.action_horizon_steps,
            "profile_version": profile.profile_version,
            "normalization_source": {
                "project": "Physical-Intelligence/openpi",
                "license": "Apache-2.0",
                "revision": OPENPI_REVISION,
                "path": OPENPI_NORMALIZE_SOURCE,
                "source_sha256": OPENPI_NORMALIZE_SOURCE_SHA256,
            },
            "artifacts": {
                "norm_stats.json": file_sha256(norm_path),
                "normalization-manifest.json": file_sha256(assets / "normalization-manifest.json"),
                "transform.json": file_sha256(assets / "transform.json"),
                "robot-schema.json": file_sha256(stage / "robot-schema.json"),
            },
        }
        if dataset_format == "lerobot_v3":
            preparation_manifest["format"] = dataset_format
        _write_json(stage / "preparation-manifest.json", preparation_manifest)
        if dataset_file_manifest(source_root) != source_files:
            raise PreparationError("source dataset changed during preparation")
        verify_preparation(stage)
        stage.replace(output_root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return PreparationResult(
        root=output_root,
        dataset=output_root / "dataset",
        assets=output_root / "assets" / PROFILE_ASSET_ID,
        schema=output_root / "robot-schema.json",
        dataset_content_digest=dataset_digest,
        transform_digest=transform_digest,
        normalization_digest=normalization_digest,
        dataset_repo_id=logical_repo_id,
        rows=rows,
        format=dataset_format,
    )


def verify_preparation(root: str | Path) -> PreparationResult:
    """Regenerate manifests, statistics, and schema; reject any byte/semantic drift."""

    package = Path(root).expanduser().resolve(strict=True)
    dataset_file_manifest(package)  # Reject symlinks across assets/schema as well as data.
    dataset = package / "dataset"
    assets = package / "assets" / PROFILE_ASSET_ID
    schema_path = package / "robot-schema.json"
    preparation = _json_object(package / "preparation-manifest.json", "preparation manifest")
    descriptor = _json_object(assets / "transform.json", "Robot transform descriptor")
    normalization_manifest = _json_object(
        assets / "normalization-manifest.json", "normalization manifest"
    )
    normalization = (assets / "norm_stats.json").read_bytes()
    profile = _profile(descriptor)
    _normalization_payload(normalization, descriptor)
    expected_descriptor = profile.descriptor()
    if descriptor != expected_descriptor:
        raise PreparationError("transform.json differs from the committed public descriptor")
    dataset_files = dataset_file_manifest(dataset)
    dataset_format = _dataset_format(dataset)
    dataset_digest = canonical_json_digest(dataset_files)
    regenerated, rows = compute_normalization(dataset, descriptor)
    if regenerated != normalization:
        raise PreparationError(
            "norm_stats.json bytes do not match regenerated horizon dataset statistics"
        )
    normalization_digest = file_sha256(assets / "norm_stats.json")
    transform_digest = canonical_json_digest(descriptor)
    dataset_repo_id = validate_repo_id(preparation.get("dataset_repo_id"))
    expected_normalization_manifest = {
        "version": NORMALIZATION_VERSION,
        "dataset_repo_id": dataset_repo_id,
        "dataset_content_digest": dataset_digest,
        "dataset_files": dataset_files,
        "transform_digest": transform_digest,
        "normalization_digest": normalization_digest,
        "openpi_revision": OPENPI_REVISION,
        "state_count": rows,
        "action_count": rows * profile.action_horizon_steps,
        "action_horizon_steps": profile.action_horizon_steps,
        "state_width": len(profile.state.canonical),
        "action_width": len(profile.action.canonical),
        "normalization": "quantile",
        "sampling": "all_rows_in_episode_order",
    }
    if dataset_format == "lerobot_v3":
        expected_normalization_manifest.update(_v3_normalization_contract(profile))
    if any(
        key in normalization_manifest
        for key in ("dataset_cache_repo_id", "dataset_root", "dataset_home")
    ):
        raise PreparationError("transient cache locator entered normalization identity")
    if normalization_manifest != expected_normalization_manifest:
        raise PreparationError("normalization-manifest.json does not bind exact data/assets")
    schema = _json_object(schema_path, "Robot RobotSchema")
    expected_schema = build_robot_schema(normalization, descriptor)
    if schema != expected_schema:
        raise PreparationError("robot-schema.json does not match calibrated assets")
    source_files = preparation.get("source_dataset_files")
    input_space = preparation.get("input_space")
    expected_canonicalization = {
        "native": "profile_native_to_canonical",
        "canonical": "validated_identity_canonical",
    }
    expected_normalization_source = {
        "project": "Physical-Intelligence/openpi",
        "license": "Apache-2.0",
        "revision": OPENPI_REVISION,
        "path": OPENPI_NORMALIZE_SOURCE,
        "source_sha256": OPENPI_NORMALIZE_SOURCE_SHA256,
    }
    if (
        preparation.get("version") != PREPARATION_VERSION
        or preparation.get("format", "lerobot_v2") != dataset_format
        or (dataset_format == "lerobot_v3" and input_space != "canonical")
        or (input_space == "canonical" and source_files != dataset_files)
        or preparation.get("dataset_repo_id") != dataset_repo_id
        or input_space not in expected_canonicalization
        or preparation.get("canonicalization") != expected_canonicalization[input_space]
        or not isinstance(source_files, list)
        or preparation.get("source_dataset_content_digest") != canonical_json_digest(source_files)
        or preparation.get("dataset_content_digest") != dataset_digest
        or preparation.get("dataset_files") != dataset_files
        or preparation.get("transform_digest") != transform_digest
        or preparation.get("normalization_digest") != normalization_digest
        or preparation.get("robot_schema_digest") != canonical_json_digest(schema)
        or preparation.get("rows") != rows
        or preparation.get("implicit_resampling") is not False
        or preparation.get("image_transcoding") is not False
        or preparation.get("horizon") != profile.action_horizon_steps
        or preparation.get("profile_version") != profile.profile_version
        or preparation.get("normalization_source") != expected_normalization_source
    ):
        raise PreparationError("preparation-manifest.json binding mismatch")
    artifacts = preparation.get("artifacts")
    expected_artifacts = {
        "norm_stats.json": file_sha256(assets / "norm_stats.json"),
        "normalization-manifest.json": file_sha256(assets / "normalization-manifest.json"),
        "transform.json": file_sha256(assets / "transform.json"),
        "robot-schema.json": file_sha256(schema_path),
    }
    if artifacts != expected_artifacts:
        raise PreparationError("prepared artifact byte digests changed")
    return PreparationResult(
        root=package,
        dataset=dataset,
        assets=assets,
        schema=schema_path,
        dataset_content_digest=dataset_digest,
        transform_digest=transform_digest,
        normalization_digest=normalization_digest,
        dataset_repo_id=dataset_repo_id,
        rows=rows,
        format=dataset_format,
    )


def build_bundle_manifest(root: str | Path) -> dict[str, Any]:
    """Verify and describe every file in one prepared upload bundle."""

    result = verify_preparation(root)
    from .uploads import scan_dataset_directory

    files = scan_dataset_directory(result.root)
    return {
        "manifest_version": "tenon-prepared-bundle-v2",
        "format": result.format,
        "compatibility_profile": "tenon-native-v1",
        "dataset_repo_id": result.dataset_repo_id,
        "dataset_content_digest": result.dataset_content_digest,
        "dataset_root": "dataset",
        "assets_root": f"assets/{PROFILE_ASSET_ID}",
        "robot_schema": "robot-schema.json",
        "preparation_manifest": "preparation-manifest.json",
        "files": [item.manifest.model_dump(mode="json") for item in files],
    }


def write_bundle_manifest(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Verify a package, scan its whole root, and write an external local manifest."""

    package = Path(root).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if destination == package or package in destination.parents:
        raise PreparationError(
            "bundle manifest output must be outside the prepared root it describes"
        )
    value = build_bundle_manifest(package)
    _write_json(destination, value)
    return value
