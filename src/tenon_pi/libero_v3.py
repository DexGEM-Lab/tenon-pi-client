"""Lossless LIBERO v2 source to canonical LeRobot v3 materialization."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, cast

from .libero import AcquisitionManifest, libero10_acquisition_manifest, verify_source_file

_REPO_ID = "tenon/libero10-mini-v3"
_IMAGE_KEY = "observation.images.image"
_WRIST_KEY = "observation.images.wrist_image"
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"


def _dependencies() -> tuple[Any, Any, Any, Any]:
    if sys.version_info < (3, 12):
        raise RuntimeError("LeRobot 0.6.1 materialization requires Python 3.12 or newer")
    try:
        import numpy as np
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise RuntimeError(
            "LIBERO v3 materialization requires: pip install 'tenon-pi-client[libero]'"
        ) from exc
    return np, pq, LeRobotDataset, Image


def _decode_image(cell: dict[str, Any], source_root: Path) -> Any:
    np, _, _, Image = _dependencies()
    data = cell.get("bytes")
    if data is None:
        relative = cell.get("path")
        if not isinstance(relative, str):
            raise ValueError("image cell has neither embedded bytes nor a path")
        data = (source_root / relative).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"decoded image isn't HWC RGB: {rgb.shape}")
    return rgb


def _update_content_digest(
    digest: Any,
    image: Any,
    wrist_image: Any,
    state: Any,
    action: Any,
) -> None:
    np, _, _, _ = _dependencies()
    for array, dtype in (
        (image, np.uint8),
        (wrist_image, np.uint8),
        (state, np.dtype("<f4")),
        (action, np.dtype("<f4")),
    ):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        digest.update(len(canonical.tobytes()).to_bytes(8, "little"))
        digest.update(canonical.tobytes())


def _tensor_to_rgb(value: Any) -> Any:
    np, _, _, _ = _dependencies()
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"v3 image isn't rank 3: {array.shape}")
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    return np.ascontiguousarray(array)


def _tensor_to_float32(value: Any) -> Any:
    np, _, _, _ = _dependencies()
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return np.ascontiguousarray(array, dtype=np.dtype("<f4"))


def _scalar(value: Any) -> int | float:
    if hasattr(value, "item"):
        return cast(int | float, value.item())
    return cast(int | float, value)


def _source_rows(source_file: Path, source_root: Path) -> list[dict[str, Any]]:
    np, pq, _, _ = _dependencies()
    table = pq.read_table(source_file)
    required = {
        "image",
        "wrist_image",
        "state",
        "actions",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"source episode missing columns: {sorted(missing)}")
    columns = {name: table[name].to_pylist() for name in required}
    rows: list[dict[str, Any]] = []
    for frame_index in range(table.num_rows):
        stored_frame = int(columns["frame_index"][frame_index])
        if stored_frame != frame_index:
            raise ValueError("source frame_index isn't contiguous")
        timestamp = float(columns["timestamp"][frame_index])
        expected_timestamp = np.float32(frame_index / 10).item()
        if np.float32(timestamp).tobytes() != np.float32(expected_timestamp).tobytes():
            raise ValueError("source timestamp isn't the expected synthetic frame_index / 10")
        rows.append(
            {
                "image": _decode_image(columns["image"][frame_index], source_root),
                "wrist_image": _decode_image(columns["wrist_image"][frame_index], source_root),
                "state": np.asarray(columns["state"][frame_index], dtype=np.float32),
                "action": np.asarray(columns["actions"][frame_index], dtype=np.float32),
                "timestamp": timestamp,
                "frame_index": frame_index,
            }
        )
    return rows


def materialize_lerobot_v3_subset(
    source_root: str | Path,
    output_root: str | Path,
    manifest: AcquisitionManifest,
) -> Path:
    """Create a valid LeRobot v3 dataset through LeRobot's public writer API.

    Decoded RGB, state/action row values, within-episode order, and the synthetic
    10 fps timebase are verified before and after materialization. No row is
    inserted, removed, averaged, or resampled.
    """

    np, _, LeRobotDataset, _ = _dependencies()
    source_path = Path(source_root).expanduser().resolve(strict=True)
    output_path = Path(output_root).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output directory must not already exist: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info_entry = next(file for file in manifest.files if file.relative_path == "meta/info.json")
    verify_source_file(source_path / info_entry.relative_path, info_entry)
    source_info = json.loads((source_path / info_entry.relative_path).read_text(encoding="utf-8"))
    source_features = source_info["features"]
    image_shape = tuple(source_features["image"]["shape"])
    wrist_shape = tuple(source_features["wrist_image"]["shape"])
    state_shape = tuple(source_features["state"]["shape"])
    action_shape = tuple(source_features["actions"]["shape"])
    features = {
        _IMAGE_KEY: {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        _WRIST_KEY: {
            "dtype": "image",
            "shape": wrist_shape,
            "names": ["height", "width", "channels"],
        },
        _STATE_KEY: {"dtype": "float32", "shape": state_shape, "names": ["state"]},
        _ACTION_KEY: {"dtype": "float32", "shape": action_shape, "names": ["actions"]},
    }

    dataset = LeRobotDataset.create(
        repo_id=_REPO_ID,
        root=output_path,
        fps=10,
        robot_type="panda",
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )
    episode_sources = [file for file in manifest.files if file.role == "episode"]
    source_digests: list[str] = []
    try:
        for source in episode_sources:
            if source.source_episode_index is None or source.frames is None or source.task is None:
                raise ValueError("episode source lacks index/frame/task provenance")
            source_file = source_path / source.relative_path
            verify_source_file(source_file, source)
            rows = _source_rows(source_file, source_path)
            if len(rows) != source.frames:
                raise ValueError(f"frame count mismatch for {source.relative_path}")
            digest = hashlib.sha256()
            for row in rows:
                _update_content_digest(
                    digest,
                    row["image"],
                    row["wrist_image"],
                    row["state"],
                    row["action"],
                )
                dataset.add_frame(
                    {
                        _IMAGE_KEY: row["image"],
                        _WRIST_KEY: row["wrist_image"],
                        _STATE_KEY: row["state"],
                        _ACTION_KEY: row["action"],
                        "task": source.task,
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            source_digests.append(digest.hexdigest())
        dataset.finalize()
    except Exception:
        # Ensure writer resources terminate before surfacing a failed artifact.
        dataset.finalize()
        raise

    converted = LeRobotDataset(repo_id=_REPO_ID, root=output_path)
    global_index = 0
    provenance_episodes: list[dict[str, Any]] = []
    for output_episode_index, (source, expected_digest) in enumerate(
        zip(episode_sources, source_digests, strict=True)
    ):
        expected_rows = _source_rows(source_path / source.relative_path, source_path)
        output_digest = hashlib.sha256()
        for local_frame_index, expected in enumerate(expected_rows):
            item = converted[global_index]
            if int(_scalar(item["episode_index"])) != output_episode_index:
                raise ValueError("v3 episode_index isn't contiguous")
            if int(_scalar(item["frame_index"])) != local_frame_index:
                raise ValueError("v3 frame_index isn't contiguous")
            if int(_scalar(item["index"])) != global_index:
                raise ValueError("v3 global index isn't contiguous")
            timestamp = np.float32(_scalar(item["timestamp"]))
            if timestamp.tobytes() != np.float32(expected["timestamp"]).tobytes():
                raise ValueError("v3 timestamp changed")
            image = _tensor_to_rgb(item[_IMAGE_KEY])
            wrist = _tensor_to_rgb(item[_WRIST_KEY])
            state = _tensor_to_float32(item[_STATE_KEY])
            action = _tensor_to_float32(item[_ACTION_KEY])
            if not np.array_equal(image, expected["image"]):
                raise ValueError("decoded base RGB changed")
            if not np.array_equal(wrist, expected["wrist_image"]):
                raise ValueError("decoded wrist RGB changed")
            if not np.array_equal(state, expected["state"]):
                raise ValueError("state row changed")
            if not np.array_equal(action, expected["action"]):
                raise ValueError("action row changed")
            _update_content_digest(output_digest, image, wrist, state, action)
            global_index += 1
        if output_digest.hexdigest() != expected_digest:
            raise ValueError("episode content digest changed")
        provenance_episodes.append(
            {
                "source_episode_index": source.source_episode_index,
                "output_episode_index": output_episode_index,
                "source_file_sha256": source.sha256,
                "decoded_content_sha256": expected_digest,
                "frames": source.frames,
                "task": source.task,
            }
        )
    if global_index != manifest.total_frames or global_index != len(converted):
        raise ValueError("materialized frame total doesn't match the acquisition manifest")

    provenance = {
        "manifest_version": "tenon-libero10-mini-v3-v1",
        "source_repo_id": manifest.source_repo_id,
        "source_revision": manifest.source_revision,
        "source_license": manifest.source_license,
        "source_format": "lerobot_v2",
        "output_format": "lerobot_v3",
        "lerobot_version": "0.6.1",
        "acquisition_manifest_sha256": manifest.canonical_manifest_sha256,
        "implicit_resampling": False,
        "stored_fps": 10,
        "source_temporal_provenance": manifest.temporal_provenance,
        "preserved_values": [
            "decoded base RGB",
            "decoded wrist RGB",
            "state",
            "action",
            "timestamp",
            "within-episode row order",
        ],
        "episodes": provenance_episodes,
    }
    (output_path / "meta" / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "README.md").write_text(
        "# Tenon LIBERO-10 mini v3\n\n"
        "Losslessly materialized through LeRobot 0.6.1 from "
        f"`{manifest.source_repo_id}@{manifest.source_revision}`. "
        "One retained episode per LIBERO-10 task; no temporal resampling.\n\n"
        "License: CC-BY-4.0. Cite LIBERO: Liu et al., arXiv:2306.03310.\n",
        encoding="utf-8",
    )
    return output_path


def materialize_libero10_v3(source_root: str | Path, output_root: str | Path) -> Path:
    """Materialize the built-in exact ten-episode selection."""

    source_path = Path(source_root)
    manifest_path = source_path / "acquisition-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source directory is missing acquisition-manifest.json")
    observed = AcquisitionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    expected = libero10_acquisition_manifest()
    if observed.canonical_manifest_sha256 != expected.canonical_manifest_sha256:
        raise ValueError("acquisition manifest doesn't match the pinned LIBERO-10 selection")
    return materialize_lerobot_v3_subset(source_path, output_root, observed)
