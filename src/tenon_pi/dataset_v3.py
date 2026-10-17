"""Direct, data-only LeRobot v3.0 source validation for Robot preparation.

Metadata definitions: huggingface/lerobot at
33cad37054c2b594ceba57463e8f11ee374fa93c, src/lerobot/datasets/{utils,
lerobot_dataset}.py. Tasks are a pandas string index; episodes locate consolidated
row files and independent per-camera video segments. No reader/view is generated.
Video bytes are bound, not decoded: container/decoded-frame validation remains a
hosted ingestion responsibility. Statistics need only the original numeric rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .preparation import Episode

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_METADATA_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_TOLERANCE_S = 1e-4


def load_v3_episodes(root: Path, descriptor: dict[str, Any]) -> list[Episode]:
    """Read original rows in global order, honoring explicit episode ownership.

    The supported writer layout keeps each episode in one data file; several
    episodes may share that file. Video file boundaries are independent of data
    boundaries, and a video segment need not begin on a keyframe.
    """
    from .preparation import (
        Episode,
        PreparationError,
        _json_object,
        _profile,
        _pyarrow,
        _validate_vectors,
    )

    profile = _profile(descriptor)

    def require(condition: bool | np.bool_, message: str) -> None:
        if not condition:
            raise PreparationError(message)

    def integer(value: object) -> bool:
        return type(value) is int and value >= 0

    def location(episode: dict[str, Any], prefix: str, template: str, **kwargs: str) -> str:
        chunk, file = episode.get(f"{prefix}/chunk_index"), episode.get(f"{prefix}/file_index")
        require(integer(chunk) and integer(file), f"v3 episode lacks valid {prefix} location")
        return template.format(chunk_index=chunk, file_index=file, **kwargs)

    def existing(relative: str) -> Path:
        path = root / relative
        require(
            path.is_file() and path.stat().st_size > 0, f"missing/empty v3 source file: {relative}"
        )
        return path

    info = _json_object(root / "meta/info.json", "LeRobot meta/info.json")
    require(
        info.get("codebase_version") == "v3.0", "Robot v3 supports codebase_version='v3.0' only"
    )
    require("tenon_view" not in info, "a derived view is not a source dataset")
    require(
        not any(root.rglob("*view-manifest.json")),
        "derived view artifacts must never be uploaded",
    )
    require(
        info.get("data_path") == DATA_PATH and info.get("video_path") == VIDEO_PATH,
        "Robot v3 requires default data_path/video_path templates",
    )
    require(
        type(info.get("fps")) in (int, float) and info["fps"] == profile.controller_hz,
        "Robot data must declare the profile fps; no implicit resampling",
    )
    for key in ("total_episodes", "total_frames", "total_tasks"):
        require(integer(info.get(key)) and info[key] > 0, f"v3 {key} must be a positive integer")
    features = info.get("features")
    if not isinstance(features, dict):
        raise PreparationError("v3 features must be an object")
    require(
        all(isinstance(v, dict) and "shape" in v and "dtype" in v for v in features.values()),
        "v3 features must declare dtype and shape",
    )
    for key, mapping in (("observation.state", profile.state), ("action", profile.action)):
        feature = features.get(key, {})
        require(
            feature.get("shape") == [len(mapping.canonical)]
            and feature.get("dtype") == "float32"
            and feature.get("names") == [c.name for c in mapping.canonical],
            f"v3 {key} requires canonical float32; native v3 is unsupported",
        )
    require(
        all(
            key in features
            for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
        ),
        "v3 scalar row features are required",
    )
    camera_keys = {f"observation.images.{name}" for name in (c.id for c in profile.cameras)}
    visual_keys = {
        k
        for k, v in features.items()
        if isinstance(v, dict) and v.get("dtype") in {"image", "video"}
    }
    require(visual_keys == camera_keys, "Robot v3 requires exactly the profile video cameras")
    for camera in profile.cameras:
        key = f"observation.images.{camera.id}"
        require(
            not features[key].get("info", {}).get("is_depth_map", False),
            "Robot v3 requires RGB, not depth video",
        )
        require(
            features[key].get("dtype") == "video"
            and features[key].get("shape") == [camera.height, camera.width, 3],
            "Robot v3 is video-only with profile RGB shapes; image variants unsupported",
        )

    pa, pq = _pyarrow()
    tasks_table = pq.read_table(existing("meta/tasks.parquet"))
    task_column = "task" if "task" in tasks_table.column_names else None
    if task_column is None:
        # Upstream save_episode_tasks creates an unnamed pandas string index;
        # Arrow persists its physical column name in schema metadata.
        pandas = json.loads((tasks_table.schema.metadata or {}).get(b"pandas", b"{}"))
        indices = pandas.get("index_columns", [])
        if len(indices) == 1 and isinstance(indices[0], str):
            task_column = indices[0]
        elif "__index_level_0__" in tasks_table.column_names:
            task_column = "__index_level_0__"
    require(
        task_column in tasks_table.column_names and "task_index" in tasks_table.column_names,
        "v3 tasks.parquet requires task_index and a task string index",
    )
    tasks: dict[int, str] = {}
    for row in tasks_table.to_pylist():
        index, text = row["task_index"], row[task_column]
        require(
            integer(index)
            and index not in tasks
            and isinstance(text, str)
            and bool(text)
            and text not in tasks.values(),
            "v3 tasks must be a bijective nonempty string mapping",
        )
        tasks[index] = text
    require(len(tasks) == info["total_tasks"], "v3 task total disagrees with tasks.parquet")

    metadata_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    require(bool(metadata_files), "v3 requires meta/episodes parquet metadata")
    metadata = []
    for path in metadata_files:
        require(
            re.fullmatch(
                r"meta/episodes/chunk-\d{3,}/file-\d{3,}\.parquet",
                path.relative_to(root).as_posix(),
            )
            is not None,
            "v3 requires default episode metadata paths",
        )
        table = pq.read_table(path)
        for episode in table.to_pylist():
            require(integer(episode.get("episode_index")), "invalid v3 episode_index")
            if "meta/episodes/chunk_index" in episode or "meta/episodes/file_index" in episode:
                require(
                    location(episode, "meta/episodes", _METADATA_PATH)
                    == path.relative_to(root).as_posix(),
                    "v3 episode metadata file ownership mismatch",
                )
            metadata.append(episode)
    metadata.sort(key=lambda e: e["episode_index"])
    require(
        [e["episode_index"] for e in metadata] == list(range(info["total_episodes"])),
        "v3 episode metadata must be exactly contiguous 0..N-1",
    )
    cursor = 0
    data_locations: list[str] = []
    video_segments: dict[str, list[tuple[float, float]]] = {}
    for episode in metadata:
        length = episode.get("length")
        require(integer(length) and length > 0, "v3 episode length must be positive")
        require(
            integer(episode.get("dataset_from_index"))
            and integer(episode.get("dataset_to_index"))
            and episode["dataset_from_index"] == cursor
            and episode["dataset_to_index"] == cursor + length,
            "v3 episode dataset index ranges must be consecutive and match length",
        )
        cursor += length
        episode_tasks = episode.get("tasks")
        require(
            isinstance(episode_tasks, list)
            and bool(episode_tasks)
            and all(isinstance(t, str) and t in tasks.values() for t in episode_tasks),
            "v3 episode tasks absent from tasks.parquet",
        )
        data_path = location(episode, "data", DATA_PATH)
        if data_path not in data_locations:
            data_locations.append(data_path)
        for key in sorted(camera_keys):
            prefix = f"videos/{key}"
            video = location(episode, prefix, VIDEO_PATH, video_key=key)
            existing(video)
            start, stop = (
                episode.get(f"{prefix}/from_timestamp"),
                episode.get(f"{prefix}/to_timestamp"),
            )
            require(
                all(type(v) in (float, int) and np.isfinite(v) and v >= 0 for v in (start, stop)),
                "v3 video segment timestamps must be finite and nonnegative",
            )
            require(
                abs(stop - start - length / profile.controller_hz) <= _TOLERANCE_S,
                "v3 video segment duration disagrees with episode length/fps",
            )
            segments = video_segments.setdefault(video, [])
            require(
                abs(start - (segments[-1][1] if segments else 0)) <= _TOLERANCE_S,
                "v3 video segments overlap or leave gaps",
            )
            segments.append((start, stop))
    require(cursor == info["total_frames"], "v3 episode lengths disagree with total_frames")
    require(data_locations == sorted(data_locations), "v3 episodes must follow data file order")
    require(
        {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.parquet")}
        == set(data_locations),
        "v3 data file set disagrees with episode metadata",
    )
    require(
        {p.relative_to(root).as_posix() for p in (root / "videos").rglob("*.mp4")}
        == set(video_segments),
        "v3 video file set disagrees with episode metadata",
    )
    _json_object(existing("meta/stats.json"), "LeRobot meta/stats.json")

    episodes = []
    next_episode = cursor = 0
    for relative in data_locations:
        table = pq.read_table(existing(relative))
        for key in features:
            if key in camera_keys:
                require(
                    key not in table.column_names,
                    "v3 video features must not be embedded row columns",
                )
                continue
            require(key in table.column_names, f"{relative} lacks declared column {key}")
        for key in ("observation.state", "action"):
            dtype = table.schema.field(key).type
            require(
                (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype))
                and dtype.value_type == pa.float32(),
                f"{relative}: {key} must store float32 vectors",
            )
        for key in ("index", "episode_index", "frame_index", "task_index", "timestamp"):
            expected_dtype = "float32" if key == "timestamp" else "int64"
            require(
                features.get(key, {}).get("dtype") == expected_dtype
                and features.get(key, {}).get("shape") == [1]
                and table.schema.field(key).type == getattr(pa, expected_dtype)(),
                f"v3 {key} must declare shape [1] and store {expected_dtype}",
            )
            require(table[key].null_count == 0, f"v3 {key} contains nulls")
        require(table.num_rows > 0, "empty v3 data parquet")
        indices = np.asarray(table["index"].to_numpy())
        require(
            np.array_equal(indices, np.arange(cursor, cursor + table.num_rows)),
            "v3 global row index must be consecutive in file order",
        )
        local = 0
        while local < table.num_rows:
            require(next_episode < len(metadata), "extra v3 rows outside episode metadata")
            episode = metadata[next_episode]
            length = episode["length"]
            require(
                location(episode, "data", DATA_PATH) == relative
                and local + length <= table.num_rows,
                "v3 rows disagree with episode data file ownership",
            )
            rows = table.slice(local, length)
            require(
                np.array_equal(rows["episode_index"].to_numpy(), np.full(length, next_episode)),
                "v3 row episode_index disagrees with metadata ownership",
            )
            require(
                np.array_equal(rows["frame_index"].to_numpy(), np.arange(length)),
                "v3 frame_index must be consecutive from zero per episode",
            )
            timestamps = np.asarray(rows["timestamp"].to_numpy(), dtype=np.float64)
            require(
                np.isfinite(timestamps).all()
                and np.all(
                    np.abs(timestamps - np.arange(length) / profile.controller_hz) <= _TOLERANCE_S
                ),
                "v3 timestamps must match frame_index at profile rate; never resample",
            )
            require(
                all(
                    t in tasks and tasks[t] in episode["tasks"]
                    for t in rows["task_index"].to_pylist()
                ),
                "v3 row task is not owned by episode",
            )
            states = np.asarray(rows["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(rows["action"].to_pylist(), dtype=np.float32)
            for key, values in (("observation.state", states), ("action", actions)):
                _validate_vectors(values, space="canonical", descriptor=descriptor, label=key)
            episodes.append(Episode(next_episode, states, actions, timestamps))
            next_episode += 1
            local += length
        cursor += table.num_rows
    require(
        next_episode == len(metadata) and cursor == info["total_frames"],
        "v3 rows must exactly cover episode metadata",
    )
    return episodes
