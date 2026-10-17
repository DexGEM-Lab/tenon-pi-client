"""Tiny generated v3 source following the upstream 33cad370 metadata contract.

No generated media/parquet files are committed. ffmpeg makes real shared
H.264 streams with long GOPs; data and video rotate at different boundaries.
Task metadata mirrors pandas.DataFrame({'task_index': ...}, index=task_strings)
written by LeRobot's save_episode_tasks/write_tasks, without a pandas dependency.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tenon_pi.data_fixture import write_data_fixture
from tenon_pi.dataset_v3 import DATA_PATH, VIDEO_PATH
from tenon_pi.robot_profile import RobotProfile

LENGTHS = (7, 53, 5)
TASKS = ("pick the red cube", "stack the blue block")


def write_v3_fixture(root: Path) -> tuple[Path, dict]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("real shared-video fixture requires ffmpeg")
    paths = write_data_fixture(root / "calibration-source")
    profile = RobotProfile.load(paths["profile"])
    camera_ids = tuple(c.id for c in profile.cameras)
    source = root / "source"
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [4],
            "names": [c.name for c in profile.state.canonical],
        },
        "action": {
            "dtype": "float32",
            "shape": [4],
            "names": [c.name for c in profile.state.canonical],
        },
        **{f"observation.images.{c}": {"dtype": "video", "shape": [48, 64, 3]} for c in camera_ids},
        "timestamp": {"dtype": "float32", "shape": [1]},
        **{
            k: {"dtype": "int64", "shape": [1]}
            for k in ("frame_index", "episode_index", "index", "task_index")
        },
    }
    (source / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 20,
                "features": features,
                "total_episodes": len(LENGTHS),
                "total_frames": sum(LENGTHS),
                "total_tasks": len(TASKS),
                "data_path": DATA_PATH,
                "video_path": VIDEO_PATH,
            }
        )
    )
    (source / "meta/stats.json").write_text("{}\n")
    tasks = pa.table({"task_index": [0, 1], "__index_level_0__": list(TASKS)})
    tasks = tasks.replace_schema_metadata(
        {
            b"pandas": json.dumps(
                {
                    "index_columns": ["__index_level_0__"],
                    "column_indexes": [],
                    "columns": [
                        {
                            "name": "task_index",
                            "field_name": "task_index",
                            "pandas_type": "int64",
                            "numpy_type": "int64",
                            "metadata": None,
                        },
                        {
                            "name": None,
                            "field_name": "__index_level_0__",
                            "pandas_type": "unicode",
                            "numpy_type": "object",
                            "metadata": None,
                        },
                    ],
                    "creator": {"library": "pyarrow", "version": pa.__version__},
                    "pandas_version": "2.2.3",
                }
            ).encode()
        }
    )
    pq.write_table(tasks, source / "meta/tasks.parquet")
    metadata, rows = [], []
    cursor = 0
    for episode, length in enumerate(LENGTHS):
        entry = {
            "episode_index": episode,
            "length": length,
            "tasks": [TASKS[episode % 2]],
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + length,
            "data/chunk_index": 0,
            "data/file_index": 0 if episode < 2 else 1,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        # Data: ep0+ep1 / ep2. Video: ep0 / ep1+ep2 (off-keyframe ep2).
        for camera in camera_ids:
            prefix = f"videos/observation.images.{camera}"
            entry.update(
                {
                    f"{prefix}/chunk_index": 0,
                    f"{prefix}/file_index": 0 if episode == 0 else 1,
                    f"{prefix}/from_timestamp": 0.0 if episode < 2 else LENGTHS[1] / 20,
                    f"{prefix}/to_timestamp": (length if episode < 2 else sum(LENGTHS[1:])) / 20,
                }
            )
        metadata.append(entry)
        for frame in range(length):
            state = profile.state.to_canonical([-30 + frame / length, -20, frame / length * 100])
            action = profile.action.to_canonical(
                [-25, -15 + frame / length, 15 + frame / length * 70]
            )
            rows.append(
                {
                    "observation.state": state.tolist(),
                    "action": action.tolist(),
                    "timestamp": frame / 20,
                    "frame_index": frame,
                    "episode_index": episode,
                    "index": cursor + frame,
                    "task_index": episode % 2,
                }
            )
        cursor += length
    pq.write_table(
        pa.Table.from_pylist(metadata), source / "meta/episodes/chunk-000/file-000.parquet"
    )
    schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), 4)),
            pa.field("action", pa.list_(pa.float32(), 4)),
            pa.field("timestamp", pa.float32()),
            *[
                pa.field(k, pa.int64())
                for k in ("frame_index", "episode_index", "index", "task_index")
            ],
        ]
    )
    split = sum(LENGTHS[:2])
    for file, subset in enumerate((rows[:split], rows[split:])):
        path = source / DATA_PATH.format(chunk_index=0, file_index=file)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(subset, schema=schema), path, row_group_size=7)
    # Distinct camera encodings; no episode-cut assumption in shared files.
    for camera_index, camera in enumerate(camera_ids):
        for file, length in enumerate((LENGTHS[0], sum(LENGTHS[1:]))):
            path = source / VIDEO_PATH.format(
                video_key=f"observation.images.{camera}", chunk_index=0, file_index=file
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=0x{40 + camera_index * 60:02x}4060:s=64x48:r=20",
                    "-frames:v",
                    str(length),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-g",
                    "10000",
                    "-keyint_min",
                    "10000",
                    "-sc_threshold",
                    "0",
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
    return source, profile.descriptor()
