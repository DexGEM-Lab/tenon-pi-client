"""Synthetic, differently shaped preparation examples; never hardware calibration."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .preparation import _pyarrow, canonical_json_digest
from .robot_profile import RobotProfile


def example_profile(layout: Literal["planar", "gantry"] = "planar") -> RobotProfile:
    """Two abstract mechanisms: rotary two-link/gripper and translational xyz/gripper."""
    if layout == "planar":
        native = [
            dict(name=name, unit="degree", lower=-90, upper=90)
            for name in ("base_angle", "elbow_angle")
        ]
        native.append(dict(name="jaw_open", unit="open_percent", lower=0, upper=100))
        canonical = [
            dict(name="elbow", unit="radian", native_index=1, conversion="degrees_to_radians"),
            dict(name="base", unit="radian", native_index=0, conversion="degrees_to_radians"),
            dict(
                name="virtual_axis",
                unit="radian",
                native_index=None,
                conversion="constant",
                constant=0,
            ),
            dict(
                name="grip",
                unit="normalized_0_1",
                native_index=2,
                conversion="gripper_closedness",
                native_closed=0,
                native_open=100,
            ),
        ]
        hz, horizon, shapes, delta, width = 20, 4, [(64, 48)], [0, 1, None, None], 8
    elif layout == "gantry":
        native = [
            dict(name=name, unit="millimeter", lower=-100, upper=100) for name in ("x", "y", "z")
        ]
        native.append(dict(name="jaw", unit="millimeter", lower=0, upper=40))
        canonical = [
            dict(name=name, unit="meter", native_index=index, conversion="millimeters_to_meters")
            for name, index in (("z", 2), ("x", 0), ("y", 1))
        ]
        canonical.append(
            dict(
                name="grip",
                unit="normalized_0_1",
                native_index=3,
                conversion="gripper_closedness",
                native_closed=0,
                native_open=40,
            )
        )
        hz, horizon, shapes, delta, width = 10, 3, [(80, 60), (40, 32)], [0, 1, 2, None], 6
    else:
        raise ValueError("fixture layout must be planar or gantry")
    basis = {"synthetic": True, "layout": layout, "native": native}
    cameras = []
    for index, (w, h) in enumerate(shapes):
        cameras.append(
            dict(
                id=f"view_{index}",
                width=w,
                height=h,
                serial=f"synthetic-{index}",
                mount=f"synthetic-view-{index}",
                capture_clock="synthetic-monotonic",
                rotation_degrees=0,
                calibration_digest=canonical_json_digest(basis),
            )
        )
    state_native, state_canonical = list(native), list(canonical)
    if layout == "gantry":
        state_native.append(dict(name="load_sensor", unit="unitless", lower=-1, upper=1))
        state_canonical.append(
            dict(name="load", unit="unitless", native_index=4, conversion="identity")
        )
    return RobotProfile.model_validate(
        dict(
            profile_version="tenon-robot-profile-v1",
            profile_id=f"synthetic-{layout}-v1",
            calibration_digest=canonical_json_digest(basis),
            state=dict(native=state_native, canonical=state_canonical),
            action=dict(native=native, canonical=canonical),
            action_semantics="joint_position",
            controller_hz=hz,
            cameras=cameras,
            image_transform=dict(target_width=224, target_height=224, resize_mode="pad"),
            action_horizon_steps=horizon,
            max_committed_prefix_steps=1,
            model_state_width=width,
            model_action_width=width,
            model_delta_state_indices=delta,
        )
    )


def _png_bytes(episode: int, frame: int, camera: int, width: int, height: int) -> bytes:
    from PIL import Image

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (episode * 47 + frame * 11 + camera * 29) % 256
    image[:, :, 1] = np.arange(width, dtype=np.uint16)[None, :] % 256
    image[:, :, 2] = np.arange(height, dtype=np.uint16)[:, None] % 256
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def write_data_fixture(
    output: str | Path,
    *,
    layout: Literal["planar", "gantry"] = "planar",
    episodes: int = 2,
    frames_per_episode: int = 4,
    profile: RobotProfile | None = None,
) -> dict[str, str]:
    """Generate native v2.1 rows, pixels, metadata and an explicit synthetic profile."""
    if episodes < 1 or frames_per_episode < 2:
        raise ValueError("fixture requires at least one episode and two frames per episode")
    profile = profile or example_profile(layout)
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"fixture output must not already exist: {root}")
    dataset = root / "native-v2"
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta").mkdir(parents=True)
    pa, pq = _pyarrow()
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    features: dict[str, Any] = {
        key: dict(
            dtype="float32", shape=[len(mapping.native)], names=[c.name for c in mapping.native]
        )
        for key, mapping in (("observation.state", profile.state), ("action", profile.action))
    }
    features.update(
        {
            "timestamp": {"dtype": "float64", "shape": [1]},
            **{
                k: {"dtype": "int64", "shape": [1]}
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
    episode_rows = []
    for episode in range(episodes):
        columns: dict[str, Any] = {}
        for key, mapping in (("observation.state", profile.state), ("action", profile.action)):
            lower, upper = mapping.bounds("native")
            values = []
            for frame in range(frames_per_episode):
                fraction = (
                    0.2
                    + 0.5
                    * (
                        (
                            episode * 7
                            + frame * 3
                            + np.arange(len(lower))
                            + (2 if key == "action" else 0)
                        )
                        % 11
                    )
                    / 10
                )
                values.append(np.asarray(lower + fraction * (upper - lower), dtype=np.float32))
            columns[key] = pa.array(
                np.stack(values).tolist(), type=pa.list_(pa.float32(), len(lower))
            )
        columns.update(
            {
                "timestamp": pa.array(
                    np.arange(frames_per_episode) / profile.controller_hz, type=pa.float64()
                ),
                "frame_index": pa.array(range(frames_per_episode), type=pa.int64()),
                "episode_index": pa.array([episode] * frames_per_episode, type=pa.int64()),
                "index": pa.array(
                    range(episode * frames_per_episode, (episode + 1) * frames_per_episode),
                    type=pa.int64(),
                ),
                "task_index": pa.array([0] * frames_per_episode, type=pa.int64()),
            }
        )
        for index, camera in enumerate(profile.cameras):
            columns[f"observation.images.{camera.id}"] = pa.array(
                [
                    {
                        "bytes": _png_bytes(episode, frame, index, camera.width, camera.height),
                        "path": None,
                    }
                    for frame in range(frames_per_episode)
                ],
                type=image_type,
            )
        from .dataset_metadata import arrow_feature_metadata

        table = pa.table(columns).replace_schema_metadata(arrow_feature_metadata(features))
        pq.write_table(
            table, dataset / f"data/chunk-000/episode_{episode:06d}.parquet", compression="snappy"
        )
        episode_rows.append(
            dict(
                episode_index=episode,
                tasks=["synthetic preparation example"],
                length=frames_per_episode,
            )
        )
    info = dict(
        codebase_version="v2.1",
        robot_type=f"synthetic-{layout}",
        total_episodes=episodes,
        total_frames=episodes * frames_per_episode,
        total_tasks=1,
        total_videos=0,
        total_chunks=1,
        chunks_size=1000,
        fps=profile.controller_hz,
        splits={"train": f"0:{episodes}"},
        features=features,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    (dataset / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (dataset / "meta/tasks.jsonl").write_text(
        json.dumps(dict(task_index=0, task="synthetic preparation example")) + "\n"
    )
    (dataset / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows)
    )
    from .dataset_metadata import write_v2_metadata_statistics

    write_v2_metadata_statistics(dataset, info)
    profile_path = root / "profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2) + "\n")
    return {"root": str(root), "dataset": str(dataset), "profile": str(profile_path)}
