from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

pytest.importorskip("lerobot")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from tenon_pi.datasets import validate_lerobot_v3_root
from tenon_pi.libero import AcquisitionManifest, SourceFile
from tenon_pi.libero_v3 import materialize_lerobot_v3_subset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (2, 2), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _episode_table(episode_index: int, task_index: int, global_start: int) -> pa.Table:
    rows = 2
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    state_type = pa.list_(pa.float32(), 8)
    action_type = pa.list_(pa.float32(), 7)
    return pa.table(
        {
            "image": pa.array(
                [
                    {"bytes": _png((episode_index + 1, 2, 3)), "path": None},
                    {"bytes": _png((4, episode_index + 1, 6)), "path": None},
                ],
                type=image_type,
            ),
            "wrist_image": pa.array(
                [
                    {"bytes": _png((7, 8, episode_index + 1)), "path": None},
                    {"bytes": _png((episode_index + 2, 9, 10)), "path": None},
                ],
                type=image_type,
            ),
            "state": pa.array(
                [[float(episode_index + i + j) for j in range(8)] for i in range(rows)],
                type=state_type,
            ),
            "actions": pa.array(
                [[float(task_index + i + j) for j in range(7)] for i in range(rows)],
                type=action_type,
            ),
            "timestamp": pa.array([0.0, 0.1], type=pa.float32()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([episode_index] * rows, type=pa.int64()),
            "index": pa.array([global_start, global_start + 1], type=pa.int64()),
            "task_index": pa.array([task_index] * rows, type=pa.int64()),
        }
    )


def _source_fixture(root: Path) -> AcquisitionManifest:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "fps": 10,
        "features": {
            "image": {
                "dtype": "image",
                "shape": [2, 2, 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [2, 2, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": [8], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
        },
    }
    info_path = root / "meta" / "info.json"
    info_path.write_text(json.dumps(info))
    entries = [
        SourceFile(
            relative_path="meta/info.json",
            size_bytes=info_path.stat().st_size,
            sha256=_sha(info_path),
            role="upstream_metadata",
        )
    ]
    global_start = 100
    for source_episode, task_index in [(10, 5), (22, 8)]:
        path = root / "data" / "chunk-000" / f"episode_{source_episode:06d}.parquet"
        pq.write_table(_episode_table(source_episode, task_index, global_start), path)
        entries.append(
            SourceFile(
                relative_path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha(path),
                role="episode",
                source_episode_index=source_episode,
                task_index=task_index,
                frames=2,
                task=f"fixture task {task_index}",
            )
        )
        global_start += 2
    return AcquisitionManifest(
        manifest_version="libero10-mini-acquisition-v1",
        source_repo_id="fixture/repo",
        source_revision="fixture-revision",
        source_license="CC-BY-4.0",
        source_format="lerobot_v2",
        selected_suite="libero_10",
        selected_episode_indices=[10, 22],
        total_episode_payload_bytes=sum(entry.size_bytes for entry in entries[1:]),
        total_frames=4,
        temporal_provenance={"stored_fps": "10"},
        conversion_contract={"implicit_resampling": "forbidden"},
        upload_contract={"source_is_upload_ready": False},
        attribution=["fixture"],
        files=entries,
        canonical_manifest_sha256="f" * 64,
    )


def test_public_lerobot_api_materializes_lossless_v3(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    manifest = _source_fixture(source_root)
    materialize_lerobot_v3_subset(source_root, output_root, manifest)

    info = validate_lerobot_v3_root(output_root)
    assert str(info["codebase_version"]).startswith("v3.")
    converted = LeRobotDataset(repo_id="tenon/libero10-mini-v3", root=output_root)
    assert len(converted) == 4
    assert int(converted[0]["episode_index"].item()) == 0
    assert int(converted[2]["episode_index"].item()) == 1
    provenance = json.loads((output_root / "meta" / "source_provenance.json").read_text())
    assert provenance["implicit_resampling"] is False
    assert provenance["source_format"] == "lerobot_v2"
    assert provenance["output_format"] == "lerobot_v3"
    assert [row["source_episode_index"] for row in provenance["episodes"]] == [10, 22]
