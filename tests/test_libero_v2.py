from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tenon_pi.datasets import validate_lerobot_root
from tenon_pi.libero import AcquisitionManifest, SourceFile
from tenon_pi.libero_v2 import materialize_lerobot_v2_subset


def _png(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (2, 2), color)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_v2_materializer_preserves_content_and_renumbers(tmp_path: Path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
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
    info_path = source / "meta" / "info.json"
    info_path.write_text(json.dumps(info))
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    table = pa.table(
        {
            "image": pa.array([{"bytes": _png((1, 2, 3)), "path": None}], type=image_type),
            "wrist_image": pa.array([{"bytes": _png((4, 5, 6)), "path": None}], type=image_type),
            "state": pa.array([[float(i) for i in range(8)]], type=pa.list_(pa.float32(), 8)),
            "actions": pa.array([[float(i) for i in range(7)]], type=pa.list_(pa.float32(), 7)),
            "timestamp": pa.array([0.0], type=pa.float32()),
            "frame_index": pa.array([0], type=pa.int64()),
            "episode_index": pa.array([27], type=pa.int64()),
            "index": pa.array([1234], type=pa.int64()),
            "task_index": pa.array([9], type=pa.int64()),
        }
    )
    episode_path = source / "data" / "chunk-000" / "episode_000027.parquet"
    pq.write_table(table, episode_path)
    files = [
        SourceFile(
            relative_path="meta/info.json",
            size_bytes=info_path.stat().st_size,
            sha256=_sha(info_path),
            role="upstream_metadata",
        ),
        SourceFile(
            relative_path="data/chunk-000/episode_000027.parquet",
            size_bytes=episode_path.stat().st_size,
            sha256=_sha(episode_path),
            role="episode",
            source_episode_index=27,
            task_index=9,
            frames=1,
            task="fixture task",
        ),
    ]
    manifest = AcquisitionManifest(
        manifest_version="libero10-mini-acquisition-v1",
        source_repo_id="fixture/repo",
        source_revision="fixture",
        source_license="CC-BY-4.0",
        source_format="lerobot_v2",
        selected_suite="libero_10",
        selected_episode_indices=[27],
        total_episode_payload_bytes=episode_path.stat().st_size,
        total_frames=1,
        temporal_provenance={"stored_fps": "10"},
        conversion_contract={"implicit_resampling": "forbidden"},
        upload_contract={"source_is_upload_ready": False},
        attribution=["fixture"],
        files=files,
        canonical_manifest_sha256="f" * 64,
    )
    output = tmp_path / "output"
    materialize_lerobot_v2_subset(source, output, manifest)
    converted = pq.read_table(output / "data" / "chunk-000" / "episode_000000.parquet")
    for name in ["image", "wrist_image", "state", "actions", "timestamp", "frame_index"]:
        assert table[name].equals(converted[name]), name
    assert converted["episode_index"].to_pylist() == [0]
    assert converted["task_index"].to_pylist() == [0]
    assert converted["index"].to_pylist() == [0]
    assert validate_lerobot_root(output, "lerobot_v2")["total_frames"] == 1
    provenance = json.loads((output / "meta" / "source_provenance.json").read_text())
    assert provenance["implicit_resampling"] is False
    assert provenance["episodes"][0]["source_episode_index"] == 27
