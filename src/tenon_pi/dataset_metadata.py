"""LeRobot v2 metadata statistics, separate from model-space horizon normalization.

The v2.1 sampling, shapes and population-moment aggregation follow LeRobot
0cf864870cf29f4738d3ade893e6fd13fbd7cdb5, common/datasets/compute_stats.py
(Apache-2.0). Only NumPy, Arrow and Pillow are needed. v2.0 uses global metadata;
v2.1 uses per-episode metadata and derives its aggregate in the reader.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import LEROBOT_V2_STATISTICS_FILES


def required_statistics_file(version: object) -> str:
    from .preparation import PreparationError

    if not isinstance(version, str) or version not in LEROBOT_V2_STATISTICS_FILES:
        raise PreparationError(f"unsupported Robot v2 revision {version!r}; v2.0/v2.1 only")
    return LEROBOT_V2_STATISTICS_FILES[version]


def arrow_feature_metadata(features: dict[str, Any]) -> dict[bytes, bytes]:
    """Embed the pinned HuggingFace image/vector type tags without importing it."""
    encoded: dict[str, Any] = {}
    for name, feature in features.items():
        dtype = feature["dtype"]
        if dtype == "image":
            encoded[name] = {"_type": "Image"}
        elif feature["shape"] == [1] and name not in {"observation.state", "action"}:
            encoded[name] = {"dtype": dtype, "_type": "Value"}
        else:
            encoded[name] = {
                "feature": {"dtype": dtype, "_type": "Value"},
                "length": feature["shape"][0],
                "_type": "Sequence",
            }
    return {b"huggingface": json.dumps({"info": {"features": encoded}}).encode()}


def validate_arrow_features(table: Any, info: dict[str, Any]) -> None:
    from .preparation import PreparationError, _pyarrow

    pa, _ = _pyarrow()
    try:
        features = json.loads((table.schema.metadata or {})[b"huggingface"])["info"]["features"]
        for name, feature in info["features"].items():
            encoded = features[name]
            if feature["dtype"] == "image":
                valid = encoded.get("_type") == "Image" and encoded.get("decode", True)
            elif feature["shape"] == [1] and name not in {"observation.state", "action"}:
                valid = encoded.get("_type") == "Value" and encoded.get("dtype") == feature["dtype"]
            else:
                valid = (
                    encoded.get("_type") == "Sequence"
                    and encoded.get("length", -1) in (-1, feature["shape"][0])
                    and encoded["feature"].get("dtype") == feature["dtype"]
                )
            physical = table.schema.field(name).type
            if feature["dtype"] not in {"image", "string"}:
                shape = feature["shape"]
                if (
                    not isinstance(shape, list)
                    or len(shape) != 1
                    or type(shape[0]) is not int
                    or shape[0] < 1
                ):
                    raise ValueError(f"{name}: numeric features require scalar [1] or vector [N]")
                leaf = pa.from_numpy_dtype(np.dtype(feature["dtype"]))
                if feature["shape"] == [1] and name not in {"observation.state", "action"}:
                    valid = valid and physical == leaf
                else:
                    valid = (
                        valid
                        and (pa.types.is_list(physical) or pa.types.is_fixed_size_list(physical))
                        and physical.value_type == leaf
                        and (
                            not pa.types.is_fixed_size_list(physical)
                            or physical.list_size == feature["shape"][0]
                        )
                    )
            if not valid:
                raise ValueError(name)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise PreparationError(
            "parquet requires matching HuggingFace Image/Sequence/Value metadata; "
            "regenerate with a supported writer"
        ) from exc


def canonical_arrow_vectors(table: Any, widths: dict[str, int]) -> Any:
    """Update only converted vector widths in an existing HF feature declaration."""
    metadata = dict(table.schema.metadata or {})
    value = json.loads(metadata[b"huggingface"])
    for name in ("observation.state", "action"):
        value["info"]["features"][name]["length"] = widths[name]
    metadata[b"huggingface"] = json.dumps(value).encode()
    return table.replace_schema_metadata(metadata)


def _moments(values: np.ndarray, *, image: bool = False) -> dict[str, Any]:
    axes = (0, 2, 3) if image else 0
    keepdims = image or values.ndim == 1
    reduction: dict[str, Any] = {"axis": axes, "keepdims": keepdims}
    result = {
        "min": np.min(values, **reduction),
        "max": np.max(values, **reduction),
        "mean": np.mean(values, **reduction),
        "std": np.std(values, **reduction),
    }
    if image:
        result = {key: np.squeeze(value / 255.0, axis=0) for key, value in result.items()}
    return {**{key: value.tolist() for key, value in result.items()}, "count": [len(values)]}


def _sampled_images(root: Path, cells: list[Any], shape: list[int]) -> np.ndarray:
    from PIL import Image

    from .preparation import PreparationError

    length = len(cells)
    count = max(min(length, 100), min(int(length**0.75), 10_000))
    indices = np.round(np.linspace(0, length - 1, count)).astype(int)
    images = []
    for index in indices:
        cell = cells[index]
        if not isinstance(cell, dict):
            raise PreparationError("image statistics require a LeRobot image cell")
        encoded = cell.get("bytes")
        if encoded is None:
            relative = cell.get("path")
            if not isinstance(relative, str):
                raise PreparationError("image statistics require bytes or a relative path")
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise PreparationError("image statistics path escapes/misses the dataset")
            encoded = path.read_bytes()
        with Image.open(io.BytesIO(encoded)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (shape[1], shape[0]):
                raise PreparationError(
                    "image statistics require original RGB PNG with declared shape"
                )
            chw = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
            # Pinned writer targets a maximum side of 150 by integer striding.
            stride = max(1, int(max(image.size) / 150))
            images.append(chw[:, ::stride, ::stride])
    # The pinned sample_images preallocates a C-contiguous NCHW uint8 array.
    # Match that reduction layout as well as its sampled pixels.
    return np.ascontiguousarray(np.stack(images))


def compute_episode_metadata(root: Path, info: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute every declared numeric/image feature from actual stored source rows."""
    from .preparation import PreparationError, _parquet_files, _pyarrow

    _, pq = _pyarrow()
    result = []
    for episode_index, path in enumerate(_parquet_files(root)):
        table = pq.read_table(path)
        validate_arrow_features(table, info)
        stats = {}
        for key, feature in info["features"].items():
            dtype = feature["dtype"]
            if dtype == "string":
                continue
            if key not in table.column_names:
                raise PreparationError(f"metadata feature absent from rows: {key}")
            if dtype == "image":
                stats[key] = _moments(
                    _sampled_images(root, table[key].to_pylist(), feature["shape"]), image=True
                )
            else:
                width = feature["shape"][0]
                shape = (
                    (table.num_rows,)
                    if width == 1 and key not in {"observation.state", "action"}
                    else (table.num_rows, width)
                )
                try:
                    # Arrow leaf type was checked above. Do not reshape ragged,
                    # nested or incorrectly sized rows to fit a declaration.
                    values = np.asarray(table[key].to_pylist(), dtype=np.dtype(dtype))
                except (TypeError, ValueError) as exc:
                    raise PreparationError(
                        f"numeric feature {key} rows disagree with declared shape "
                        f"{feature['shape']}"
                    ) from exc
                if values.shape != shape:
                    raise PreparationError(
                        f"numeric feature {key} rows disagree with declared shape "
                        f"{feature['shape']}"
                    )
                if not np.isfinite(values).all():
                    raise PreparationError(f"metadata requires finite numeric feature {key}")
                stats[key] = _moments(values)
        result.append({"episode_index": episode_index, "stats": stats})
    return result


def aggregate_metadata(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """The pinned parallel population-variance aggregation, including frame counts."""
    result = {}
    for key in episodes[0]["stats"]:
        summaries = [episode["stats"][key] for episode in episodes]
        means = np.stack([np.asarray(summary["mean"]) for summary in summaries])
        variances = np.stack([np.asarray(summary["std"]) ** 2 for summary in summaries])
        counts = np.stack([np.asarray(summary["count"]) for summary in summaries])
        total_count = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        mean = (means * counts).sum(axis=0) / total_count
        variance = ((variances + (means - mean) ** 2) * counts).sum(axis=0) / total_count
        result[key] = {
            "min": np.min(np.stack([summary["min"] for summary in summaries]), axis=0).tolist(),
            "max": np.max(np.stack([summary["max"] for summary in summaries]), axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": total_count.tolist(),
        }
    return result


def write_v2_metadata_statistics(root: Path, info: dict[str, Any]) -> None:
    """Write the revision's required file and refresh any optional other summary.

    A native conversion changes vectors, so no stale per-episode or global
    summary may survive it. Images and all other columns remain untouched.
    """
    episodes = compute_episode_metadata(root, info)
    required = required_statistics_file(info["codebase_version"])
    episode_path, global_path = root / "meta/episodes_stats.jsonl", root / "meta/stats.json"
    if required.endswith("episodes_stats.jsonl") or episode_path.exists():
        episode_path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in episodes)
        )
    if required.endswith("stats.json") or global_path.exists():
        global_path.write_text(
            json.dumps(aggregate_metadata(episodes), indent=2, allow_nan=False) + "\n"
        )


def validate_v2_metadata_statistics(root: Path, info: dict[str, Any]) -> None:
    """Check reader-required shapes/counts and numeric summaries against source rows.

    Every image summary, including v2.0 globals, must match the supported
    pinned per-episode sampled-PNG estimator and population aggregate export.
    Other historical estimators require independent support, not a value bypass.
    """
    from .preparation import PreparationError, _json_object, _parquet_files, _pyarrow

    version = info["codebase_version"]
    required = root / required_statistics_file(version)
    if not required.is_file():
        raise PreparationError(f"incomplete LeRobot {version}: missing {required.name}")
    expected = compute_episode_metadata(root, info)

    def validate(
        actual: Any, reference: dict[str, Any], label: str, *, allow_missing_count: bool = False
    ) -> None:
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise PreparationError(f"{label} must describe every non-string feature")
        for feature, expected_stats in reference.items():
            value = actual[feature]
            if not isinstance(value, dict):
                raise PreparationError(f"{label} {feature} must contain statistics")
            for statistic in ("min", "max", "mean", "std"):
                try:
                    array = np.asarray(value[statistic], dtype=np.float64)
                except (KeyError, TypeError, ValueError) as exc:
                    raise PreparationError(f"{label} {feature}.{statistic} is invalid") from exc
                target = np.asarray(expected_stats[statistic])
                declared_shape = (
                    (3, 1, 1)
                    if info["features"][feature]["dtype"] == "image"
                    else tuple(info["features"][feature]["shape"])
                )
                if (
                    array.shape != declared_shape
                    or target.shape != declared_shape
                    or not np.isfinite(array).all()
                ):
                    raise PreparationError(
                        f"{label} {feature}.{statistic} has wrong shape/nonfinite values"
                    )
                image = info["features"][feature]["dtype"] == "image"
                if not np.allclose(array, target, rtol=1e-5, atol=1e-6):
                    convention = (
                        "; supported PNG statistics use pinned per-episode sampling "
                        "and aggregate export"
                        if image
                        else ""
                    )
                    raise PreparationError(
                        f"{label} {feature}.{statistic} disagrees with source rows/pixels"
                        f"{convention}"
                    )
            if info["features"][feature]["dtype"] == "image":
                low, high, mean, std = (
                    np.asarray(value[name]) for name in ("min", "max", "mean", "std")
                )
                slack = 32 * np.finfo(np.float64).eps
                if (
                    np.any(low < 0)
                    or np.any(high > 1)
                    or np.any(mean < low - slack)
                    or np.any(mean > high + slack)
                    or np.any(std > (high - low) / 2 + slack)
                ):
                    raise PreparationError(f"{label} {feature} has impossible RGB moments")
            if (not allow_missing_count or "count" in value) and value.get(
                "count"
            ) != expected_stats["count"]:
                raise PreparationError(
                    f"{label} {feature}.count disagrees with source rows/samples"
                )
            if np.any(np.asarray(value["std"]) < 0) or np.any(
                np.asarray(value["min"]) > np.asarray(value["max"])
            ):
                raise PreparationError(f"{label} {feature} has invalid ranges")

    episode_path, global_path = root / "meta/episodes_stats.jsonl", root / "meta/stats.json"
    if episode_path.exists():
        try:
            actual = [json.loads(line) for line in episode_path.read_text().splitlines()]
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError("invalid episodes_stats.jsonl") from exc
        if len(actual) != len(expected) or [row.get("episode_index") for row in actual] != list(
            range(len(expected))
        ):
            raise PreparationError("episode statistics must exactly match episode ownership")
        for row, reference in zip(actual, expected, strict=True):
            validate(
                row.get("stats"), reference["stats"], f"episode {row['episode_index']} statistics"
            )
    if global_path.exists():
        reference = aggregate_metadata(expected)
        if version == "v2.0":
            # Legacy globals are calculated over rows, rather than a float32
            # per-episode reduction followed by a weighted merge.
            _, pq = _pyarrow()
            for key, feature in info["features"].items():
                if feature["dtype"] not in {"image", "string"}:
                    arrays = [
                        np.asarray(
                            pq.read_table(path, columns=[key])[key].to_pylist(), dtype=np.float64
                        )
                        for path in _parquet_files(root)
                    ]
                    reference[key] = _moments(np.concatenate(arrays))
        validate(
            _json_object(global_path, "LeRobot global statistics"),
            reference,
            "global statistics",
            allow_missing_count=version == "v2.0",
        )
