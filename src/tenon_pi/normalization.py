"""OpenPI-compatible data-only normalization statistics for Robot preparation.

The histogram algorithm and JSON representation are faithfully adapted from
``Physical-Intelligence/openpi`` ``src/openpi/shared/normalize.py`` at the
revision pinned by the Robot transform descriptor. OpenPI is Apache-2.0.
This module deliberately has no JAX, Torch, model-weight, or GPU dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import TypeAdapter

OPENPI_NORMALIZE_SOURCE = "src/openpi/shared/normalize.py"
OPENPI_NORMALIZE_SOURCE_SHA256 = "6c2cea4946fb07e51801530400d2b2fd94730e195cd290d6b6960114eca9739d"


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray

    def json_value(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
        }


class RunningStats:
    """Compute OpenPI's online mean/std and 5,000-bin approximate quantiles."""

    def __init__(self) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._minimum: np.ndarray | None = None
        self._maximum: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = 5_000

    @property
    def count(self) -> int:
        return self._count

    def update(self, batch: np.ndarray) -> None:
        values = np.asarray(batch)
        if values.ndim < 1 or values.shape[-1] < 1 or not np.isfinite(values).all():
            raise ValueError("normalization batch must contain finite vectors")
        values = values.reshape(-1, values.shape[-1])
        num_elements, vector_length = values.shape
        if num_elements == 0:
            raise ValueError("normalization batch must not be empty")

        if self._count == 0:
            self._mean = np.mean(values, axis=0)
            self._mean_of_squares = np.mean(values**2, axis=0)
            self._minimum = np.min(values, axis=0)
            self._maximum = np.max(values, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(
                    self._minimum[index] - 1e-10,
                    self._maximum[index] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for index in range(vector_length)
            ]
        else:
            mean = self._require_array(self._mean)
            minimum = self._require_array(self._minimum)
            maximum = self._require_array(self._maximum)
            if vector_length != mean.size:
                raise ValueError("normalization vector width changed")
            new_maximum = np.max(values, axis=0)
            new_minimum = np.min(values, axis=0)
            maximum_changed = np.any(new_maximum > maximum)
            minimum_changed = np.any(new_minimum < minimum)
            self._maximum = np.maximum(maximum, new_maximum)
            self._minimum = np.minimum(minimum, new_minimum)
            if maximum_changed or minimum_changed:
                self._adjust_histograms()

        self._count += num_elements
        mean = self._require_array(self._mean)
        mean_of_squares = self._require_array(self._mean_of_squares)
        batch_mean = np.mean(values, axis=0)
        batch_mean_of_squares = np.mean(values**2, axis=0)
        self._mean = mean + (batch_mean - mean) * (num_elements / self._count)
        self._mean_of_squares = mean_of_squares + (batch_mean_of_squares - mean_of_squares) * (
            num_elements / self._count
        )
        self._update_histograms(values)

    def get_statistics(self) -> NormalizationStats:
        if self._count < 2:
            raise ValueError("cannot compute normalization statistics from fewer than two vectors")
        mean = self._require_array(self._mean)
        mean_of_squares = self._require_array(self._mean_of_squares)
        variance = mean_of_squares - mean**2
        standard_deviation = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles((0.01, 0.99))
        return NormalizationStats(mean=mean, std=standard_deviation, q01=q01, q99=q99)

    @staticmethod
    def _require_array(value: np.ndarray | None) -> np.ndarray:
        if value is None:  # pragma: no cover - internal invariant
            raise RuntimeError("running statistics are not initialized")
        return value

    def _require_histograms(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self._histograms is None or self._bin_edges is None:  # pragma: no cover
            raise RuntimeError("running histograms are not initialized")
        return self._histograms, self._bin_edges

    def _adjust_histograms(self) -> None:
        histograms, edges = self._require_histograms()
        minimum = self._require_array(self._minimum)
        maximum = self._require_array(self._maximum)
        for index in range(len(histograms)):
            old_edges = edges[index]
            new_edges = np.linspace(minimum[index], maximum[index], self._num_quantile_bins + 1)
            new_histogram, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=histograms[index]
            )
            histograms[index] = new_histogram
            edges[index] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        histograms, edges = self._require_histograms()
        for index in range(batch.shape[1]):
            histogram, _ = np.histogram(batch[:, index], bins=edges[index])
            histograms[index] += histogram

    def _compute_quantiles(self, quantiles: tuple[float, ...]) -> list[np.ndarray]:
        histograms, edges = self._require_histograms()
        results: list[np.ndarray] = []
        for quantile in quantiles:
            target_count = quantile * self._count
            values: list[float] = []
            for histogram, bin_edges in zip(histograms, edges, strict=True):
                cumulative = np.cumsum(histogram)
                index = int(np.searchsorted(cumulative, target_count))
                values.append(float(bin_edges[index]))
            results.append(np.asarray(values))
        return results


def serialize_normalization(stats: dict[str, NormalizationStats]) -> bytes:
    """Return exactly the JSON representation consumed by pinned OpenPI."""

    if list(stats) != ["state", "actions"]:
        raise ValueError("normalization requires ordered state and actions statistics")
    payload: dict[str, Any] = {
        "norm_stats": {name: value.json_value() for name, value in stats.items()}
    }
    if any(
        not np.isfinite(values).all()
        for value in stats.values()
        for values in (value.mean, value.std, value.q01, value.q99)
    ):
        raise ValueError("normalization statistics must be finite")
    # OpenPI uses Pydantic's JSON encoder (e-8, not Python json's e-08).
    # Newer Pydantic emits e+20; the pinned encoder emits e20. This payload
    # contains only fixed keys and numeric arrays, so canonicalize that spelling.
    return TypeAdapter(dict[str, Any]).dump_json(payload, indent=2).replace(b"e+", b"e")
