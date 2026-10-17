from __future__ import annotations

import hashlib

import numpy as np

from tenon_pi.normalization import NormalizationStats, serialize_normalization


def test_exact_pinned_json_exponent_and_signed_zero_serialization():
    # Independent golden from OpenPI 215abfb serialize_json / NormStats. Python
    # json.dumps spells e-08/e+20 instead; numerically equal is not byte-equal.
    values = np.array([1.2325354958875323e-8, 1e20, 1e-10, -0.0], np.float64)
    stats = {
        key: NormalizationStats(values, values, values, values) for key in ("state", "actions")
    }
    encoded = serialize_normalization(stats)
    assert hashlib.sha256(encoded).hexdigest() == (
        "65ae67e426c8da722de40d4f9c80524cb5779ff7fb6bdede909b1d8401159fad"
    )
