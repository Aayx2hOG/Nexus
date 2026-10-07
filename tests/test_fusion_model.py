"""Exact packaged policy boundaries and manifest integrity."""

import hashlib
import json

import joblib
import numpy as np
import pytest

from nexus.fusion_model import FrozenFusionModel, decide, load_fusion_model


def config(kind):
    return {
        "format_version": 1,
        "feature_columns": ["a"],
        "binary": {"decision_threshold": 0.6},
        "ae": {"boundaries": [1.0, 2.0, 3.0]},
        "decision_rule": {
            "kind": kind,
            "confidence_band": [0.1, 0.65],
            "mode_attack_cutoffs": [0.65, 0.6, 0.6, 0.6],
        },
    }


def test_band_endpoints_and_four_mode_decisions():
    pred, modes, base = decide(
        [0.09, 0.1, 0.6, 0.65, 0.66, 0.5], [3, 2, 1, 0, 0, 3], config("uncertainty_band_four_mode")
    )
    np.testing.assert_array_equal(pred, [0, 1, 1, 0, 1, 1])
    np.testing.assert_array_equal(modes, [3, 2, 1, 0, 0, 3])
    np.testing.assert_array_equal(base, [0, 0, 1, 1, 1, 0])


def test_confidence_mode_and_cutoff_equality():
    pred, _, _ = decide([0.65, 0.64, 0.6, 0.59], [0, 0, 1, 2], config("mode_confidence"))
    np.testing.assert_array_equal(pred, [1, 0, 1, 0])


def test_loader_and_tampered_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(config("mode_confidence")))
    (tmp_path / "metrics.json").write_text("{}")
    joblib.dump({}, tmp_path / "weights.joblib")
    hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ["config.json", "metrics.json", "weights.joblib"]
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": hashes}))
    assert isinstance(load_fusion_model(tmp_path), FrozenFusionModel)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_fusion_model(tmp_path)


@pytest.mark.parametrize("probability", [[np.nan], [1.1], [-0.1]])
def test_invalid_scores(probability):
    with pytest.raises(ValueError):
        decide(probability, [0.0], config("mode_confidence"))
