"""Regression tests for grouped native-feature experiments and saved cascades."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from fast_lightgbm_models import AttackCascade, NativeFeatures, NativeModel  # noqa: E402
from tune_lightgbm_fast import (  # noqa: E402
    Deadline,
    audit_training,
    conservative_threshold,
    grouped_splits,
    macro_f1,
    run_search,
    selection_rank,
)


def test_grouped_split_keeps_duplicate_and_conflicting_rows_together():
    frame = pd.DataFrame({"x": np.repeat(np.arange(210), 2)})
    y = np.repeat(np.arange(210) % 3, 2)
    y[0] = 1
    groups, audit = audit_training(frame, y)
    split = grouped_splits(y, groups, 42)
    assert audit["duplicate_rows_beyond_first"] == 210
    assert audit["conflicting_label_groups"] == 1
    seen = set()
    rows = set()
    for indices in split.values():
        current = set(groups[indices])
        assert not seen & current
        seen |= current
        assert not rows & set(indices)
        rows |= set(indices)
    assert rows == set(range(len(y)))


def test_native_vocab_only_uses_fitting_rows_and_unknown_becomes_missing():
    features = NativeFeatures().fit(pd.DataFrame({"proto": ["tcp", "udp"], "n": [1, 2]}))
    transformed = features.transform(pd.DataFrame({"proto": ["sctp"], "n": [np.inf]}))
    assert list(transformed.proto.cat.categories) == ["tcp", "udp"]
    assert transformed.isna().all().all()


def test_recall_confidence_is_more_conservative():
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    scores = np.linspace(0, 1, 2000)
    threshold, target = conservative_threshold(y, scores, 0.95, 0.95)
    ordinary, target0 = conservative_threshold(y, scores, 0.95, 0)
    assert target > target0
    assert threshold <= ordinary
    assert np.mean(scores[y == 1] >= threshold) >= target
    with pytest.raises(ValueError, match="Too few"):
        conservative_threshold(np.array([0, 1]), np.array([0.1, 0.8]), 0.95, 0.95)


def test_macro_callback_matches_sklearn():
    rng = np.random.default_rng(42)
    scores = rng.random((60, 3))
    y = np.arange(60) % 3
    name, value, higher = macro_f1(y, scores)
    assert name == "macro_f1" and higher
    assert value == pytest.approx(f1_score(y, scores.argmax(axis=1), average="macro"))


def test_deadline_preserves_best_iteration():
    callback = Deadline(float("inf"))
    callback(SimpleNamespace(iteration=0, evaluation_result_list=[("valid", "loss", 0.2, False)]))
    callback.end = 0
    with pytest.raises(lgb.callback.EarlyStopException) as exc:
        callback(
            SimpleNamespace(iteration=1, evaluation_result_list=[("valid", "loss", 0.4, False)])
        )
    assert exc.value.best_iteration == 0


def test_native_search_and_cascade_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    n = 1400
    y = np.arange(n) % 4
    raw = pd.DataFrame({"x": y + rng.normal(0, 0.2, n), "proto": np.where(y % 2, "tcp", "udp")})
    groups, _ = audit_training(raw, y)
    split = grouped_splits(y, groups, 42)
    features = NativeFeatures().fit(raw.iloc[split["fit"]])
    X = features.transform(raw)
    args = SimpleNamespace(
        minutes=0.2,
        trials=2,
        max_rounds=12,
        patience=3,
        seed=42,
        n_jobs=1,
        minimum_recall=0.95,
        recall_confidence=0,
        no_cascade=False,
    )
    gate_bundle, _ = run_search(
        "binary", tmp_path / "binary", X, y, split, features, ["Normal", "A", "B", "C"], args
    )
    bundle, result = run_search(
        "multiclass",
        tmp_path / "multiclass",
        X,
        y,
        split,
        features,
        ["Normal", "A", "B", "C"],
        args,
        gate_bundle.estimator,
    )
    saved = joblib.load(tmp_path / "multiclass/model.joblib")
    np.testing.assert_array_equal(saved.predict(raw), bundle.predict(raw))
    assert result["selection_constraint_met"]
    trials = json.loads((tmp_path / "multiclass/all_trial_metrics.json").read_text())
    assert [t["kind"] for t in trials] == ["direct", "cascade", "direct"]
    attack_rows = y > 0
    attack = lgb.LGBMClassifier(n_estimators=3, n_jobs=1, verbosity=-1).fit(
        X.loc[attack_rows], y[attack_rows] - 1
    )
    cascade = AttackCascade(gate_bundle.estimator, attack, 1.0, 4)
    native = NativeModel(features, cascade)
    path = tmp_path / "cascade.joblib"
    joblib.dump(native, path)
    loaded = joblib.load(path)
    assert (loaded.predict(raw) == 0).all()
    np.testing.assert_allclose(loaded.predict_proba(raw).sum(axis=1), 1)
    assert loaded.classes_.tolist() == [0, 1, 2, 3]
    assert selection_rank({"recall": 0.94, "false_positive_rate": 0}, "binary", 0.95)[0] == 1
