"""Export the two user-selected AE-FPR configurations and verify full test parity."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from nexus.fusion_model import load_fusion_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "experiments/four_mode_ae_20261007"
    experiment = ROOT / "experiments/ae_fpr_5_10_20261007"
    thresholds = json.loads((experiment / "frozen_thresholds.json").read_text())
    protocol = json.loads((experiment / "protocol.json").read_text())
    results = json.loads((experiment / "test_results.json").read_text())
    calibration = json.loads((experiment / "ae_calibration.json").read_text())
    audit = json.loads((experiment / "audit.json").read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def dump(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    for name in ["binary.joblib", "autoencoder.joblib"]:
        path = source / name
        assert digest(path) == audit["source_hashes"][str(path.relative_to(ROOT))]
    binary = joblib.load(source / "binary.joblib")
    ae = joblib.load(source / "autoencoder.joblib")
    # Only standard sklearn/LightGBM objects and built-in mappings enter the package.
    f = binary.features
    weights = dict(
        binary=dict(
            estimator=binary.estimator,
            columns=f.native.columns,
            categories=f.native.categories,
            numeric=f.numeric,
            categorical=f.categorical,
            encoder=f.encoder,
        ),
        ae=ae,
    )
    versions = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        scikit_learn=sklearn.__version__,
        lightgbm=lightgbm.__version__,
        joblib=joblib.__version__,
    )
    selections = [
        (
            "uncertainty_band_ae10",
            0.1,
            "Uncertainty-band four-mode",
            dict(kind="uncertainty_band_four_mode", confidence_band=protocol["uncertainty_band"]),
        ),
        (
            "mode_confidence_ae05",
            0.05,
            "Mode confidence",
            dict(kind="mode_confidence", mode_attack_cutoffs=protocol["confidence_cutoffs"]),
        ),
    ]
    testpath = ROOT / "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv"
    assert digest(testpath) == audit["test_sha256"]
    test = pd.read_csv(testpath)
    expected = np.load(experiment / "test_predictions.npz")
    np.testing.assert_array_equal(test.label.to_numpy(), expected["y"])
    for slug, budget, name, rule in selections:
        out = ROOT / "models" / slug
        out.mkdir(exist_ok=False)
        config = dict(
            format_version=1,
            model_id=slug,
            display_name=name,
            seed=42,
            labels={"0": "normal", "1": "attack"},
            feature_columns=f.native.columns,
            inference_batch_size=2048,
            decision_rule=rule,
            binary=dict(
                decision_threshold=protocol["binary_cutoff"],
                parameters=binary.estimator.get_params(),
            ),
            ae=dict(
                calibration_fpr_target=budget,
                boundaries=thresholds[str(budget)]["recent_boundaries"],
                mode_names=["no_anomaly", "low", "mid", "high"],
                boundary_equality="higher mode",
                numeric_columns=ae["numerical_columns"],
                categorical_columns=ae["categorical_columns"],
                error=(
                    "Mean squared reconstruction error over standardized numeric and one-hot inputs"
                ),
                parameters=ae["model"].get_params(),
                training=json.loads((source / "ae_training.json").read_text()),
                calibration=next(
                    r for r in calibration if r["model"] == "Recent AE" and r["target"] == budget
                ),
            ),
            preprocessing=(
                "All fitted binary and AE encoders, numeric imputer/scaler are in weights.joblib"
            ),
            versions=versions,
            source_experiment=str(experiment.relative_to(ROOT)),
            source_sha256={n: digest(source / n) for n in ["binary.joblib", "autoencoder.joblib"]},
            score_semantics="LightGBM attack score is not a calibrated final attack probability",
        )
        metrics = next(r for r in results if r["name"] == name and r["ae_target"] == budget)
        dump(out / "config.json", config)
        dump(
            out / "metrics.json",
            dict(
                dataset="Official UNSW-NB15 testing set",
                test_sha256=audit["test_sha256"],
                rows=len(test),
                result=metrics,
            ),
        )
        joblib.dump(weights, out / "weights.joblib", compress=3)
        dump(
            out / "manifest.json",
            dict(
                format_version=1,
                model_id=slug,
                sha256={
                    n: digest(out / n) for n in ["config.json", "weights.joblib", "metrics.json"]
                },
            ),
        )
        packaged = load_fusion_model(out)
        details = packaged.predict_details(test)
        np.testing.assert_array_equal(details.prediction.to_numpy(), expected[f"{budget}/{name}"])
        np.testing.assert_allclose(
            details.ae_error.to_numpy(), expected["recent_ae_error"], rtol=1e-6, atol=1e-10
        )
        assert (
            confusion_matrix(test.label, details.prediction, labels=[0, 1]).tolist()
            == metrics["confusion_matrix"]
        )
        dump(
            out / "verification.json",
            dict(
                full_test_rows=len(test),
                prediction_mismatches=0,
                ae_score_parity=True,
                confusion_matrix=metrics["confusion_matrix"],
                verified_from_raw_features=True,
                training_module_imports_required=False,
            ),
        )
        (out / "README.md").write_text(
            f"# {name} — AE calibration FPR {budget:.0%}\n\n"
            "Load this directory with `nexus.fusion_model.load_fusion_model`. "
            "The package includes all fitted preprocessing and both base estimators. "
            "See `config.json` for exact cutoffs, `metrics.json` for test results, and "
            "`verification.json` for full-test prediction parity.\n\n"
            "AE calibration FPR is not a system/test FPR guarantee. "
            "Input is a DataFrame of the 42 raw flow features; "
            "extra label/id columns are ignored.\n"
        )
        print(f"Exported {out}: all {len(test):,} predictions match", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
