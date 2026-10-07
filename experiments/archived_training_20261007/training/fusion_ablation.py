"""Matched fusion controls and opt-in overlapping failure-slice diagnostics."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARTITIONS = ("fit", "early", "fusion", "calibration", "evaluation")


def validate_partitions(split, labels, families, held_family):
    seen = set()
    for name in PARTITIONS:
        rows = np.asarray(split[name])
        if (
            rows.ndim != 1
            or not np.issubdtype(rows.dtype, np.integer)
            or not len(rows)
            or np.any(rows < 0)
            or np.any(rows >= len(labels))
        ):
            raise ValueError(f"Invalid {name} row indices")
        ids = set(rows.tolist())
        if len(ids) != len(rows) or seen.intersection(ids):
            raise ValueError(f"Duplicate/overlapping partition rows: {name}")
        seen.update(ids)
        if name != "evaluation" and held_family != "none" and np.any(families[rows] == held_family):
            raise ValueError(f"Held family leaked into {name}")
    if len(seen) != len(labels):
        raise ValueError("Partitions must cover the source exactly")


def fit_ablations(scores, split, labels, families, held_family, seed, c, representations):
    """Fit all controls together against one shared, validated partition bundle."""
    validate_partitions(split, labels, families, held_family)
    configurations = {"fusion_control": ("lightgbm",), "fusion_latent": ("lightgbm", "ae_latent")}
    for representation in representations:
        configurations[f"fusion_{representation}"] = ("lightgbm", representation)
        configurations[f"fusion_{representation}_latent"] = (
            "lightgbm",
            representation,
            "ae_latent",
        )
    fitted = {}
    # Validate every input before fitting any control.
    columns = set(sum(configurations.values(), ()))
    for part in ("fusion", "calibration", "evaluation"):
        for column in columns:
            if column not in scores[part]:
                raise ValueError(f"Missing saved representation {column}; no implicit retraining")
            values = np.asarray(scores[part][column])
            if values.shape != (len(split[part]),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid/alignment-mismatched scores: {part}/{column}")
            if column != "lightgbm" and np.any(values < 0):
                raise ValueError("Anomaly errors/distances must be nonnegative")
    for name, columns in configurations.items():

        def matrix(part, columns=columns):
            return np.column_stack(
                [
                    scores[part][key] if key == "lightgbm" else np.log1p(scores[part][key])
                    for key in columns
                ]
            )

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=seed),
        )
        model.fit(matrix("fusion"), labels[split["fusion"]])
        for part in ("fusion", "calibration", "evaluation"):
            scores[part][name] = model.predict_proba(matrix(part))[:, 1]
        fitted[name] = {"columns": columns, "model": model}
    return fitted


def slice_diagnostics(frame, labels, families, scores, decisions):
    """Overlapping feature slices; benign rows are contextual FP comparators."""
    output = []
    for column, value in (
        ("proto", "tcp"),
        ("proto", "udp"),
        ("service", "-"),
        ("service", "http"),
        ("state", "FIN"),
    ):
        selected = frame[column].astype(str).to_numpy() == value
        for cohort, mask in (
            ("Reconnaissance", selected & (families == "Reconnaissance")),
            ("benign", selected & (labels == 0)),
        ):
            distributions = {
                key: (
                    dict(
                        zip(
                            ("min", "q25", "median", "q75", "max"),
                            np.quantile(values[mask], [0, 0.25, 0.5, 0.75, 1]).tolist(),
                            strict=True,
                        )
                    )
                    if mask.any()
                    else None
                )
                for key, values in scores.items()
            }
            for method, prediction in decisions.items():
                budget = method.rsplit("__", 1)[1]
                baseline = decisions[f"lightgbm__{budget}"]
                output.append(
                    dict(
                        slice=f"{column}={value}",
                        cohort=cohort,
                        method=method,
                        rows=int(mask.sum()),
                        overlapping=True,
                        score_distributions=distributions,
                        recovered=int((mask & ~baseline & prediction & (labels == 1)).sum()),
                        lost=int((mask & baseline & ~prediction & (labels == 1)).sum()),
                        benign_false_positives=int((mask & prediction & (labels == 0)).sum()),
                        attack_misses=int((mask & ~prediction & (labels == 1)).sum()),
                    )
                )
    return output
