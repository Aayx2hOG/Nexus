"""Budget-calibrated recovery of supervised detector misses.

Gate settings are declared before evaluation. Calibration receives only benign
scores; it cannot optimize against evaluation labels or withheld attack families.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SelectiveFusion:
    baseline_threshold: float
    recovery_threshold: float
    uncertain_lower_ratio: float
    suspicious_quantile: float
    calibration_rows: int
    baseline_false_positives: int
    allowed_additional_false_positives: int
    calibration_false_positives: int
    primary_budget_fraction: float = 1.0
    full_budget_baseline_threshold: float | None = None

    def route(self, tree, anomaly_rank):
        return (tree < self.baseline_threshold) & (
            (tree >= self.baseline_threshold * self.uncertain_lower_ratio)
            | (anomaly_rank >= self.suspicious_quantile)
        )

    def predict(self, tree, fusion, anomaly_rank):
        return (tree >= self.baseline_threshold) | (
            self.route(tree, anomaly_rank) & (fusion >= self.recovery_threshold)
        )

    def score(self, tree, fusion, anomaly_rank):
        """Continuous ranking matching the conditional decision order.

        LightGBM positives rank highest; eligible negatives rank by fusion score;
        ineligible negatives rank lowest. This score depends on the budget gate.
        """
        return np.where(
            tree >= self.baseline_threshold,
            2 + tree,
            np.where(self.route(tree, anomaly_rank), fusion, tree - 2),
        )

    def to_dict(self):
        return asdict(self)


def calibrate_selective(
    tree,
    fusion,
    anomaly_rank,
    baseline_threshold,
    budget,
    uncertain_lower_ratio=0.5,
    suspicious_quantile=0.99,
    min_fusion_score=0.0,
    primary_budget_fraction=1.0,
):
    """Calibrate recovery using benign rows and a predeclared primary share.

    Share 1 preserves full-budget alerts and spends remaining headroom. Smaller
    shares recalibrate the primary branch and can lose original baseline alerts.
    Arrays contain benign calibration rows only. With zero FP headroom, recovery
    requires a score strictly above every routed benign calibration score.
    """
    tree, fusion, anomaly_rank = (np.asarray(a, dtype=float) for a in (tree, fusion, anomaly_rank))
    if not len(tree) or tree.shape != fusion.shape or tree.shape != anomaly_rank.shape:
        raise ValueError("Calibration scores must be aligned nonempty arrays")
    if not all(np.isfinite(a).all() for a in (tree, fusion, anomaly_rank)):
        raise ValueError("Non-finite calibration scores")
    if not 0 < budget < 1 or not all(
        0 <= v <= 1
        for v in (
            uncertain_lower_ratio,
            suspicious_quantile,
            min_fusion_score,
        )
    ):
        raise ValueError("Budget must be in (0,1); gate parameters must be in [0,1]")
    if not 0 < primary_budget_fraction <= 1:
        raise ValueError("Primary budget fraction must be in (0,1]")
    full_budget_threshold = baseline_threshold
    if primary_budget_fraction < 1:
        from anomaly_detection_models import fpr_threshold

        baseline_threshold = fpr_threshold(tree, budget * primary_budget_fraction)
    baseline_fp = int((tree >= baseline_threshold).sum())
    remaining = int(np.floor(budget * len(tree))) - baseline_fp
    if remaining < 0:
        raise ValueError("Baseline already exceeds calibration FPR budget")
    if primary_budget_fraction < 1:
        remaining = min(
            remaining, int(np.floor(budget * (1 - primary_budget_fraction) * len(tree)))
        )
    policy = SelectiveFusion(
        baseline_threshold,
        min_fusion_score,
        uncertain_lower_ratio,
        suspicious_quantile,
        len(tree),
        baseline_fp,
        remaining,
        baseline_fp,
    )
    policy.primary_budget_fraction = primary_budget_fraction
    policy.full_budget_baseline_threshold = full_budget_threshold
    routed = fusion[policy.route(tree, anomaly_rank)]
    if len(routed) > remaining:
        boundary = np.sort(routed)[len(routed) - remaining - 1]
        policy.recovery_threshold = max(min_fusion_score, float(np.nextafter(boundary, np.inf)))
    policy.calibration_false_positives = int(policy.predict(tree, fusion, anomaly_rank).sum())
    if policy.calibration_false_positives > int(np.floor(budget * len(tree))):
        raise AssertionError("Selective calibration violated its FP budget")
    return policy
