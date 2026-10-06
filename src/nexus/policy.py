"""Read-only threshold counterfactuals on labeled replay, never release calibration."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from nexus.routes import Store, require_analyst

router = APIRouter(
    prefix="/api/v1/replay", tags=["replay"], dependencies=[Depends(require_analyst)]
)


class PolicyQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    bundle_version: str = Field(min_length=1, max_length=128)
    threshold: float = Field(ge=0, le=1)


def metrics(tn: int, fp: int, fn: int, tp: int) -> dict:
    return {
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (tn + fp) if tn + fp else None,
        "alert_count": tp + fp,
    }


@router.get("/policy")
def policy_lab(store: Store, query: Annotated[PolicyQuery, Query()]) -> dict:
    rows = store.get_policy_counts(query.bundle_version, query.threshold)
    baseline = [0, 0, 0, 0]  # TN, FP, FN, TP
    candidate = [0, 0, 0, 0]
    recovered = lost = added_fp = removed_fp = total = labeled = last_sequence = 0
    families: dict[str, dict] = {}
    for row in rows:
        n = row["n"]
        total += n
        last_sequence = max(last_sequence, row["last_sequence"])
        if row["label"] is None:
            continue
        labeled += n
        attack = row["label"] == 1
        before = row["baseline"] == "alert"
        after = bool(row["candidate"])
        baseline[2 * int(attack) + int(before)] += n
        candidate[2 * int(attack) + int(after)] += n
        if attack:
            recovered += n if after and not before else 0
            lost += n if before and not after else 0
            family = families.setdefault(
                row["attack_cat"],
                {
                    "family": row["attack_cat"],
                    "total": 0,
                    "baseline_detected": 0,
                    "candidate_detected": 0,
                },
            )
            family["total"] += n
            family["baseline_detected"] += n * before
            family["candidate_detected"] += n * after
        else:
            added_fp += n if after and not before else 0
            removed_fp += n if before and not after else 0
    return {
        "bundle_version": query.bundle_version,
        "threshold": query.threshold,
        "total_predictions": total,
        "labeled_predictions": labeled,
        "unlabeled_predictions": total - labeled,
        "last_sequence": last_sequence,
        "baseline": metrics(*baseline),
        "candidate": metrics(*candidate),
        "recovered_attacks": recovered,
        "lost_attacks": lost,
        "added_false_positives": added_fp,
        "removed_false_positives": removed_fp,
        "families": sorted(families.values(), key=lambda f: f["family"]),
        "note": "Historical labeled replay only; exploratory threshold simulation, not "
        "fusion, calibration, independent evaluation, or a deployment change.",
    }
