"""Authenticated shadow evidence; no promotion or alert-creation endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from nexus.alerts import PageQuery
from nexus.routes import Store, no_query, require_analyst

router = APIRouter(
    prefix="/api/v1/shadow", tags=["shadow"], dependencies=[Depends(require_analyst)]
)


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


def summarize(rows: list[dict]) -> dict:
    counts = {name: [0, 0, 0, 0] for name in ("live", "reference", "candidate")}
    deltas = {
        name: {
            "recovered_attacks": 0,
            "lost_attacks": 0,
            "added_false_positives": 0,
            "removed_false_positives": 0,
        }
        for name in ("live", "reference")
    }
    total = scored = errors = labeled = additions = removals = 0
    families: dict[str, dict] = {}
    for row in rows:
        n = row["n"]
        total += n
        if row["status"] != "scored":
            errors += n if row["status"] == "error" else 0
            continue
        scored += n
        candidate = row["candidate"] == "alert"
        additions += n if candidate and row["live"] != "alert" else 0
        removals += n if not candidate and row["live"] == "alert" else 0
        if row["label"] is None:
            continue
        labeled += n
        attack = row["label"] == 1
        for name in counts:
            counts[name][2 * int(attack) + int(row[name] == "alert")] += n
        for name, delta in deltas.items():
            before = row[name] == "alert"
            if candidate and not before:
                delta["recovered_attacks" if attack else "added_false_positives"] += n
            if before and not candidate:
                delta["lost_attacks" if attack else "removed_false_positives"] += n
        if attack:
            family = families.setdefault(
                row["attack_cat"],
                {
                    "family": row["attack_cat"],
                    "total": 0,
                    "live_detected": 0,
                    "reference_detected": 0,
                    "candidate_detected": 0,
                },
            )
            family["total"] += n
            for name in counts:
                family[f"{name}_detected"] += n if row[name] == "alert" else 0
    return {
        "total_predictions": total,
        "scored": scored,
        "errors": errors,
        "not_shadowed": total - scored - errors,
        "labeled_scored": labeled,
        "unlabeled_scored": scored - labeled,
        "candidate_additions_vs_live": additions,
        "candidate_removals_vs_live": removals,
        "metrics": {name: metrics(*values) for name, values in counts.items()},
        "vs_live": deltas["live"],
        "vs_reference": deltas["reference"],
        "families": sorted(families.values(), key=lambda f: f["family"]),
    }


@router.get("", dependencies=[Depends(no_query)])
def shadow_summary(request: Request, store: Store) -> dict:
    service = request.app.state.shadow_service
    status = service.status()
    version = request.app.state.settings.bundle_version
    rows = (
        store.shadow_counts(version, status["manifest_sha256"])
        if service.bundle and version
        else []
    )
    return {**status, "production_bundle_version": version, **summarize(rows)}


@router.get("/predictions")
def shadow_predictions(
    request: Request, store: Store, query: Annotated[PageQuery, Query()]
) -> dict:
    service = request.app.state.shadow_service
    version = request.app.state.settings.bundle_version
    if service.bundle is None or version is None:
        return {"items": [], "next_after": None}
    return store.list_shadow(version, service.bundle.manifest_sha256, query)
