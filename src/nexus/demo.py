"""Explicitly seed synthetic alerts for local API/dashboard development."""

from uuid import NAMESPACE_URL, uuid5

from nexus.alerts import AlertData, FeatureContribution
from nexus.config import Settings
from nexus.storage import AlertStore


def demo_alerts() -> list[AlertData]:
    return [
        AlertData(
            alert_id=str(uuid5(NAMESPACE_URL, f"nexus-demo-alert-{index}")),
            flow_id=str(uuid5(NAMESPACE_URL, f"nexus-demo-flow-{index}")),
            event_time="2026-10-02T10:00:00Z",
            source="mock",
            bundle_version="demo-v1",
            predicted_class="SyntheticAttack",
            probability=0.9,
            novelty_score=0.7,
            risk=0.8,
            severity=severity,
            top_features=[FeatureContribution(feature="sbytes", contribution=0.25)],
        )
        for index, severity in enumerate(("high", "critical"), start=1)
    ]


def main() -> None:
    store = AlertStore(Settings.from_env().database_path)
    store.initialize()
    alerts = demo_alerts()
    for alert in alerts:
        store.record_alert(alert)
    print(
        f"Loaded {len(alerts)} synthetic alerts (source=mock). "
        "Existing identical alerts were preserved."
    )


if __name__ == "__main__":
    main()
