"""Alert and analyst review contracts shared by storage and HTTP routes."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, Field, model_validator

from nexus.schemas import Category, Identifier, StrictModel, Timestamp

Severity = Literal["low", "medium", "high", "critical", "alert"]
Verdict = Literal["confirmed_attack", "false_positive", "needs_investigation", "pending"]
Probability = Annotated[float, Field(ge=0, le=1)]


def readable_text(value: str) -> str:
    if value != value.strip() or not value:
        raise ValueError("text must be nonempty with no surrounding whitespace")
    if any(not char.isprintable() and char not in "\n\t" for char in value):
        raise ValueError("text contains control characters")
    return value


Notes = Annotated[str, Field(min_length=1, max_length=2000), AfterValidator(readable_text)]


FeatureName = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_./+-]+$")]


class FeatureContribution(StrictModel):
    feature: FeatureName
    contribution: Annotated[float, Field(ge=-1e15, le=1e15)]
    value: float | str | None = None


class AlertData(StrictModel):
    """Internal ingestion contract. Public clients cannot create scored alerts."""

    alert_id: Identifier
    flow_id: Identifier
    event_time: Timestamp
    source: Literal["mock", "model"]
    bundle_version: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")]
    predicted_class: Category
    score: float | None = None
    threshold: float | None = None
    probability: Probability | None = None
    novelty_score: Annotated[float, Field(ge=0, le=1e15)] | None = None
    risk: Probability | None = None
    severity: Severity
    top_features: Annotated[list[FeatureContribution], Field(max_length=20)]

    @model_validator(mode="after")
    def unique_features(self) -> "AlertData":
        names = [item.feature for item in self.top_features]
        if len(names) != len(set(names)):
            raise ValueError("explanation features must be unique")
        return self


class AlertRecord(AlertData):
    sequence: int
    created_at: Timestamp
    feedback_version: int


class FeedbackInput(StrictModel):
    feedback_id: Identifier
    expected_version: Annotated[int, Field(ge=0, le=2**53 - 1)]
    verdict: Verdict
    attack_category: Category | None = None
    notes: Notes | None = None

    @model_validator(mode="after")
    def category_matches_verdict(self) -> "FeedbackInput":
        if (self.verdict == "confirmed_attack") != (self.attack_category is not None):
            raise ValueError("attack_category is required only for confirmed attacks")
        return self


class FeedbackRecord(StrictModel):
    sequence: int
    feedback_id: Identifier
    alert_id: Identifier
    version: int
    verdict: Verdict
    attack_category: Category | None
    notes: Notes | None
    reviewer_id: str
    created_at: Timestamp
    provenance: Literal["analyst_api"] = "analyst_api"
    # A recorded verdict still needs a separate curation gate before training.
    training_eligible: Literal[False] = False


def query_integer(value: object) -> object:
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal() or len(value) > 16:
            raise ValueError("expected an unsigned decimal integer")
        if len(value) > 1 and value.startswith("0"):
            raise ValueError("leading zeros are not supported")
        return int(value)
    return value


class PageQuery(StrictModel):
    limit: Annotated[int, BeforeValidator(query_integer), Field(ge=1, le=100)] = 50
    after: Annotated[int, BeforeValidator(query_integer), Field(ge=0, le=2**53 - 1)] = 0


class AlertQuery(PageQuery):
    severity: Severity | None = None


class AlertPage(StrictModel):
    items: list[AlertRecord]
    next_after: int | None


class FeedbackPage(StrictModel):
    items: list[FeedbackRecord]
    next_after: int | None
