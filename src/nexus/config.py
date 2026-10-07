"""Explicit application configuration; no filesystem work during import."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path("artifacts/nexus.sqlite3")
    bundles_dir: Path = Path("artifacts/bundles")
    bundle_version: str | None = None
    api_token: str | None = field(default=None, repr=False)
    reviewer_id: str = "local-analyst"

    def __post_init__(self) -> None:
        if self.api_token is not None and (
            len(self.api_token) < 32
            or not self.api_token.isascii()
            or not self.api_token.isprintable()
            or any(c.isspace() for c in self.api_token)
        ):
            raise ValueError("API token must contain at least 32 non-whitespace ASCII characters")
        if not 1 <= len(self.reviewer_id) <= 64 or not all(
            c.isascii() and (c.isalnum() or c in "_.-") for c in self.reviewer_id
        ):
            raise ValueError("Reviewer ID must be a 1–64 character ASCII identifier")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_path=Path(os.environ.get("NEXUS_DATABASE_PATH", "artifacts/nexus.sqlite3")),
            bundles_dir=Path(os.environ.get("NEXUS_BUNDLES_DIR", "artifacts/bundles")),
            bundle_version=os.environ.get("NEXUS_BUNDLE_VERSION"),
            api_token=os.environ.get("NEXUS_API_TOKEN"),
            reviewer_id=os.environ.get("NEXUS_REVIEWER_ID", "local-analyst"),
        )
