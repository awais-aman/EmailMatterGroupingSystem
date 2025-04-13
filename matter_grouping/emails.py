"""The email record and its JSON Lines storage format."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Reply/forward markers, possibly stacked ("Re: Fwd: RE:") and in any case.
_REPLY_PREFIX = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Email:
    email_id: int
    sender: str
    recipient: str
    timestamp: datetime
    subject: str
    body: str
    true_matter_id: int | None = None

    def to_json(self) -> dict:
        record = asdict(self)
        record["timestamp"] = self.timestamp.isoformat()
        return record

    @classmethod
    def from_json(cls, record: dict) -> Email:
        return cls(
            email_id=int(record["email_id"]),
            sender=record["sender"],
            recipient=record["recipient"],
            timestamp=datetime.fromisoformat(record["timestamp"]),
            subject=record["subject"],
            body=record["body"],
            true_matter_id=record.get("true_matter_id"),
        )


def normalize_subject(subject: str) -> str:
    """Strip reply/forward prefixes so a thread shares one subject."""
    return _REPLY_PREFIX.sub("", subject).strip()


def clean_text(text: str) -> str:
    """Lowercase and collapse whitespace so formatting doesn't change embeddings."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def write_jsonl(emails: Iterable[Email], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for email in emails:
            fh.write(json.dumps(email.to_json()) + "\n")


def read_jsonl(path: Path) -> list[Email]:
    with path.open(encoding="utf-8") as fh:
        return [Email.from_json(json.loads(line)) for line in fh if line.strip()]
