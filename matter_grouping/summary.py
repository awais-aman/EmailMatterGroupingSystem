"""Per-matter digests built from the emails alone, and prompts for an LLM.

The notebook sent each group to Gemini. That needs a paid API key, so the
default here is a deterministic digest. `build_prompt` still produces the
structured prompt for anyone who wants to paste it into a model of their
choice.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from .emails import Email, normalize_subject


@dataclass(frozen=True)
class MatterDigest:
    label: int
    subject: str
    emails: tuple[Email, ...]

    @property
    def participants(self) -> list[str]:
        return sorted({e.sender for e in self.emails} | {e.recipient for e in self.emails})

    @property
    def first(self) -> datetime:
        return self.emails[0].timestamp

    @property
    def last(self) -> datetime:
        return self.emails[-1].timestamp


def digest_matters(emails: Sequence[Email], labels: Sequence[int]) -> list[MatterDigest]:
    """One digest per cluster, emails oldest first, matters by first activity."""
    if len(emails) != len(labels):
        raise ValueError("emails and labels differ in length")
    groups: dict[int, list[Email]] = defaultdict(list)
    for email, label in zip(emails, labels, strict=True):
        groups[label].append(email)

    digests = []
    for label, members in groups.items():
        members.sort(key=lambda e: (e.timestamp, e.email_id))
        subjects = Counter(normalize_subject(e.subject) for e in members)
        # Most common subject; ties go to the alphabetically first for stable output.
        subject = min(subjects, key=lambda s: (-subjects[s], s))
        digests.append(MatterDigest(label, subject, tuple(members)))
    return sorted(digests, key=lambda d: (d.first, d.label))


def render_digest(digest: MatterDigest) -> str:
    lines = [
        f"Matter {digest.label}: {digest.subject}",
        f"  {len(digest.emails)} emails, {digest.first:%Y-%m-%d} to {digest.last:%Y-%m-%d}",
        f"  Participants ({len(digest.participants)}): {', '.join(digest.participants)}",
        "  Chronology:",
    ]
    for e in digest.emails:
        when = f"{e.timestamp:%Y-%m-%d %H:%M}"
        lines.append(f"    {when}  #{e.email_id:<3} {e.sender} -> {e.recipient}: {e.subject}")
    return "\n".join(lines)


def build_prompt(digest: MatterDigest) -> str:
    """The notebook's five-section prompt, fed only facts from the emails.

    The original prompt stated a fee arrangement and task list that appear
    in no email, so the model summarised invented terms as if they were in
    the correspondence. Here the model is told to say when a section isn't
    supported by the emails instead.
    """
    messages = "\n\n".join(
        f"[{e.timestamp:%Y-%m-%d %H:%M}] From: {e.sender} To: {e.recipient}\n"
        f"Subject: {e.subject}\n{e.body}"
        for e in digest.emails
    )
    return (
        "You are summarising one legal matter from its email correspondence.\n"
        "Write a 1-2 page summary with these sections:\n"
        "1. Recap of instructions\n"
        "2. Recipient details\n"
        "3. Scope of work and fee information\n"
        "4. Ongoing tasks\n"
        "5. Chronology of events\n"
        "Use only what the emails say. If a section is not covered by them, "
        "write 'Not stated in the correspondence.'\n\n"
        f"Matter subject: {digest.subject}\n"
        f"Participants: {', '.join(digest.participants)}\n\n"
        f"Emails, oldest first:\n\n{messages}\n"
    )
