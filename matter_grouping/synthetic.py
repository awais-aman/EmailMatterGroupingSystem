"""Seeded generator for synthetic legal emails.

All addresses come from Faker's safe mode (example.com/.net/.org), so the
output never contains a real person's mailbox.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from faker import Faker

from .emails import Email

MATTER_TEMPLATES: dict[int, dict] = {
    1: {
        "subject": "Litigation Case Update",
        "bodies": [
            "Recent developments in the litigation have been significant. "
            "Please review the attached case documents for updates.",
            "The litigation case has seen new progress. Kindly examine the "
            "updated court submissions and legal opinions.",
        ],
    },
    2: {
        "subject": "Contract Negotiation Discussion",
        "bodies": [
            "The contract draft has been revised based on our recent negotiations. "
            "Please verify the changes and suggest modifications.",
            "We have updated the contract terms in line with the latest negotiation "
            "feedback. Your input on the proposed clauses is appreciated.",
        ],
    },
    3: {
        "subject": "Patent Infringement Inquiry",
        "bodies": [
            "An inquiry regarding potential patent infringement has been received. "
            "We need to review the attached documents thoroughly.",
            "Please assess the evidence related to the patent infringement issue. "
            "Your expert opinion on the technical details is required.",
        ],
    },
    4: {
        "subject": "Property Dispute Resolution",
        "bodies": [
            "The property dispute has entered a critical phase. Review the attached "
            "resolution strategy document and provide your recommendations.",
            "Updates have been made on the property dispute. Please examine the "
            "legal analysis and offer your guidance on the next steps.",
        ],
    },
}

REPLY_BODY = (
    "Following up on our previous discussion, please see my comments and suggested revisions."
)

# A fixed anchor keeps output identical for a given seed, whatever today is.
DEFAULT_END = datetime(2025, 1, 1)


def generate_emails(
    count: int = 25,
    replies: int = 3,
    seed: int = 7,
    end: datetime = DEFAULT_END,
) -> list[Email]:
    """Return `count` matter emails plus `replies` threaded replies, oldest first."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if replies < 0:
        raise ValueError("replies cannot be negative")

    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)
    start = end - timedelta(days=365)

    emails: list[Email] = []
    for email_id in range(1, count + 1):
        matter_id = rng.choice(sorted(MATTER_TEMPLATES))
        template = MATTER_TEMPLATES[matter_id]
        prefix = rng.choice(["", "Re: ", "Fwd: "])
        emails.append(
            Email(
                email_id=email_id,
                sender=fake.email(safe=True),
                recipient=fake.email(safe=True),
                timestamp=fake.date_time_between(start_date=start, end_date=end),
                subject=prefix + template["subject"],
                body=rng.choice(template["bodies"]),
                true_matter_id=matter_id,
            )
        )

    for _ in range(replies):
        original = rng.choice(emails)
        subject = original.subject
        if not subject.startswith("Re: "):
            subject = "Re: " + subject
        emails.append(
            Email(
                email_id=len(emails) + 1,
                sender=original.recipient,
                recipient=original.sender,
                timestamp=original.timestamp + timedelta(hours=rng.randint(1, 24)),
                subject=subject,
                body=REPLY_BODY,
                true_matter_id=original.true_matter_id,
            )
        )

    return sorted(emails, key=lambda e: (e.timestamp, e.email_id))
