from datetime import datetime

import pytest

from matter_grouping import Email, group_emails
from matter_grouping.summary import build_prompt, digest_matters, render_digest


def make(email_id, day, subject, sender="a@example.com", recipient="b@example.org"):
    return Email(email_id, sender, recipient, datetime(2024, 3, day, 9), subject, "Body text.")


@pytest.fixture
def small_matter():
    emails = [
        make(2, 5, "Re: Lease Review", sender="b@example.org", recipient="a@example.com"),
        make(1, 1, "Lease Review"),
        make(3, 9, "Fwd: Lease Review", recipient="c@example.net"),
    ]
    return digest_matters(emails, [0, 0, 0])[0]


def test_digest_orders_emails_and_uses_bare_subject(small_matter):
    assert [e.email_id for e in small_matter.emails] == [1, 2, 3]
    assert small_matter.subject == "Lease Review"
    assert small_matter.first == datetime(2024, 3, 1, 9)
    assert small_matter.last == datetime(2024, 3, 9, 9)


def test_participants_include_senders_not_just_recipients(small_matter):
    assert small_matter.participants == ["a@example.com", "b@example.org", "c@example.net"]


def test_matters_are_ordered_by_first_activity():
    emails = [make(1, 9, "Late"), make(2, 1, "Early")]
    digests = digest_matters(emails, [5, 3])
    assert [d.label for d in digests] == [3, 5]


def test_subject_tie_breaks_alphabetically():
    emails = [make(1, 1, "Zeta"), make(2, 2, "Alpha")]
    assert digest_matters(emails, [0, 0])[0].subject == "Alpha"


def test_label_length_mismatch():
    with pytest.raises(ValueError):
        digest_matters([make(1, 1, "X")], [0, 1])


def test_render_digest_lists_every_email(small_matter):
    text = render_digest(small_matter)
    assert text.startswith("Matter 0: Lease Review\n  3 emails, 2024-03-01 to 2024-03-09")
    assert text.count(" -> ") == 3


def test_prompt_has_the_five_sections_and_no_invented_terms(small_matter):
    prompt = build_prompt(small_matter)
    for section in (
        "Recap of instructions",
        "Recipient details",
        "Scope of work",
        "Ongoing tasks",
        "Chronology of events",
    ):
        assert section in prompt
    assert "Not stated in the correspondence" in prompt
    assert "fees applied per consultation" not in prompt
    assert prompt.count("Subject: ") == 3


def test_digests_cover_every_synthetic_email(emails):
    labels = group_emails(emails).labels
    digests = digest_matters(emails, labels)
    assert len(digests) == 4
    assert sum(len(d.emails) for d in digests) == len(emails)
