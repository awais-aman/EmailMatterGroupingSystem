from datetime import datetime

import pytest

from matter_grouping import generate_emails
from matter_grouping.synthetic import DEFAULT_END, MATTER_TEMPLATES, REPLY_BODY

SAFE_DOMAINS = ("@example.com", "@example.net", "@example.org")


def test_count_includes_replies(emails):
    assert len(emails) == 28
    assert sum(e.body == REPLY_BODY for e in emails) == 3


def test_ids_are_unique(emails):
    assert sorted(e.email_id for e in emails) == list(range(1, 29))


def test_same_seed_gives_same_emails():
    assert generate_emails(seed=3) == generate_emails(seed=3)


def test_different_seed_changes_output():
    assert generate_emails(seed=3) != generate_emails(seed=4)


def test_sorted_oldest_first(emails):
    stamps = [e.timestamp for e in emails]
    assert stamps == sorted(stamps)


def test_timestamps_stay_in_the_year_before_the_anchor(emails):
    originals = [e for e in emails if e.body != REPLY_BODY]
    assert all(datetime(2024, 1, 1) <= e.timestamp <= DEFAULT_END for e in originals)


def test_addresses_use_reserved_example_domains(emails):
    for email in emails:
        assert email.sender.endswith(SAFE_DOMAINS)
        assert email.recipient.endswith(SAFE_DOMAINS)


def test_subjects_match_their_matter(emails):
    for email in emails:
        assert MATTER_TEMPLATES[email.true_matter_id]["subject"] in email.subject


def test_replies_swap_sender_and_recipient(emails):
    by_pair = {(e.sender, e.recipient) for e in emails if e.body != REPLY_BODY}
    for reply in (e for e in emails if e.body == REPLY_BODY):
        assert (reply.recipient, reply.sender) in by_pair
        assert reply.subject.startswith("Re: ")
        assert not reply.subject.startswith("Re: Re: ")


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"replies": -1}])
def test_rejects_bad_sizes(kwargs):
    with pytest.raises(ValueError):
        generate_emails(**kwargs)
