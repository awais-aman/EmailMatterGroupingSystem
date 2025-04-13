import pytest

from matter_grouping import clean_text, normalize_subject, read_jsonl, write_jsonl


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("Litigation Case Update", "Litigation Case Update"),
        ("Re: Litigation Case Update", "Litigation Case Update"),
        ("RE: fwd: Re:Litigation Case Update", "Litigation Case Update"),
        ("FW: Property Dispute", "Property Dispute"),
        ("Regarding the lease", "Regarding the lease"),
    ],
)
def test_normalize_subject_strips_stacked_prefixes(subject, expected):
    assert normalize_subject(subject) == expected


def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  Please\n review   the FILE. ") == "please review the file."


def test_jsonl_round_trip(tmp_path, emails):
    path = tmp_path / "emails.jsonl"
    write_jsonl(emails, path)
    assert read_jsonl(path) == emails


def test_read_jsonl_skips_blank_lines(tmp_path, emails):
    path = tmp_path / "emails.jsonl"
    write_jsonl(emails[:2], path)
    path.write_text(path.read_text() + "\n\n")
    assert len(read_jsonl(path)) == 2
