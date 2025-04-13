import io
import json

import pytest

from matter_grouping.cli import main


@pytest.fixture
def emails_file(tmp_path):
    path = tmp_path / "emails.jsonl"
    assert main(["generate", "--seed", "7", "-o", str(path)]) == 0
    return path


def test_generate_to_stdout_is_json_lines(capsys):
    main(["generate", "--count", "5", "--replies", "1", "--seed", "1"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6
    assert {"email_id", "sender", "recipient", "subject"} <= json.loads(lines[0]).keys()


def test_generate_to_file_writes_every_email(emails_file):
    assert len(emails_file.read_text().splitlines()) == 28


def test_group_prints_matters_and_scores(emails_file, capsys):
    assert main(["group", str(emails_file)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("28 emails -> 4 matters\n")
    assert "ARI: 1.000  NMI: 1.000" in out


def test_group_json(emails_file, capsys):
    main(["group", str(emails_file), "--clusters", "4", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result["clusters"] == 4
    assert result["scores"] == {"ari": 1.0, "nmi": 1.0}
    assert len(result["assignments"]) == 28


def test_group_reads_stdin(emails_file, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(emails_file.read_text()))
    main(["group", "-"])
    assert "4 matters" in capsys.readouterr().out


def test_scores_skipped_without_ground_truth(tmp_path, emails_file, capsys):
    unlabeled = tmp_path / "unlabeled.jsonl"
    rows = [json.loads(line) for line in emails_file.read_text().splitlines()]
    for row in rows:
        row.pop("true_matter_id")
    unlabeled.write_text("\n".join(json.dumps(r) for r in rows))
    main(["group", str(unlabeled)])
    assert "ARI" not in capsys.readouterr().out


def test_summarize_digest_and_prompt(emails_file, capsys):
    main(["summarize", str(emails_file)])
    digest = capsys.readouterr().out
    assert digest.count("Matter ") == 4
    assert "Participants" in digest

    main(["summarize", str(emails_file), "--prompt"])
    prompts = capsys.readouterr().out
    assert prompts.count("Recap of instructions") == 4


def test_bad_cluster_count_is_a_clean_error(emails_file, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["group", str(emails_file), "--clusters", "99"])
    assert exc.value.code == 2
    assert "clusters must be between 1 and 28" in capsys.readouterr().err


def test_missing_file_is_a_clean_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["group", str(tmp_path / "nope.jsonl")])
    assert exc.value.code == 2
    assert "No such file" in capsys.readouterr().err
