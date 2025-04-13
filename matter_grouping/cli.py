"""Command line: generate synthetic emails, group them, summarise the matters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .emails import Email, read_jsonl, write_jsonl
from .grouping import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_SUBJECT_WEIGHT,
    EMBEDDERS,
    Grouping,
    group_emails,
    make_embedder,
)
from .metrics import grouping_scores
from .summary import build_prompt, digest_matters, render_digest
from .synthetic import generate_emails


def _load(source: str) -> list[Email]:
    if source == "-":
        return [Email.from_json(json.loads(line)) for line in sys.stdin if line.strip()]
    return read_jsonl(Path(source))


def _group(args: argparse.Namespace, emails: list[Email]) -> Grouping:
    return group_emails(
        emails,
        clusters=args.clusters,
        embedder=make_embedder(args.embedder),
        subject_weight=args.subject_weight,
        distance_threshold=args.threshold,
    )


def _scores(emails: list[Email], grouping: Grouping) -> dict[str, float] | None:
    truth = [e.true_matter_id for e in emails]
    if not emails or any(t is None for t in truth):
        return None
    return grouping_scores(truth, grouping.labels)


def cmd_generate(args: argparse.Namespace) -> int:
    emails = generate_emails(count=args.count, replies=args.replies, seed=args.seed)
    if args.output:
        write_jsonl(emails, Path(args.output))
        print(f"wrote {len(emails)} emails to {args.output}", file=sys.stderr)
    else:
        for email in emails:
            print(json.dumps(email.to_json()))
    return 0


def cmd_group(args: argparse.Namespace) -> int:
    emails = _load(args.input)
    grouping = _group(args, emails)
    scores = _scores(emails, grouping)
    if args.json:
        print(
            json.dumps(
                {
                    "clusters": grouping.clusters,
                    "silhouette": grouping.silhouette,
                    "scores": scores,
                    "assignments": [
                        {"email_id": e.email_id, "matter": label}
                        for e, label in zip(emails, grouping.labels, strict=True)
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(f"{len(emails)} emails -> {grouping.clusters} matters")
    for digest in digest_matters(emails, grouping.labels):
        ids = ", ".join(str(e.email_id) for e in digest.emails)
        print(f"  matter {digest.label}: {digest.subject} ({len(digest.emails)}) [{ids}]")
    if grouping.silhouette is not None:
        print(f"silhouette (cosine): {grouping.silhouette:.3f}")
    if scores:
        print(f"ARI: {scores['ari']:.3f}  NMI: {scores['nmi']:.3f}  (against true_matter_id)")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    emails = _load(args.input)
    grouping = _group(args, emails)
    blocks = [
        build_prompt(d) if args.prompt else render_digest(d)
        for d in digest_matters(emails, grouping.labels)
    ]
    separator = "\n" + "-" * 72 + "\n" if args.prompt else "\n\n"
    print(separator.join(blocks))
    return 0


def _add_grouping_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="JSON Lines file of emails, or - for stdin")
    parser.add_argument(
        "--clusters",
        type=int,
        help="number of matters if known (KMeans); omit to group by distance threshold",
    )
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default="tfidf")
    parser.add_argument(
        "--subject-weight",
        type=float,
        default=DEFAULT_SUBJECT_WEIGHT,
        help="share of each email vector given to the subject line (default %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help="cosine distance that separates matters when --clusters is omitted "
        "(default %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matter-grouping", description="Group legal emails into matters."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write seeded synthetic legal emails as JSON Lines")
    gen.add_argument("--count", type=int, default=25, help="emails before replies")
    gen.add_argument("--replies", type=int, default=3, help="threaded replies to add")
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("-o", "--output", help="file to write (default: stdout)")
    gen.set_defaults(func=cmd_generate)

    grp = sub.add_parser("group", help="assign each email to a matter")
    _add_grouping_options(grp)
    grp.add_argument("--json", action="store_true", help="machine-readable output")
    grp.set_defaults(func=cmd_group)

    summ = sub.add_parser("summarize", help="print a digest (or an LLM prompt) per matter")
    _add_grouping_options(summ)
    summ.add_argument("--prompt", action="store_true", help="print LLM prompts instead")
    summ.set_defaults(func=cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"matter-grouping: error: {exc}\n")
