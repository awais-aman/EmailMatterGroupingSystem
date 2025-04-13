"""Group legal emails into matters and summarise each matter."""

from .emails import Email, clean_text, normalize_subject, read_jsonl, write_jsonl
from .grouping import Grouping, group_emails, make_embedder
from .metrics import grouping_scores, jaccard_similarity
from .synthetic import generate_emails

__version__ = "0.2.0"

__all__ = [
    "Email",
    "Grouping",
    "clean_text",
    "generate_emails",
    "group_emails",
    "grouping_scores",
    "jaccard_similarity",
    "make_embedder",
    "normalize_subject",
    "read_jsonl",
    "write_jsonl",
]
