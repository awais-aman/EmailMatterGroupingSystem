"""Embed emails and cluster them into matters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .emails import Email, clean_text, normalize_subject

SBERT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SUBJECT_WEIGHT = 0.8
DEFAULT_DISTANCE_THRESHOLD = 0.5


class Embedder(Protocol):
    """Turns texts into one L2-normalised row per text (dense or sparse)."""

    def embed(self, texts: Sequence[str]): ...


class TfidfEmbedder:
    """Offline default: TF-IDF over word unigrams and bigrams.

    Returns a sparse matrix, so large mailboxes are never densified for KMeans.
    """

    def embed(self, texts: Sequence[str]):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            return vectorizer.fit_transform(texts)
        except ValueError:
            # Every text was empty or stop words only: no vocabulary to weight.
            return sparse.csr_matrix((len(texts), 1))


class SentenceTransformerEmbedder:
    """Sentence-BERT embeddings, as in the original notebook.

    Needs the optional `sbert` extra (sentence-transformers pulls in PyTorch)
    and downloads the model on first use.
    """

    def __init__(self, model_name: str = SBERT_MODEL, model=None):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("the sbert embedder needs: pip install '.[sbert]'") from exc
            model = SentenceTransformer(model_name)
        self._model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        )


EMBEDDERS = {"tfidf": TfidfEmbedder, "sbert": SentenceTransformerEmbedder}


def make_embedder(name: str) -> Embedder:
    try:
        factory = EMBEDDERS[name]
    except KeyError:
        raise ValueError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}") from None
    return factory()


def email_features(
    emails: Sequence[Email],
    embedder: Embedder,
    subject_weight: float = DEFAULT_SUBJECT_WEIGHT,
):
    """Subject and body embedded separately, then joined with fixed weights.

    A matter is held together mostly by its subject line (replies and
    forwards keep it, behind a prefix we strip), while body wording varies a
    lot inside one matter. Embedding the two apart and weighting the subject
    stops KMeans from splitting a matter by body template. The weights keep
    each row at unit length.
    """
    if not 0.0 <= subject_weight <= 1.0:
        raise ValueError("subject_weight must be between 0 and 1")
    body_weight = float(np.sqrt(1.0 - subject_weight**2))
    subjects = embedder.embed([clean_text(normalize_subject(e.subject)) for e in emails])
    bodies = embedder.embed([clean_text(e.body) for e in emails])
    if sparse.issparse(subjects) or sparse.issparse(bodies):
        return sparse.hstack([subjects * subject_weight, bodies * body_weight]).tocsr()
    return np.hstack([subjects * subject_weight, bodies * body_weight])


@dataclass
class Grouping:
    labels: list[int]
    clusters: int
    silhouette: float | None


def _silhouette(vectors, labels: np.ndarray) -> float | None:
    distinct = len(set(labels.tolist()))
    if distinct < 2 or distinct >= len(labels):
        return None
    return float(silhouette_score(vectors, labels, metric="cosine"))


def cluster_vectors(
    vectors,
    clusters: int | None = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    seed: int = 42,
) -> Grouping:
    """KMeans when the matter count is known, otherwise a distance threshold.

    Without a count, average-linkage agglomerative clustering merges emails
    until the cosine distance between groups passes `distance_threshold`.
    Picking k by best silhouette was tried and dropped: it rewards splitting
    near-duplicate emails, and on the seed-7 synthetic mailbox it peaks at
    k=5 rather than the true 4.
    """
    n = vectors.shape[0]
    if n == 0:
        return Grouping(labels=[], clusters=0, silhouette=None)
    if clusters is not None:
        if not 1 <= clusters <= n:
            raise ValueError(f"clusters must be between 1 and {n}")
        labels = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit_predict(vectors)
    elif n == 1:
        labels = np.zeros(1, dtype=int)
    else:
        dense = vectors.toarray() if sparse.issparse(vectors) else np.asarray(vectors)
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        ).fit_predict(dense)
    return Grouping(labels.tolist(), len(set(labels.tolist())), _silhouette(vectors, labels))


def group_emails(
    emails: Sequence[Email],
    clusters: int | None = None,
    embedder: Embedder | None = None,
    subject_weight: float = DEFAULT_SUBJECT_WEIGHT,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    seed: int = 42,
) -> Grouping:
    if not emails:
        return Grouping(labels=[], clusters=0, silhouette=None)
    vectors = email_features(emails, embedder or TfidfEmbedder(), subject_weight)
    return cluster_vectors(vectors, clusters, distance_threshold, seed)
