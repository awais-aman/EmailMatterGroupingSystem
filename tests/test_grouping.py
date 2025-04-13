import numpy as np
import pytest

from matter_grouping import group_emails, grouping_scores, jaccard_similarity, make_embedder
from matter_grouping.grouping import (
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    cluster_vectors,
    email_features,
)


def truth(emails):
    return [e.true_matter_id for e in emails]


def test_kmeans_recovers_the_four_matters(emails):
    result = group_emails(emails, clusters=4)
    scores = grouping_scores(truth(emails), result.labels)
    assert result.clusters == 4
    assert scores == pytest.approx({"ari": 1.0, "nmi": 1.0})


def test_threshold_mode_finds_four_matters_without_being_told(emails):
    result = group_emails(emails)
    assert result.clusters == 4
    assert grouping_scores(truth(emails), result.labels)["ari"] == pytest.approx(1.0)
    assert result.silhouette > 0.3


@pytest.mark.parametrize("seed", [1, 2, 3, 11, 42])
def test_grouping_holds_across_generator_seeds(seed):
    from matter_grouping import generate_emails

    emails = generate_emails(count=40, replies=6, seed=seed)
    result = group_emails(emails)
    assert grouping_scores(truth(emails), result.labels)["ari"] == pytest.approx(1.0)


def test_replies_and_forwards_land_with_their_matter(emails):
    labels = group_emails(emails).labels
    by_matter = {}
    for email, label in zip(emails, labels, strict=True):
        by_matter.setdefault(email.true_matter_id, set()).add(label)
    assert all(len(found) == 1 for found in by_matter.values())


def test_grouping_is_deterministic(emails):
    assert group_emails(emails, clusters=4).labels == group_emails(emails, clusters=4).labels


def test_features_are_unit_length(emails):
    vectors = email_features(emails, TfidfEmbedder()).toarray()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_subject_weight_is_bounded(emails):
    with pytest.raises(ValueError):
        email_features(emails, TfidfEmbedder(), subject_weight=1.5)


def test_stop_word_only_text_does_not_crash():
    vectors = TfidfEmbedder().embed(["the", "and", ""])
    assert vectors.shape == (3, 1)


def test_empty_input():
    result = group_emails([])
    assert (result.labels, result.clusters, result.silhouette) == ([], 0, None)


def test_single_email_is_one_matter(emails):
    assert group_emails(emails[:1]).labels == [0]


def test_threshold_separates_orthogonal_and_merges_identical():
    assert cluster_vectors(np.array([[1.0, 0.0], [0.0, 1.0]])).clusters == 2
    assert cluster_vectors(np.array([[1.0, 0.0], [1.0, 0.0]])).clusters == 1


def test_explicit_cluster_count_is_bounded():
    with pytest.raises(ValueError):
        cluster_vectors(np.eye(3), clusters=4)


def test_make_embedder():
    assert isinstance(make_embedder("tfidf"), TfidfEmbedder)
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("bag-of-words")


class FakeSentenceModel:
    """Maps each matter keyword to its own axis, standing in for SBERT."""

    AXES = ("litigation", "contract", "patent", "property")

    def encode(self, texts, **_):
        rows = [[float(axis in t) for axis in self.AXES] + [0.1] for t in texts]
        return [np.array(r) / np.linalg.norm(r) for r in rows]


def test_sentence_transformer_embedder_uses_injected_model(emails):
    embedder = SentenceTransformerEmbedder(model=FakeSentenceModel())
    result = group_emails(emails, clusters=4, embedder=embedder)
    assert grouping_scores(truth(emails), result.labels)["ari"] == pytest.approx(1.0)


def test_grouping_scores_ignore_label_names():
    assert grouping_scores([1, 1, 2, 2], [7, 7, 3, 3]) == {"ari": 1.0, "nmi": 1.0}
    with pytest.raises(ValueError):
        grouping_scores([1, 2], [1])


def test_jaccard_similarity():
    assert jaccard_similarity("The matter", "the MATTER") == 1.0
    assert jaccard_similarity("a b", "b c") == pytest.approx(1 / 3)
    assert jaccard_similarity("", "") == 0.0
