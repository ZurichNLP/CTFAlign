"""Core smoke tests that run without any embedding framework installed."""
import torch

from ctfalign import (
    WordAligner,
    align_from_embeddings,
    alignment_record,
    to_word_alignment_label_notation,
)
from ctfalign.encoders import EncodedText


def _toy_embeddings():
    # 3 source words, 3 target words, one token each; identity-ish similarity.
    emb_a = torch.eye(3)
    emb_b = torch.eye(3)
    word_ids = [0, 1, 2]
    return emb_a, emb_b, word_ids


def test_simalign_diagonal():
    emb_a, emb_b, w = _toy_embeddings()
    pairs = align_from_embeddings(emb_a, emb_b, w, w, method="simalign", mode="argmax")
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_all_methods_run():
    emb_a, emb_b, w = _toy_embeddings()
    for method in ("simalign", "mdpalign-strict", "mdpalign-fuzzy", "ctfalign"):
        pairs = align_from_embeddings(emb_a, emb_b, w, w, method=method)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


def test_subword_projection():
    # 4 tokens -> 2 words on each side; alignment should collapse to word pairs.
    emb_a = torch.eye(4)
    emb_b = torch.eye(4)
    word_ids = [0, 0, 1, 1]
    pairs = align_from_embeddings(emb_a, emb_b, word_ids, word_ids,
                                  method="simalign", mode="argmax")
    assert pairs == [(0, 0), (1, 1)]


class _DummyEncoder:
    """One identity token per whitespace word; no framework needed."""

    def encode(self, text):
        words = text.split()
        n = max(len(words), 1)
        return EncodedText(embeddings=torch.eye(n), word_ids=list(range(len(words))))


def test_align_pairs_batch():
    aligner = WordAligner(_DummyEncoder(), method="simalign", mode="argmax")
    results = aligner.align_pairs([("a b c", "x y z"), ("", "x"), ("p q", "r s")])
    assert results[0] == [(0, 0), (1, 1), (2, 2)]
    assert results[1] == []                       # blank input -> empty alignment
    assert results[2] == [(0, 0), (1, 1)]


def test_to_word_alignment_label_notation():
    assert to_word_alignment_label_notation([(0, 0), (1, 1), (2, 3)]) == "0-0 1-1 2-3"
    assert to_word_alignment_label_notation([]) == ""


def test_alignment_record():
    pairs = [(0, 0), (1, 1)]
    rec = alignment_record("the cat", "le chat", pairs, fmt="string")
    assert rec == {"text_a": "the cat", "text_b": "le chat", "labels": "0-0 1-1"}
    rec_pairs = alignment_record("the cat", "le chat", pairs, fmt="pairs")
    assert rec_pairs["labels"] == [[0, 0], [1, 1]]
    # blank pair still yields a record with empty labels
    assert alignment_record("", "x", [], fmt="string")["labels"] == ""
