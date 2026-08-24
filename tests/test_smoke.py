"""Core smoke tests that run without any embedding framework installed."""
import pytest
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
        return EncodedText(embeddings=torch.eye(n), word_ids=list(range(len(words))),
                           tokens=words or [""])


class _NoTokensEncoder(_DummyEncoder):
    """An encoder that supplies no token surface strings."""

    def encode(self, text):
        enc = super().encode(text)
        enc.tokens = None
        return enc


class _SubwordEncoder:
    """Two subword tokens per whitespace word, so words != tokens."""

    def encode(self, text):
        words = text.split()
        n_tok = max(2 * len(words), 1)
        return EncodedText(
            embeddings=torch.eye(n_tok),
            word_ids=[i // 2 for i in range(n_tok)],
            tokens=[f"{w}#{h}" for w in words for h in (0, 1)] or [""],
        )


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


def test_units_words_is_the_default():
    aligner = WordAligner(_SubwordEncoder(), method="simalign", mode="argmax")
    assert aligner.units == "words"
    # 2 tokens per word collapse to one pair per word.
    assert aligner.align("a b", "x y") == [(0, 0), (1, 1)]


def test_units_tokens_skips_word_projection():
    aligner = WordAligner(_SubwordEncoder(), method="simalign", mode="argmax",
                          units="tokens")
    pairs, units_a, units_b = aligner.align_with_units("a b", "x y")
    assert pairs == [(0, 0), (1, 1), (2, 2), (3, 3)]        # no collapsing
    assert units_a == ["a#0", "a#1", "b#0", "b#1"]
    assert units_b == ["x#0", "x#1", "y#0", "y#1"]


def test_align_with_units_words_returns_whitespace_words():
    aligner = WordAligner(_DummyEncoder(), method="simalign", mode="argmax")
    pairs, units_a, units_b = aligner.align_with_units("the cat", "le chat")
    assert pairs == [(0, 0), (1, 1)]
    assert (units_a, units_b) == (["the", "cat"], ["le", "chat"])


def test_align_pairs_with_units():
    aligner = WordAligner(_SubwordEncoder(), method="simalign", mode="argmax",
                          units="tokens")
    results = aligner.align_pairs([("a b", "x y"), ("", "x")], with_units=True)
    assert results[0][1] == ["a#0", "a#1", "b#0", "b#1"]
    assert results[1] == ([], [], [])             # blank input, units shape kept


def test_units_tokens_requires_token_strings():
    aligner = WordAligner(_NoTokensEncoder(), method="simalign", mode="argmax",
                          units="tokens")
    with pytest.raises(ValueError, match="tokens=None"):
        aligner.align("a b", "x y")


def test_unknown_units_rejected():
    with pytest.raises(ValueError, match="Unknown units"):
        WordAligner(_DummyEncoder(), units="subwords")


def test_encoded_text_rejects_mismatched_tokens():
    with pytest.raises(ValueError, match="tokens length"):
        EncodedText(embeddings=torch.eye(3), word_ids=[0, 1, 2], tokens=["a", "b"])


def test_alignment_record_carries_units():
    rec = alignment_record("ab", "xy", [(0, 0)], fmt="string",
                           units_a=["a", "b"], units_b=["x", "y"])
    assert rec["units_a"] == ["a", "b"] and rec["units_b"] == ["x", "y"]
    # omitted by default, since whitespace words are recoverable from the text
    assert "units_a" not in alignment_record("ab", "xy", [(0, 0)])
