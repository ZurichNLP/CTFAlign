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
from ctfalign.align import _onehot_rows, iter_max
from ctfalign.masks import _level_is_inert, ctfalign


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


def _reference_ctfalign(sim, method, width):
    """CTFAlign with every level run, including the ones ctfalign skips.

    ``_level_is_inert`` and the all-ones check are pure efficiency shortcuts, so
    forcing every level must give the identical alignment.
    """
    import ctfalign.masks as masks
    inert, saved = masks._level_is_inert, masks._level_is_inert
    masks._level_is_inert = lambda h, w, width: False
    try:
        return masks.ctfalign(sim, method, width=width)
    finally:
        masks._level_is_inert = saved


def test_level_is_inert_condition():
    assert _level_is_inert(2, 2, width=8)        # buffer spans the whole grid
    assert _level_is_inert(9, 9, width=8)
    assert not _level_is_inert(10, 10, width=8)  # first level that can constrain
    assert not _level_is_inert(2, 2, width=0)    # no buffer -> never inert
    assert _level_is_inert(3, 17, width=16)      # driven by the longer side


def test_skipping_inert_levels_does_not_change_the_alignment():
    torch.manual_seed(0)
    for m, n in ((9, 9), (40, 33), (128, 96)):
        sim = torch.randn(m, n) * 0.4
        for width in (0, 1, 4, 8, 32):
            for mode in ("simalign-argmax", "simalign-itermax"):
                fast = sorted(map(tuple, ctfalign(sim, mode, width=width)))
                full = sorted(map(tuple, _reference_ctfalign(sim, mode, width)))
                assert fast == full, (m, n, width, mode)


def test_onehot_rows_matches_eye_indexing():
    """_onehot_rows replaces np.eye(n)[idx] without materialising the identity."""
    import numpy as np
    rng = np.random.default_rng(0)
    for m, n in ((1, 1), (3, 7), (7, 3), (40, 40)):
        idx = rng.integers(0, n, m)
        assert (_onehot_rows(idx, n) == np.eye(n)[idx]).all()


def test_itermax_early_exit_matches_running_the_iteration_out():
    """The 'everything already aligned' break must not change the result.

    A near-permutation matrix makes the first intersection cover every row and
    column, which is exactly the branch that used to zero the masks and run the
    iteration out on all-zero arrays.
    """
    import numpy as np
    rng = np.random.default_rng(1)
    for size in (5, 20, 60):
        sim = rng.normal(0, 0.05, (size, size))
        for i, j in enumerate(rng.permutation(size)):
            sim[i, j] += 10.0
        one_pass = sorted(map(tuple, iter_max(sim, max_count=1)))
        many_pass = sorted(map(tuple, iter_max(sim, max_count=10)))
        # extra iterations find nothing once everything is aligned
        assert one_pass == many_pass
