"""Public API: the low-level ``align_from_embeddings`` and the high-level
``WordAligner`` that ties an encoder to the alignment pipeline.

Algorithms:
    ctfalign         - coarse-to-fine constraint (recommended for documents)
    mdpalign-strict  - hard diagonal-band constraint
    mdpalign-fuzzy   - soft (Gaussian) diagonal-band constraint
    simalign         - no positional constraint (base SimAlign)
``mode`` selects the base SimAlign variant: "argmax" (default) or "itermax".
"""
import sys
import warnings

import torch

from .align import argmax_align, iter_max
from .defaults import resolve_k, resolve_layer
from .masks import (
    add_belt_mask_token_count_fixed_on_longer_side,
    add_soft_belt_mask_token_count_fixed_on_longer_side,
    pyramid_hard_lenient2,
)
from .similarity import get_sim_matrix
from .wordmap import project_token_alignment

_METHOD_ALIASES = {
    "simalign": "simalign",
    "mdpalign-strict": "mdpalign-strict",
    "mdpalign_strict": "mdpalign-strict",
    "mdpalign-fuzzy": "mdpalign-fuzzy",
    "mdpalign_fuzzy": "mdpalign-fuzzy",
    "ctfalign": "ctfalign",
}
METHODS = ("simalign", "mdpalign-strict", "mdpalign-fuzzy", "ctfalign")


def _normalize_method(method):
    key = method.strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
    return _METHOD_ALIASES[key]


def _print_progress(idx, total):
    """Default ``progress_callback`` for ``WordAligner.align_pairs``: prints to stderr."""
    end = "\n" if idx == total else ""
    print(f"\r  aligned {idx}/{total} pairs", end=end, file=sys.stderr, flush=True)


def align_from_similarity(sim, word_ids_a, word_ids_b,
                          method="ctfalign", mode="argmax", k=None, max_count=2):
    """Align a pre-computed token similarity matrix to word-level pairs.

    sim             : ``(n_tokens_a, n_tokens_b)`` similarity matrix (array-like).
                      Used as-is: negative entries stay negative (they are
                      clamped only within a pooled CTFAlign block, see
                      ``masks.py``) and never license an alignment.
    word_ids_a / b  : per-token whitespace-word indices (b2w maps).
    Returns a sorted list of ``(word_idx_a, word_idx_b)`` pairs.
    """
    method = _normalize_method(method)
    k = resolve_k(method, k)

    sim = torch.as_tensor(sim, dtype=torch.float32)

    if method == "mdpalign-strict":
        sim = add_belt_mask_token_count_fixed_on_longer_side(sim, k)
    elif method == "mdpalign-fuzzy":
        sim = add_soft_belt_mask_token_count_fixed_on_longer_side(sim, k)

    if method == "ctfalign":
        token_alignment = pyramid_hard_lenient2(sim, f"simalign-{mode}", width=int(k))
    elif mode == "argmax":
        token_alignment = argmax_align(sim, drop_negative=True)
    elif mode == "itermax":
        token_alignment = iter_max(sim.numpy(), max_count=max_count,
                                   drop_negative=True)
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'argmax' or 'itermax'.")

    return project_token_alignment(token_alignment, word_ids_a, word_ids_b)


def align_from_embeddings(emb_a, emb_b, word_ids_a, word_ids_b,
                          method="ctfalign", mode="argmax", k=None, max_count=2):
    """Align two pre-computed token-embedding matrices to word-level pairs.

    emb_a / emb_b   : ``(n_tokens, hidden)`` tensors (or array-likes).
    word_ids_a / b  : per-token whitespace-word indices (b2w maps).
    Returns a sorted list of ``(word_idx_a, word_idx_b)`` pairs.
    """
    emb_a = torch.as_tensor(emb_a, dtype=torch.float32)
    emb_b = torch.as_tensor(emb_b, dtype=torch.float32)
    sim = get_sim_matrix(emb_a, emb_b)
    return align_from_similarity(
        sim, word_ids_a, word_ids_b,
        method=method, mode=mode, k=k, max_count=max_count,
    )


class WordAligner:
    """High-level aligner: encode two texts and return word-level alignments."""

    def __init__(self, encoder, method="ctfalign", mode="argmax", k=None, max_count=2):
        self.encoder = encoder
        self.method = _normalize_method(method)
        self.mode = mode
        self.k = resolve_k(self.method, k)
        self.max_count = max_count

    @classmethod
    def from_huggingface(cls, model_name, layer=None, lang_pair=None, granularity="documents",
                         method="ctfalign", mode="argmax", k=None, max_count=2,
                         device=None, token=None, **encoder_kwargs):
        """Build an aligner backed by a HuggingFace model.

        ``layer`` defaults to the empirically-best layer for the model (and
        language pair, if known), taken from the document-level table by default
        (``granularity="documents"``); otherwise a depth-relative heuristic with
        a warning.
        """
        from .encoders.huggingface import HFEncoder

        short = model_name.split("/")[-1]
        if layer is None:
            layer = resolve_layer(short, lang_pair, granularity)
        if layer is None:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, token=token)
            n_layers = getattr(cfg, "num_hidden_layers", None)
            if n_layers is None:
                raise ValueError(
                    f"No tuned layer default for {short!r} and could not infer "
                    f"layer count; pass layer=... explicitly."
                )
            layer = round((2 / 3) * n_layers)
            warnings.warn(
                f"No tuned layer for {short!r}; using depth-relative default "
                f"layer {layer} (of {n_layers}). Consider tuning on a dev set."
            )

        encoder = HFEncoder(model_name, layer, device=device, token=token, **encoder_kwargs)
        return cls(encoder, method=method, mode=mode, k=k, max_count=max_count)

    def align(self, text_a, text_b):
        """Return a sorted list of ``(word_idx_a, word_idx_b)`` alignment pairs."""
        a = self.encoder.encode(text_a)
        b = self.encoder.encode(text_b)
        return align_from_embeddings(
            a.embeddings, b.embeddings, a.word_ids, b.word_ids,
            method=self.method, mode=self.mode, k=self.k, max_count=self.max_count,
        )

    def align_pairs(self, pairs, show_progress=False, progress_callback=None):
        """Align many text pairs.

        ``pairs``: iterable of ``(text_a, text_b)`` tuples.
        Returns a list of per-pair alignments, in input order. Empty/blank texts
        yield an empty alignment so positional correspondence is preserved.

        ``progress_callback``, if given, is called as ``callback(idx, total)``
        after each pair (1-indexed). ``show_progress=True`` installs a default
        callback that prints ``idx/total`` to stderr; pass your own
        ``progress_callback`` instead for other UIs (e.g. a notebook or a
        different CLI's progress bar).
        """
        pairs = list(pairs)
        total = len(pairs)
        if progress_callback is None and show_progress:
            progress_callback = _print_progress

        results = []
        for idx, (text_a, text_b) in enumerate(pairs, 1):
            if not text_a.strip() or not text_b.strip():
                results.append([])
            else:
                results.append(self.align(text_a, text_b))
            if progress_callback:
                progress_callback(idx, total)
        return results
