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
    apply_mdpalign_strict,
    apply_mdpalign_fuzzy,
    ctfalign,
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
UNITS = ("words", "tokens")


def _normalize_method(method):
    key = method.strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
    return _METHOD_ALIASES[key]


def _normalize_units(units):
    if units not in UNITS:
        raise ValueError(f"Unknown units {units!r}; choose from {UNITS}.")
    return units


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
        sim = apply_mdpalign_strict(sim, k)
    elif method == "mdpalign-fuzzy":
        sim = apply_mdpalign_fuzzy(sim, k)

    if method == "ctfalign":
        token_alignment = ctfalign(sim, f"simalign-{mode}", width=int(k))
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
    """High-level aligner: encode two texts and return their alignment.

    ``units`` selects what the returned indices point at:

    ``"words"`` (default)
        Whitespace-split words -- the setting used throughout the paper. Texts in
        languages without whitespace word boundaries (zh, ja, ...) must be
        pre-segmented, as they were for the reported experiments.
    ``"tokens"``
        The encoder's own subword tokens, i.e. no projection to words at all.
        Use this when the input is not whitespace-segmented and pre-segmenting is
        not an option. The indices then point into ``EncodedText.tokens``, which
        ``align_with_units`` returns alongside the pairs.

    The choice is explicit rather than inferred from the script: guessing would
    silently change the meaning of the output, including for the pre-segmented
    zh/ja inputs the paper's numbers are based on.
    """

    def __init__(self, encoder, method="ctfalign", mode="argmax", k=None, max_count=2,
                 units="words"):
        self.encoder = encoder
        self.method = _normalize_method(method)
        self.mode = mode
        self.k = resolve_k(self.method, k)
        self.max_count = max_count
        self.units = _normalize_units(units)

    @classmethod
    def from_huggingface(cls, model_name, layer=None, lang_pair=None, granularity="documents",
                         method="ctfalign", mode="argmax", k=None, max_count=2,
                         units="words", device=None, token=None, **encoder_kwargs):
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
        return cls(encoder, method=method, mode=mode, k=k, max_count=max_count,
                   units=units)

    def _units_and_ids(self, text, encoded):
        """The display units for one side and the b2w map to project onto them.

        In ``"tokens"`` mode the map is the identity, so the token-level
        alignment is returned as-is with no projection.
        """
        if self.units == "words":
            return text.split(), encoded.word_ids
        if encoded.tokens is None:
            raise ValueError(
                f"units='tokens' needs token surface strings, but "
                f"{type(self.encoder).__name__} returned EncodedText(tokens=None). "
                f"Populate EncodedText.tokens in your encoder, or use units='words'."
            )
        return list(encoded.tokens), list(range(len(encoded.tokens)))

    def align(self, text_a, text_b):
        """Return a sorted list of ``(idx_a, idx_b)`` alignment pairs.

        Indices refer to whitespace words or to encoder tokens, per ``units``.
        """
        return self.align_with_units(text_a, text_b)[0]

    def align_with_units(self, text_a, text_b):
        """Return ``(pairs, units_a, units_b)`` from a single encoding pass.

        ``units_a`` / ``units_b`` are the strings the pair indices point at, so
        the alignment is readable without re-deriving the segmentation. Useful in
        ``units="tokens"`` mode, where the indices are otherwise opaque.
        """
        a = self.encoder.encode(text_a)
        b = self.encoder.encode(text_b)
        units_a, word_ids_a = self._units_and_ids(text_a, a)
        units_b, word_ids_b = self._units_and_ids(text_b, b)
        pairs = align_from_embeddings(
            a.embeddings, b.embeddings, word_ids_a, word_ids_b,
            method=self.method, mode=self.mode, k=self.k, max_count=self.max_count,
        )
        return pairs, units_a, units_b

    def align_pairs(self, pairs, show_progress=False, progress_callback=None,
                    with_units=False):
        """Align many text pairs.

        ``pairs``: iterable of ``(text_a, text_b)`` tuples.
        Returns a list of per-pair alignments, in input order. Empty/blank texts
        yield an empty alignment so positional correspondence is preserved.

        ``with_units=True`` returns ``(pairs, units_a, units_b)`` triples instead
        of bare pair lists, without a second encoding pass.

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
                results.append(([], [], []) if with_units else [])
            elif with_units:
                results.append(self.align_with_units(text_a, text_b))
            else:
                results.append(self.align(text_a, text_b))
            if progress_callback:
                progress_callback(idx, total)
        return results
