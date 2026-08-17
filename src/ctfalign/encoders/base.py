"""Encoder interface: the single framework-dependent step.

An encoder turns text into per-token contextual hidden states at a chosen layer
plus a token-to-word map. Everything downstream (similarity, masking, alignment)
is framework-independent.

Eligibility rule: an encoder must yield per-token hidden states at a selectable
layer. Pooled-sentence-vector outputs (most embedding APIs) cannot be aligned
token-to-token and are out of scope.
"""
import warnings
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import torch


@dataclass
class EncodedText:
    """Output of an encoder for one text.

    embeddings: tensor of shape ``(n_tokens, hidden)`` over non-special tokens.
    word_ids:   length-``n_tokens`` list mapping each token to a whitespace word
                index (a "b2w map"). ``len(word_ids)`` must equal
                ``embeddings.shape[0]``.
    """
    embeddings: torch.Tensor
    word_ids: List[int]

    def __post_init__(self):
        # Lengths match for normal sequences; long-document chunking can shift
        # token counts slightly (re-tokenisation), which projection tolerates by
        # bounds-guarding. Warn rather than raise so that path keeps working.
        if len(self.word_ids) != self.embeddings.shape[0]:
            warnings.warn(
                f"word_ids length ({len(self.word_ids)}) != n_tokens "
                f"({self.embeddings.shape[0]}); extra tokens will be dropped "
                f"during word projection."
            )


@runtime_checkable
class Encoder(Protocol):
    """Minimal contract for a token-level encoder."""

    def encode(self, text: str) -> EncodedText:
        ...
