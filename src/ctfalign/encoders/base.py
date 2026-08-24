"""Encoder interface: the single framework-dependent step.

An encoder turns text into per-token contextual hidden states at a chosen layer,
a token-to-word map, and (optionally) the token surface strings. Everything
downstream (similarity, masking, alignment) is framework-independent.

Eligibility rule: an encoder must yield per-token hidden states at a selectable
layer. Pooled-sentence-vector outputs (most embedding APIs) cannot be aligned
token-to-token and are out of scope.
"""
import warnings
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

import torch


@dataclass
class EncodedText:
    """Output of an encoder for one text.

    embeddings: tensor of shape ``(n_tokens, hidden)`` over non-special tokens.
    word_ids:   length-``n_tokens`` list mapping each token to a whitespace word
                index (a "b2w map"). ``len(word_ids)`` must equal
                ``embeddings.shape[0]``.
    tokens:     optional length-``n_tokens`` list of token surface strings. Only
                needed for token-level alignment (``units="tokens"``), where the
                returned indices point into this list rather than into
                whitespace words; ``None`` if the encoder does not provide it.
    """
    embeddings: torch.Tensor
    word_ids: List[int]
    tokens: Optional[List[str]] = None

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
        if self.tokens is not None and len(self.tokens) != self.embeddings.shape[0]:
            raise ValueError(
                f"tokens length ({len(self.tokens)}) != n_tokens "
                f"({self.embeddings.shape[0]}); token-level alignment indexes "
                f"tokens directly, so a mismatch would mislabel the output."
            )


@runtime_checkable
class Encoder(Protocol):
    """Minimal contract for a token-level encoder."""

    def encode(self, text: str) -> EncodedText:
        ...
