"""ctfalign: document-level word alignment from multilingual token embeddings.

Quick start (HuggingFace backend, included in ``pip install ctfalign``)::

    from ctfalign import WordAligner
    aligner = WordAligner.from_huggingface("Qwen/Qwen3-Embedding-4B", method="ctfalign")
    aligner.align("the cat sat", "le chat assis")   # -> [(0, 0), (1, 1), (2, 2)]

Bring your own embeddings (no framework dependency)::

    from ctfalign import align_from_embeddings
    align_from_embeddings(emb_a, emb_b, word_ids_a, word_ids_b, method="mdpalign-fuzzy")
"""
from .aligner import (
    METHODS,
    UNITS,
    WordAligner,
    align_from_embeddings,
    align_from_similarity,
)
from .cli import alignment_record, to_word_alignment_label_notation
from .encoders import EncodedText, Encoder

__version__ = "0.1.0"

__all__ = [
    "WordAligner",
    "align_from_embeddings",
    "align_from_similarity",
    "alignment_record",
    "to_word_alignment_label_notation",
    "METHODS",
    "UNITS",
    "EncodedText",
    "Encoder",
]
