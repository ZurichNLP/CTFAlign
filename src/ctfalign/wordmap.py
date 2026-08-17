"""Token-to-word mapping and projection of token alignments to word alignments.

A ``word_ids`` list assigns each *non-special* token an index into the
whitespace-split words of its text (a "b2w map"). The offset-based strategy here
is the HuggingFace path (uses character offsets); a custom encoder can build
``word_ids`` differently (e.g. from SentencePiece pieces) and feed the same projection.
"""


def get_word_boundaries(text: str):
    """Return (start, end) character offsets for each space-separated word."""
    boundaries = []
    pos = 0
    for w in text.split():
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text):
            start = pos
            end = pos + len(w)
            boundaries.append((start, end))
            pos = end
    return boundaries


def offset_mapping_to_word_ids(offset_mapping, word_boundaries):
    """Map each token to a word index using character offset overlap.

    A token at ``[tok_start, tok_end)`` belongs to word ``i`` if it overlaps
    ``word_boundaries[i]``. Robust to tokenisers whose offsets don't sit exactly
    on word boundaries. Special / padding tokens (offset ``(0, 0)``) map to None.
    """
    result = []
    for tok_start, tok_end in offset_mapping:
        if tok_start == 0 and tok_end == 0:          # special / padding token
            result.append(None)
            continue
        assigned = None
        for word_idx, (w_start, w_end) in enumerate(word_boundaries):
            if tok_start < w_end and tok_end > w_start:   # overlap test
                assigned = word_idx
                break
        result.append(assigned)
    return result


def build_b2w_map(tokenizer, sentence: str) -> list:
    """Build a token-to-word map via full-sentence tokenisation + offset mapping.

    Consistent with how embeddings are computed (full-sentence tokenisation,
    special tokens removed). Requires a HuggingFace fast tokenizer that returns
    ``offset_mapping``.
    """
    word_boundaries = get_word_boundaries(sentence)

    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        add_special_tokens=True,
        truncation=False,
        verbose=False,   # long docs are chunked downstream; see encoders/huggingface.py
    )

    offset_mapping = encoding["offset_mapping"]
    special_tokens_mask = encoding["special_tokens_mask"]

    word_ids = offset_mapping_to_word_ids(offset_mapping, word_boundaries)

    # Filter out special tokens (same tokens excluded when building similarity matrix)
    b2w_map = [
        wid for wid, is_special in zip(word_ids, special_tokens_mask)
        if not is_special
    ]

    # Replace None (unmapped tokens) with nearest valid word index
    for i, wid in enumerate(b2w_map):
        if wid is None:
            for delta in range(1, len(b2w_map)):
                if i - delta >= 0 and b2w_map[i - delta] is not None:
                    b2w_map[i] = b2w_map[i - delta]
                    break
                if i + delta < len(b2w_map) and b2w_map[i + delta] is not None:
                    b2w_map[i] = b2w_map[i + delta]
                    break

    return b2w_map


def project_token_alignment(token_alignment, word_ids_a, word_ids_b):
    """Project a token-level alignment to word-level using two b2w maps.

    ``token_alignment``: list of ``(tok_idx_src, tok_idx_tgt)`` over the
    non-special token sequence. ``word_ids_a/b``: per-token word indices.
    Returns a sorted, deduplicated list of ``(word_idx_src, word_idx_tgt)``.
    """
    word_alignment = set()
    for tok_i, tok_j in token_alignment:
        if tok_i < len(word_ids_a) and tok_j < len(word_ids_b):
            word_alignment.add((word_ids_a[tok_i], word_ids_b[tok_j]))
    return sorted(word_alignment)
