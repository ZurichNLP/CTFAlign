"""Empirically-tuned defaults from the paper experiments.

Layer tables are keyed by ``(lang_pair, short_model_name)`` where the short name
is the part of the HuggingFace id after the final ``/``. These were measured on
HuggingFace models with HuggingFace layer indexing (index 0 = embeddings,
1..N = transformer blocks) and do NOT transfer to other backends.
"""
import warnings

# Best layer per (lang_pair, model) at sentence granularity.
BEST_LAYERS_SENTS = {
    "en-fr": {"bert-base-multilingual-cased": 9,  "mmBERT-base": 17, "xlm-roberta-large": 9,  "EuroBERT-610m": 6,  "LaBSE": 5, "Qwen3-Embedding-4B": 7},
    "en-ro": {"bert-base-multilingual-cased": 9,  "mmBERT-base": 14, "xlm-roberta-large": 10, "EuroBERT-610m": 5,  "LaBSE": 5, "Qwen3-Embedding-4B": 8},
    "en-ja": {"bert-base-multilingual-cased": 8,  "mmBERT-base": 14, "xlm-roberta-large": 19, "EuroBERT-610m": 9,  "LaBSE": 8, "Qwen3-Embedding-4B": 23},
    "en-zh": {"bert-base-multilingual-cased": 8,  "mmBERT-base": 14, "xlm-roberta-large": 14, "EuroBERT-610m": 9,  "LaBSE": 5, "Qwen3-Embedding-4B": 8},
    "la-gr": {"bert-base-multilingual-cased": 8,  "mmBERT-base": 15, "xlm-roberta-large": 16, "EuroBERT-610m": 10, "LaBSE": 8, "Qwen3-Embedding-4B": 15},
}

# Best layer per (lang_pair, model) at document granularity (itermax=2).
BEST_LAYERS_DOCS = {
    "en-fr": {"mmBERT-base": 15, "EuroBERT-610m": 9,  "LaBSE": 11, "LaBSE-chunked": 9,  "Qwen3-Embedding-4B": 19, "Qwen3-Embedding-0.6B": 16},
    "en-ro": {"mmBERT-base": 17, "EuroBERT-610m": 9,  "LaBSE": 8,  "LaBSE-chunked": 5,  "Qwen3-Embedding-4B": 20, "Qwen3-Embedding-0.6B": 18},
    "en-ja": {"mmBERT-base": 15, "EuroBERT-610m": 11, "LaBSE": 8,  "LaBSE-chunked": 12, "Qwen3-Embedding-4B": 20, "Qwen3-Embedding-0.6B": 16},
    "en-zh": {"mmBERT-base": 15, "EuroBERT-610m": 11, "LaBSE": 8,  "LaBSE-chunked": 9,  "Qwen3-Embedding-4B": 20, "Qwen3-Embedding-0.6B": 16},
    "la-gr": {"mmBERT-base": 14, "EuroBERT-610m": 9,  "LaBSE": 8,  "LaBSE-chunked": 11, "Qwen3-Embedding-4B": 21, "Qwen3-Embedding-0.6B": 18},
}

# Default mask hyperparameter (k tokens for MDPAlign, half-width w in coarse
# blocks for CTFAlign), used when the user does not pass one.
DEFAULT_K = {
    "simalign":        None,
    "mdpalign-strict": 25,    # modal-best on dev; not reported on test (weakest variant)
    "mdpalign-fuzzy":  150,
    "ctfalign":        8,
}


def _modal_best_layer(table, short_model):
    """Most frequent best layer for a model across all known language pairs."""
    counts = {}
    for lp_dict in table.values():
        if short_model in lp_dict:
            counts[lp_dict[short_model]] = counts.get(lp_dict[short_model], 0) + 1
    return max(counts, key=counts.get) if counts else None


def resolve_layer(short_model, lang_pair=None, granularity="sentences", fallback_fraction=2 / 3):
    """Tiered layer default.

    1. ``(lang_pair, model)`` lookup if both are known.
    2. otherwise the modal-best layer for that model across language pairs.
    3. otherwise ``None`` -- caller should apply a depth-relative heuristic
       (e.g. ``round(fallback_fraction * n_layers)``) and warn.
    """
    table = BEST_LAYERS_SENTS if granularity == "sentences" else BEST_LAYERS_DOCS
    if lang_pair and lang_pair in table and short_model in table[lang_pair]:
        return table[lang_pair][short_model]
    modal = _modal_best_layer(table, short_model)
    if modal is not None:
        if lang_pair:
            warnings.warn(
                f"No tuned layer for ({lang_pair}, {short_model}); using modal-best "
                f"layer {modal} for {short_model}."
            )
        return modal
    return None


def resolve_k(method, k=None):
    """Default mask hyperparameter for a method unless the user provides one."""
    if k is not None:
        return k
    return DEFAULT_K.get(method)
