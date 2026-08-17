"""HuggingFace token-level encoder (the batteries-included default).

Extracts per-token hidden states at a chosen layer, with the same long-sequence
chunking strategy as the research pipeline, and builds the token-to-word map via
character offsets. Uses ``transformers`` (a core dependency of ctfalign).

Long-sequence warnings: ``transformers`` emits its own "Token indices sequence
length is longer ..." message, which claims running the sequence "will result in
indexing errors" and is keyed off ``tokenizer.model_max_length``. That is
misleading here -- we deliberately over-tokenise and then chunk, and
``model_max_length`` is unset (a ~1e19 sentinel) on some models, e.g. EuroBERT,
so the message never appears for them however long the document is. We pass
``verbose=False`` to silence it and warn ourselves off the model's real context
window (``config.max_position_embeddings``).
"""
import warnings

import torch

from ..wordmap import build_b2w_map
from .base import EncodedText


class HFEncoder:
    """Per-token encoder backed by ``transformers.AutoModel``.

    layer: index into ``output_hidden_states`` (0 = embeddings, 1..N = blocks).
           Supports negative indexing (-1 = top layer).
    """

    def __init__(self, model_name, layer, device=None, token=None,
                 trust_remote_code=True, tokenizer_name=None):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name, token=token
        )
        self.model = AutoModel.from_pretrained(
            model_name, token=token, trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()
        self.max_length = self.model.config.max_position_embeddings
        self._warned_long = False

    # -- embedding -----------------------------------------------------------

    def _warn_long_sequence(self, length, chunk_size):
        """Warn once per encoder that a text exceeds the model's context window.

        Reported off ``config.max_position_embeddings`` (e.g. 512 for LaBSE, 8192
        for mmBERT-base and EuroBERT-610m, 32768 / 40960 for Qwen3-Embedding-0.6B
        / -4B) rather than ``tokenizer.model_max_length``, which is unset on some
        models. Warns once per encoder instance so that aligning a corpus of long
        documents does not emit one message per document.
        """
        if self._warned_long:
            return
        self._warned_long = True
        n_chunks = -(-length // chunk_size)
        warnings.warn(
            f"Token sequence length ({length}) exceeds the maximum sequence length "
            f"for {self.model_name} ({self.max_length}); running it through the model "
            f"in one pass may result in indexing errors. Splitting it into {n_chunks} "
            f"chunks of at most {chunk_size} tokens, which are encoded independently: "
            f"no alignment can cross a chunk boundary. Further long sequences will not "
            f"be reported."
        )

    def _embed(self, text):
        """Return non-special token embeddings ``(n_tokens, hidden)`` for text."""
        chunk_size = self.max_length - 2
        length = len(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))
        if length > chunk_size:
            self._warn_long_sequence(length, chunk_size)
            return self._embed_chunked(text)

        enc = self.tokenizer(
            [text], padding=False, truncation=False, return_tensors="pt",
            return_special_tokens_mask=True, verbose=False,
        ).to(self.device)
        enc = dict(enc)
        special = enc.pop("special_tokens_mask").bool()
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
            if out.get("hidden_states") is None:
                raise ValueError(
                    "Model did not return hidden states; it may be a "
                    "sentence-transformer wrapper exposing only pooled vectors."
                )
            emb = out["hidden_states"][self.layer][0]
            keep = enc["attention_mask"].bool()[0] & ~special[0]
            return emb[keep]

    def _embed_chunked(self, text):
        """Embed a long text in non-overlapping context-window-sized chunks."""
        chunk_size = self.max_length - 2
        full_ids = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        chunk_embeddings = []
        for i in range(0, len(full_ids), chunk_size):
            chunk_ids = full_ids[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            enc = self.tokenizer(
                chunk_text, return_tensors="pt", return_special_tokens_mask=True,
                truncation=True, max_length=self.max_length, add_special_tokens=True,
            ).to(self.device)
            enc = dict(enc)
            special = enc.pop("special_tokens_mask").bool()
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
                emb = out["hidden_states"][self.layer][0]
                keep = enc["attention_mask"].bool()[0] & ~special[0]
                chunk_embeddings.append(emb[keep])
        return torch.cat(chunk_embeddings, dim=0)

    # -- public API ----------------------------------------------------------

    def encode(self, text: str) -> EncodedText:
        embeddings = self._embed(text)
        word_ids = build_b2w_map(self.tokenizer, text)
        return EncodedText(embeddings=embeddings.cpu(), word_ids=word_ids)
