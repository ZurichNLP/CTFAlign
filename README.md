# CTFAlign: Document-level Word Alignment

[![Paper](https://img.shields.io/badge/Paper-B31B1B.svg)](TODO)
[![Experimental Code](https://img.shields.io/badge/Experimental%20Code-6A5ACD.svg)](https://github.com/ZurichNLP/document-level-word-alignment)
[![Demo](https://img.shields.io/badge/Demo-4C8BF5.svg)](https://huggingface.co/spaces/miwytt/ctfalign)

CTFAlign provides training-free word alignment for parallel documents, without requiring sentence-level segmentation or alignment. It extends embedding-based word alignment to long documents by constraining the alignment search space.

The package implements two methods from *Scaling Unsupervised Word Alignment to Documents with Structural Constraints* (TODO: link): **CTFAlign**, a coarse-to-fine approach designed to accommodate structural differences between documents, and **MDPAlign**, a lightweight diagonal prior for documents with approximately parallel structure. Both methods are encoder-agnostic and can be used with built-in Hugging Face models or your own token embeddings.

The experimental code from the paper can be found here: https://github.com/ZurichNLP/document-level-word-alignment

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Aligning parallel files](#aligning-parallel-files)
- [Bring your own embeddings](#bring-your-own-embeddings)
- [Methods and performance](#methods-and-performance)
- [Visualizing alignments](#visualizing-alignments)
- [Citation](#citation)

## Install

Python 3.11 recommended.

```bash
pip install ctfalign # not yet published to PyPI; install from source with pip install .
```

The built-in encoder is HuggingFace `transformers`; you can also bring your own embeddings from any framework (see [Bring your own embeddings](#bring-your-own-embeddings)).

## Quick start

```python
from ctfalign import WordAligner

aligner = WordAligner.from_huggingface("Qwen/Qwen3-Embedding-4B", method="ctfalign")
aligner.align("each of them is very complex , but the link between the two is even more complex which makes the whole situation for most people understandably confusing . the commissionners went on : they are constrained by limits which are imposed in order to ensure that the freedom of one person does not violate that of another .", "chacun en lui - même est très complexe et le lien entre les deux le est encore davantage de sorte que pour beaucoup la situation présente est confuse . je poursuis la lecture de les recommandations de les commissaires : ils sont restreints par certaines limites qui ont été fixées pour garantir que la liberté de une personne ne empiète pas sur celle de une autre .")   # -> [(0, 0), (2, 0), (3, 5), (4, 6), (5, 7), (8, 9), (9, 10), (10, 11), (12, 13), (14, 16), (15, 17), (17, 20), (19, 17), (21, 24), (22, 21), (23, 22), (26, 27), (27, 28), (29, 38), (31, 31), (32, 39), (33, 40), (34, 41), (35, 42), (36, 43), (37, 45), (38, 46), (39, 45), (40, 49), (44, 51), (45, 51), (46, 53), (47, 54), (48, 55), (50, 57), (52, 60), (53, 61), (55, 63), (56, 65), (57, 66)]
```

`method` controls the structural constraint:
* `"ctfalign"` applies the coarse-to-fine constraint proposed in our paper.
* `"mdpalign-fuzzy"` applies the MDPAlign diagonal prior.
* `"mdpalign-strict"` applies a hard diagonal band instead of the fuzzy prior: similarities further than `k` tokens from the diagonal are zeroed rather than downweighted.
* `"simalign"` applies no structural constraint and provides the unconstrained SimAlign baseline ([Jalili Sabet et al., 2020](https://aclanthology.org/2020.findings-emnlp.147/)).

`mode` controls how alignments are extracted: `argmax` (default) or `itermax` ([Jalili Sabet et al., 2020](https://aclanthology.org/2020.findings-emnlp.147/)). With `itermax`, `max_count` sets the number of refinement iterations (default 2; higher means more recall).

`layer` and `k` default to the values found best in our experiments; both are overridable. `k` is the one constraint hyperparameter for every method: the buffer in coarse blocks for CTFAlign, and a tolerance in tokens for MDPAlign. The layer is set to the optimal for document-level by default (`granularity="documents"`); pass `lang_pair="en-fr"` to use a language-pair-optimal layer.

The built-in optimal layers cover the models and language pairs from the accompanying paper. At document granularity: `sentence-transformers/LaBSE`, `jhu-clsp/mmBERT-base`, `EuroBERT/EuroBERT-610m`, `Qwen/Qwen3-Embedding-0.6B/4B`; EN-FR/RO/JA/ZH, LA-GR. `granularity="sentences"` additionally covers `bert-base-multilingual-cased` and `xlm-roberta-large`. Note that EN-CZ, which appears in the results table below, has no tuned layer in either table: passing `lang_pair="en-cz"` falls back to the model's modal-best layer across the other pairs.

For any other model, we recommend finding the optimal the layer on a development set. Find the development sets for the language pairs covered in our paper here: https://huggingface.co/datasets/ZurichNLP/document-level-word-alignment

## Aligning parallel files

Two parallel files, one document per line, with matching line counts.

From the command line — writes JSONL, one object per line with the original texts and the alignment labels:

```bash
ctfalign align examples/documents.en examples/documents.fr -m Qwen/Qwen3-Embedding-4B --method ctfalign --lang-pair en-fr -o examples/align.jsonl
```

Each output line looks like:

```json
{"text_a": "the cat sat", "text_b": "le chat assis", "labels": "0-0 1-1 2-2"}
```

`labels` is word-alignment-label-notation (`i-j` pairs) by default; pass `--label-format pairs` to get a list of `[i, j]` pairs instead.

Or in Python:

```python
from ctfalign import WordAligner

aligner = WordAligner.from_huggingface("Qwen/Qwen3-Embedding-4B", method="ctfalign")
src = open("source.en").read().splitlines()
tgt = open("target.fr").read().splitlines()
for pairs in aligner.align_pairs(zip(src, tgt)):
    print(" ".join(f"{i}-{j}" for i, j in pairs))
```

Alignment indices are over whitespace-split words; for languages without whitespace (e.g. zh, ja) pre-segment the text with spaces.

### Documents longer than the model's context window

A document that does not fit the encoder's context window is split into consecutive, non-overlapping chunks that are encoded independently, and the resulting token embeddings are concatenated. Alignments are still computed over the whole document, but because each chunk is encoded without the others as context. The aligner warns once when this happens.

## Bring your own embeddings

Already have token embeddings? Skip the encoder entirely and pass four arrays per text pair:


| Argument     | Shape / type                 | Meaning                                                        |
| ------------ | ---------------------------- | -------------------------------------------------------------- |
| `emb_a`      | `(n_tokens_a, hidden)` float | source token embeddings, one row per subword token             |
| `emb_b`      | `(n_tokens_b, hidden)` float | target token embeddings                                        |
| `word_ids_a` | length-`n_tokens_a` int list | for each token, the index of the whitespace word it belongs to |
| `word_ids_b` | length-`n_tokens_b` int list | same, for the target                                           |


`word_ids` maps subword tokens back to words: `[0, 0, 1, 2, 2]` means tokens 0–1 form word 0, token 2 is word 1, tokens 3–4 are word 2. The returned `(i, j)` pairs are word indices.

```python
import numpy as np
from ctfalign import align_from_embeddings

emb_a = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])   # 3 tokens, hidden=2
emb_b = np.array([[1.0, 0.0], [0.0, 1.0]])               # 2 tokens
word_ids_a = [0, 0, 1]    # tokens 0,1 -> word 0 ; token 2 -> word 1
word_ids_b = [0, 1]       # one token per word

align_from_embeddings(emb_a, emb_b, word_ids_a, word_ids_b, method="ctfalign")
# -> [(0, 0), (1, 1)]
```

If you already have the token-level similarity matrix rather than the embeddings, `align_from_similarity(sim, word_ids_a, word_ids_b, method=...)` takes the same arguments with `sim` of shape `(n_tokens_a, n_tokens_b)` in place of `emb_a`/`emb_b`. Negative entries are clamped to zero before the constraint is applied.

Or implement the `Encoder` protocol (`encode(text) -> EncodedText`) for your framework and hand it to `WordAligner(my_encoder, method=...)`.

## Methods and performance


| Method           | Constraint                      | Default hyperparameter |
| ---------------- | ------------------------------- | ---------------------- |
| `ctfalign`       | coarse-to-fine pruning          | `k=8` blocks           |
| `mdpalign-fuzzy` | fuzzy (Gaussian) diagonal prior | `k=150` tokens         |
| `mdpalign-strict`| hard diagonal band              | `k=25` tokens          |
| `simalign`       | none                            | —                      |


`mode` selects the base SimAlign ([Jalili Sabet et al. 2020](https://github.com/cisnlp/simalign)) variant: `"argmax"` (default) or `"itermax"`.

The following table shows the document-level word-alignment AER (alignment error rate; lower is better) on the tested language pairs in our experiments:


| Model              | Method   | Mode    | en-fr     | en-ro     | en-ja     | en-zh     | la-gr     | en-cz     | Avg       |
| ------------------ | -------- | ------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Qwen3-Embedding-4B | CTFAlign | argmax  | **0.151** | 0.339     | 0.535     | 0.255     | **0.293** | **0.191** | **0.294** |
| Qwen3-Embedding-4B | CTFAlign | itermax | 0.181     | 0.332     | **0.518** | 0.276     | 0.371     | 0.233     | 0.319     |
| Qwen3-Embedding-4B | MDPAlign | argmax  | 0.232     | 0.342     | 0.752     | **0.254** | 0.397     | 0.221     | 0.366     |
| Qwen3-Embedding-4B | MDPAlign | itermax | 0.239     | **0.330** | 0.728     | 0.271     | 0.455     | 0.257     | 0.380     |

For full experiment code and test data see (TODO: Add link to experiment repo).

## Visualizing alignments

`viewer.html` is a viewer for the JSONL output:

1. Open `viewer.html`.
2. Choose or drag-and-drop a JSONL file produced by `ctfalign align`.
3. The two texts appear side by side. Hover a word to highlight its aligned
  words in the other text; click to pin the highlight; use the ← / →
   keys or the Prev/Next buttons to flip through samples.

It reads both `--label-format` encodings (`string` and `pairs`).

Find a more interactive demo here: https://huggingface.co/spaces/miwytt/ctfalign

## Citation

```
@TODO{authorlastname2026,
  author       = {Last, First},
  title        = {Title},
  year         = {2026},
}
```

