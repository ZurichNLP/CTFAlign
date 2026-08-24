# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-24

First release. Implements CTFAlign and MDPAlign from
[*Scaling Unsupervised Word Alignment to Documents via Structural Constraints*](https://arxiv.org/abs/2608.21023).

### Added

- `WordAligner` with a HuggingFace backend, plus `align_from_embeddings` and
  `align_from_similarity` for bring-your-own token embeddings.
- Alignment methods `ctfalign`, `mdpalign-strict`, `mdpalign-fuzzy` and
  `simalign`, each in `argmax` and `itermax` mode.
- Tuned per-model and per-language-pair encoder layer defaults, at both document
  and sentence granularity.
- `units="tokens"`, aligning at the encoder's subword tokens with no projection
  to whitespace words, for input that cannot be pre-segmented. `EncodedText`
  carries the token surface strings and `align_with_units` returns them
  alongside the pairs. Defaults to `units="words"`, as used for the paper.
- `ctfalign` command-line interface, writing JSONL.
- Chunked encoding for documents longer than the model's context window.

[Unreleased]: https://github.com/ZurichNLP/CTFAlign/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZurichNLP/CTFAlign/releases/tag/v0.1.0
