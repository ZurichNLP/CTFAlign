"""Command-line interface: ``ctfalign align SOURCE TARGET``.

Aligns two parallel text files (one segment per line) and writes JSONL: one
object per line with the original texts and the alignment labels.
"""
import argparse
import json
import sys

from .aligner import METHODS, UNITS, WordAligner


def to_word_alignment_label_notation(pairs):
    """Format word-index pairs as a word-alignment-label-notation line: ``0-0 1-1 2-2``."""
    return " ".join(f"{i}-{j}" for i, j in pairs)


def alignment_record(text_a, text_b, pairs, fmt="string",
                     units_a=None, units_b=None):
    """Build one JSONL record: original texts plus the alignment labels.

    ``fmt="string"`` encodes the alignment as a word-alignment-label-notation
    string (``"0-0 1-1"``); ``fmt="pairs"`` as a list of ``[i, j]`` pairs.

    ``units_a`` / ``units_b``, when given, add the strings the label indices
    point at. Whitespace words are recoverable from ``text_a``/``text_b``, so
    these are only emitted for token-level alignment, where the indices refer to
    the encoder's subword tokens and are otherwise uninterpretable.
    """
    labels = ([list(p) for p in pairs] if fmt == "pairs"
              else to_word_alignment_label_notation(pairs))
    record = {"text_a": text_a, "text_b": text_b, "labels": labels}
    if units_a is not None:
        record["units_a"] = list(units_a)
    if units_b is not None:
        record["units_b"] = list(units_b)
    return record


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def cmd_align(args):
    aligner = WordAligner.from_huggingface(
        args.model,
        layer=args.layer,
        lang_pair=args.lang_pair,
        granularity=args.granularity,
        method=args.method,
        k=args.k,
        units=args.units,
        device=args.device,
        **({"mode": args.mode} if args.mode else {}),
    )

    src = _read_lines(args.source)
    tgt = _read_lines(args.target)
    if len(src) != len(tgt):
        sys.exit(
            f"line count mismatch: {args.source} has {len(src)} lines, "
            f"{args.target} has {len(tgt)}"
        )

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        token_level = args.units == "tokens"
        results = aligner.align_pairs(zip(src, tgt), show_progress=not args.quiet,
                                      with_units=token_level)
        for text_a, text_b, result in zip(src, tgt, results):
            pairs, units_a, units_b = result if token_level else (result, None, None)
            record = alignment_record(text_a, text_b, pairs, fmt=args.label_format,
                                      units_a=units_a, units_b=units_b)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out.close()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ctfalign",
        description="Document-level word alignment (CTFAlign, MDPAlign, SimAlign).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("align", help="align two parallel text files")
    a.add_argument("source", help="source file, one segment per line")
    a.add_argument("target", help="target file, one segment per line (same line count)")
    a.add_argument("-m", "--model", required=True,
                   help="HuggingFace model id, e.g. Qwen/Qwen3-Embedding-4B")
    a.add_argument("--method", default="ctfalign", choices=list(METHODS),
                   help="alignment algorithm (default: ctfalign)")
    a.add_argument("--mode", choices=["argmax", "itermax"],
                   help="base SimAlign variant (default: argmax)")
    a.add_argument("--layer", type=int,
                   help="encoder layer (default: the tuned best layer for the model/lang-pair)")
    a.add_argument("--granularity", choices=["documents", "sentences"], default="documents",
                   help="which tuned-layer table to use when --layer is not given "
                        "(default: documents)")
    a.add_argument("--k", type=float,
                   help="constraint hyperparameter: w blocks (CTFAlign) or k tokens (MDPAlign)")
    a.add_argument("--lang-pair", dest="lang_pair",
                   help="e.g. en-fr, to pick a language-pair-specific tuned layer")
    a.add_argument("--units", choices=list(UNITS), default="words",
                   help="what the label indices refer to: 'words' = whitespace-split "
                        "words (default; pre-segment zh/ja input), or 'tokens' = the "
                        "encoder's subword tokens, with no projection to words. "
                        "Token mode also writes 'units_a'/'units_b' to each record")
    a.add_argument("--device", help="torch device, e.g. cuda:0 or cpu")
    a.add_argument("-o", "--output", help="output JSONL file (default: stdout)")
    a.add_argument("--label-format", dest="label_format",
                   choices=["string", "pairs"], default="string",
                   help="encoding of the 'labels' field in each JSONL record: "
                        "'string' = word-alignment-label-notation, e.g. \"0-0 1-1\" "
                        "(default), or 'pairs' = list of [i, j]")
    a.add_argument("-q", "--quiet", action="store_true", help="suppress progress")
    a.set_defaults(func=cmd_align)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
