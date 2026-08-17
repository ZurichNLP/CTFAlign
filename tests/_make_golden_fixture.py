"""Regenerate the golden fixture from the research repo's scripts/get_predictions.py.

The research repo is https://github.com/ZurichNLP/document-level-word-alignment.

Run with the project env active, pointing at a checkout of that repo which has
its ``similarity_matrices/`` data present::

    python tests/_make_golden_fixture.py
    python tests/_make_golden_fixture.py --research-repo /path/to/doc-level-word-alignment

Captures the expected word alignments for every method/mode/k case over each
similarity-matrix file in ``SIM_FILES``, so the package can be tested against a
frozen reference even after get_predictions.py changes (a live-import comparison
would be circular once that script imports the package).

The reference is computed through the research functions in the exact order
get_predictions.py's main() applies them: clamp(min=0) -> belt mask -> extractor
(CTFAlign bypasses the belt) -> word projection.
"""
import argparse
import importlib.util
import json
import os

import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESEARCH_REPO = os.path.abspath(
    os.path.join(HERE, "..", "..", "document-level-word-alignment")
)
OUT = os.path.join(HERE, "fixtures", "golden_qwen19_docs.json")
HF_MODEL = "Qwen/Qwen3-Embedding-4B"

# Similarity-matrix files to capture, relative to <research repo>/similarity_matrices.
# All are Qwen3-Embedding-4B, so one tokenizer covers them.
SIM_FILES = [
    "Qwen3-Embedding-4B/19/documents/en-fr/dev.json",
    "Qwen3-Embedding-4B/19/documents/en-fr/test.json",
    "Qwen3-Embedding-4B/en-ja-test.json",
]

# case_key -> (research method, research mask, k)
CASES = [
    ("simalign/argmax",         "simalign-argmax",  "none",            None),
    ("mdpalign-strict/argmax",  "simalign-argmax",  "mdpalign_strict", 25.0),
    ("mdpalign-fuzzy/argmax",   "simalign-argmax",  "mdpalign_fuzzy",  150.0),
    ("ctfalign/argmax/8",       "simalign-argmax",  "ctfalign",        8.0),
    ("ctfalign/argmax/4",       "simalign-argmax",  "ctfalign",        4.0),
    ("mdpalign-strict/itermax", "simalign-itermax", "mdpalign_strict", 25.0),
    ("mdpalign-fuzzy/itermax",  "simalign-itermax", "mdpalign_fuzzy",  150.0),
    ("ctfalign/itermax/8",      "simalign-itermax", "ctfalign",        8.0),
    ("simalign/itermax",        "simalign-itermax", "none",            None),
]


def load_source(research_repo):
    path = os.path.join(research_repo, "scripts", "get_predictions.py")
    if not os.path.exists(path):
        raise SystemExit(f"get_predictions.py not found at {path}")
    spec = importlib.util.spec_from_file_location("gp_source", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(gp, tok, sim, b2w_a, b2w_b, method, mask, k):
    """One case for one row, via the research functions in main()'s order."""
    sm = torch.tensor(sim).clamp(min=0)
    if mask == "mdpalign_strict":
        sm = gp.apply_mdpalign_strict(sm, k)
    elif mask == "mdpalign_fuzzy":
        sm = gp.apply_mdpalign_fuzzy(sm, k)

    if mask == "ctfalign":
        ta = gp.ctfalign(sm, method, width=int(k))
    elif method == "simalign-argmax":
        ta = gp.argmax_align(sm)
    else:
        ta = gp.iter_max(sm.numpy(), max_count=2)

    return gp.get_word_align_from_tok_align(tok, ta, b2w_a, b2w_b, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-repo", default=os.environ.get(
        "CTFALIGN_RESEARCH_REPO", DEFAULT_RESEARCH_REPO))
    args = ap.parse_args()

    gp = load_source(args.research_repo)
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    fixture = {
        "_source": "scripts/get_predictions.py",
        "_model": HF_MODEL,
        "datasets": {},
    }

    for rel in SIM_FILES:
        path = os.path.join(args.research_repo, "similarity_matrices", rel)
        if not os.path.exists(path):
            print(f"skip (missing): {rel}")
            continue
        cases = {key: [] for key, _, _, _ in CASES}
        # Streamed: these files reach ~1GB, so never materialise them at once.
        with open(path, encoding="utf-8") as f:
            for row_i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sim = record.pop("similarity_matrices")
                b2w_a = gp.build_b2w_map(tok, record["text_a"])
                b2w_b = gp.build_b2w_map(tok, record["text_b"])
                for key, method, mask, k in CASES:
                    ref = reference(gp, tok, sim, b2w_a, b2w_b, method, mask, k)
                    cases[key].append([list(p) for p in ref])
                print(f"  {rel}:{row_i} ({len(b2w_a)}x{len(b2w_b)} tokens)", flush=True)
                del sim, record
        fixture["datasets"][rel] = cases
        n_rows = len(cases[CASES[0][0]])
        print(f"{rel}: {n_rows} rows x {len(CASES)} cases")

    if not fixture["datasets"]:
        raise SystemExit("no similarity-matrix files found; nothing written")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(fixture, f)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
