"""Golden tests: package output == frozen reference from the research pipeline.

The reference in ``fixtures/golden_qwen19_docs.json`` was captured by running the
research repo's ``scripts/get_predictions.py`` functions directly (see
``_make_golden_fixture.py``). Comparing against a frozen file keeps this a real
regression guard even once that script imports this package -- a live comparison
would be circular.

Needs the research repo's ``similarity_matrices/`` data plus a (cached) Qwen
tokenizer, so it is marked ``slow`` and skips automatically when either is
absent. Point it at a checkout of
https://github.com/ZurichNLP/document-level-word-alignment with::

    CTFALIGN_RESEARCH_REPO=/path/to/document-level-word-alignment \
        pytest tests/test_golden.py

Without that variable it looks for the repo as a sibling of this one. Run a
single dataset with e.g. ``-k en-ja``.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "golden_qwen19_docs.json")
RESEARCH_REPO = os.environ.get(
    "CTFALIGN_RESEARCH_REPO",
    os.path.abspath(os.path.join(HERE, "..", "..", "doc-level-word-alignment")),
)
HF_MODEL = "Qwen/Qwen3-Embedding-4B"

# case_key -> (method, mode, k); must stay in sync with _make_golden_fixture.CASES.
CASES = {
    "simalign/argmax":         ("simalign", "argmax", None),
    "mdpalign-strict/argmax":  ("mdpalign-strict", "argmax", 25.0),
    "mdpalign-fuzzy/argmax":   ("mdpalign-fuzzy", "argmax", 150.0),
    "ctfalign/argmax/8":       ("ctfalign", "argmax", 8.0),
    "ctfalign/argmax/4":       ("ctfalign", "argmax", 4.0),
    "mdpalign-strict/itermax": ("mdpalign-strict", "itermax", 25.0),
    "mdpalign-fuzzy/itermax":  ("mdpalign-fuzzy", "itermax", 150.0),
    "ctfalign/itermax/8":      ("ctfalign", "itermax", 8.0),
    "simalign/itermax":        ("simalign", "itermax", None),
}

pytest.importorskip("transformers")

if not os.path.exists(FIXTURE):
    pytest.skip("golden fixture not available", allow_module_level=True)

with open(FIXTURE) as f:
    GOLD = json.load(f)["datasets"]


def _sim_path(rel):
    return os.path.join(RESEARCH_REPO, "similarity_matrices", rel)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(HF_MODEL)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.mark.slow
@pytest.mark.parametrize("rel", sorted(GOLD))
def test_matches_frozen_reference(tokenizer, rel):
    """Every case for every row of one similarity-matrix file.

    The file is streamed and all cases are checked per row: these reach ~1GB, so
    reading once per case (or loading a file whole) is not viable.
    """
    from ctfalign import align_from_similarity
    from ctfalign.wordmap import build_b2w_map

    path = _sim_path(rel)
    if not os.path.exists(path):
        pytest.skip(
            f"research matrices not found at {path}; set CTFALIGN_RESEARCH_REPO "
            f"to a checkout of document-level-word-alignment"
        )

    gold = GOLD[rel]
    n_rows = 0
    with open(path, encoding="utf-8") as f:
        for row_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sim = record.pop("similarity_matrices")
            b2w_a = build_b2w_map(tokenizer, record["text_a"])
            b2w_b = build_b2w_map(tokenizer, record["text_b"])
            for case_key, (method, mode, k) in CASES.items():
                got = align_from_similarity(
                    sim, b2w_a, b2w_b, method=method, mode=mode, k=k, max_count=2,
                )
                expected = {tuple(p) for p in gold[case_key][row_i]}
                assert set(got) == expected, f"{rel} {case_key} row {row_i} mismatch"
            n_rows += 1
            del sim, record

    assert n_rows == len(gold[next(iter(CASES))]), (
        f"{rel}: fixture has {len(gold[next(iter(CASES))])} rows, data has {n_rows}"
    )
