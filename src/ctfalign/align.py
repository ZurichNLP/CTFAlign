"""Base alignment algorithms (SimAlign; Jalili-Sabet et al., 2020).

Both functions operate on a single token-level similarity matrix and return a
list of ``(src_token_idx, tgt_token_idx)`` pairs over the *non-special* token
sequence. They are framework-independent: ``argmax_align`` takes a torch tensor,
``iter_max`` takes a numpy array (matching the SimAlign reference).

Both accept ``drop_negative``: when set, pairs whose similarity is not strictly
positive are discarded. A negative cosine similarity means the two token
embeddings point in opposite directions, so it should never license an
alignment -- even when it happens to be the row/column maximum.
"""
import numpy as np


def argmax_align(similarity_matrix, drop_negative=False):
    """Greedy bidirectional argmax, intersected (SimAlign argmax)."""
    en_cz = similarity_matrix.argmax(dim=1)
    cz_en = similarity_matrix.argmax(dim=0)

    intersection = []
    for i, j in enumerate(en_cz):
        j = j.item()
        if cz_en[j] == i:
            if drop_negative and float(similarity_matrix[i, j]) <= 0:
                continue
            intersection.append((i, j))
    return intersection


def _keep_positive(pairs, matrix, drop_negative):
    """Drop pairs whose similarity is not strictly positive (see argmax_align)."""
    if not drop_negative:
        return pairs
    return [(i, j) for i, j in pairs if float(matrix[i, j]) > 0]


def _onehot_rows(idx: np.ndarray, n: int) -> np.ndarray:
    """Row-wise one-hot matrix, i.e. ``np.eye(n)[idx]`` without the n x n identity.

    The identity is a transient the result never keeps: only ``len(idx)`` of its
    rows are gathered. Building the result directly avoids allocating it (at the
    deepest pyramid level of a long document it runs to hundreds of MB).
    """
    out = np.zeros((idx.shape[0], n))
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def iter_max(sim_matrix: np.ndarray, max_count: int = 2,
             drop_negative: bool = False) -> np.ndarray:
    """Iterative refinement of the argmax intersection (SimAlign itermax).

    ``max_count`` controls the number of iterations; higher means more recall.
    Reference: https://github.com/cisnlp/simalign
    """
    alpha_ratio = 0.9
    m, n = sim_matrix.shape
    forward = _onehot_rows(sim_matrix.argmax(axis=1), n)   # m x n
    backward = _onehot_rows(sim_matrix.argmax(axis=0), m)  # n x m
    inter = forward * backward.transpose()

    if min(m, n) <= 2:
        rows, cols = np.where(inter > 0)
        return _keep_positive(list(zip(rows.tolist(), cols.tolist())),
                              sim_matrix, drop_negative)

    count = 1
    while count < max_count:
        # mask_x / mask_y are constant along one axis, so they are kept as a
        # column and a row vector and expanded by broadcasting rather than tiled.
        mask_x = 1.0 - inter.sum(1)[:, np.newaxis].clip(0.0, 1.0)  # m x 1
        mask_y = 1.0 - inter.sum(0)[np.newaxis, :].clip(0.0, 1.0)  # 1 x n
        if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
            # Every row (or every column) is already aligned. The original
            # zeroed both masks here and ran the iteration out on all-zero
            # arrays, which yields an empty new_inter and breaks on the next
            # equality check regardless -- so break directly.
            break

        mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
        mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))

        new_sim = sim_matrix * mask
        fwd = _onehot_rows(new_sim.argmax(axis=1), n) * mask_zeros
        bac = _onehot_rows(new_sim.argmax(axis=0), m).transpose() * mask_zeros
        new_inter = fwd * bac

        if np.array_equal(inter + new_inter, inter):
            break
        inter = inter + new_inter
        count += 1
    rows, cols = np.where(inter > 0)
    return _keep_positive(list(zip(rows.tolist(), cols.tolist())),
                          sim_matrix, drop_negative)
