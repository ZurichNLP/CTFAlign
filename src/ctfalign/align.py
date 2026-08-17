"""Base alignment algorithms (SimAlign; Jalili-Sabet et al., 2020).

Both functions operate on a single token-level similarity matrix and return a
list of ``(src_token_idx, tgt_token_idx)`` pairs over the *non-special* token
sequence. They are framework-independent: ``argmax_align`` takes a torch tensor,
``iter_max`` takes a numpy array (matching the SimAlign reference).
"""
import numpy as np


def argmax_align(similarity_matrix):
    """Greedy bidirectional argmax, intersected (SimAlign argmax)."""
    en_cz = similarity_matrix.argmax(dim=1)
    cz_en = similarity_matrix.argmax(dim=0)

    intersection = []
    for i, j in enumerate(en_cz):
        j = j.item()
        if cz_en[j] == i:
            intersection.append((i, j))
    return intersection


def iter_max(sim_matrix: np.ndarray, max_count: int = 2) -> np.ndarray:
    """Iterative refinement of the argmax intersection (SimAlign itermax).

    ``max_count`` controls the number of iterations; higher means more recall.
    Reference: https://github.com/cisnlp/simalign
    """
    alpha_ratio = 0.9
    m, n = sim_matrix.shape
    forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
    backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
    inter = forward * backward.transpose()

    if min(m, n) <= 2:
        rows, cols = np.where(inter > 0)
        return list(zip(rows.tolist(), cols.tolist()))

    new_inter = np.zeros((m, n))
    count = 1
    while count < max_count:
        mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
        mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
        mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
        mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
        if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
            mask *= 0.0
            mask_zeros *= 0.0

        new_sim = sim_matrix * mask
        fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
        bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
        new_inter = fwd * bac

        if np.array_equal(inter + new_inter, inter):
            break
        inter = inter + new_inter
        count += 1
    rows, cols = np.where(inter > 0)
    return list(zip(rows.tolist(), cols.tolist()))
