"""Positional masking strategies applied to the similarity matrix.

- ``add_belt_mask_token_count_fixed_on_longer_side``      -> MDPAlign strict
- ``add_soft_belt_mask_token_count_fixed_on_longer_side`` -> MDPAlign fuzzy
- ``pyramid_hard_lenient2``                               -> CTFAlign

All belt functions normalise positions so the diagonal spans both corners and
``token_count`` is interpreted as a tolerance of +/-N tokens on the *longer*
side; the shorter side scales by the length ratio.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .align import argmax_align, iter_max


def add_belt_mask_token_count_fixed_on_longer_side(similarity_matrix, token_count):
    """Hard diagonal band (MDPAlign strict)."""
    m, n = similarity_matrix.shape
    i_idx = np.arange(m)[:, None] / m
    j_idx = np.arange(n)[None, :] / n
    dist = i_idx - j_idx
    mask = torch.tensor(np.abs(dist) <= token_count / max(m, n))
    return similarity_matrix * mask


def add_soft_belt_mask_token_count_fixed_on_longer_side(similarity_matrix, token_count):
    """Soft (Gaussian) diagonal band (MDPAlign fuzzy)."""
    m, n = similarity_matrix.shape
    i_idx = np.arange(m)[:, None] / m
    j_idx = np.arange(n)[None, :] / n
    dist = i_idx - j_idx
    sigma = token_count / max(m, n)
    weights = torch.tensor(np.exp(-0.5 * (dist / sigma) ** 2))
    return similarity_matrix * weights


def pyramid_hard_lenient2(similarity_matrix, method, width=0, drop_negative=True):
    """Coarse-to-fine pyramid masking with lenient recovery (CTFAlign).

    Starts from a 2x2 grid and halves the block size each level. Aligned coarse
    blocks (plus a +/-``width`` buffer) are retained; any ``(row, col)`` at the
    intersection of an empty block-row and empty block-column is recovered with
    the same +/-``width`` buffer. Returns the token-level alignment at the
    finest (1x1) level.

    ``method`` is ``"simalign-argmax"`` or ``"simalign-itermax"``.
    ``drop_negative`` discards pairs whose similarity is not strictly positive
    at every level.
    """
    m, n = similarity_matrix.shape
    block_h = max(m // 2, 1)
    block_w = max(n // 2, 1)

    while block_h >= 1 and block_w >= 1:
        n_blocks_h = (m + block_h - 1) // block_h
        n_blocks_w = (n + block_w - 1) // block_w

        if block_h == 1 and block_w == 1:
            coarse = similarity_matrix
        else:
            coarse = F.avg_pool2d(
                similarity_matrix[None, None].clamp(min=0),
                kernel_size=(block_h, block_w),
                stride=(block_h, block_w),
                ceil_mode=True,
                count_include_pad=False,
            ).squeeze(0, 1)

        if method == 'simalign-argmax':
            coarse_alignment = argmax_align(coarse, drop_negative=drop_negative)
        elif method == 'simalign-itermax':
            coarse_alignment = iter_max(coarse.cpu().numpy(),
                                        drop_negative=drop_negative)
        else:
            raise NotImplementedError("Method not implemented for pyramid_hard_lenient2.")

        if block_h == 1 and block_w == 1:
            return coarse_alignment

        coarse_mask = torch.zeros(n_blocks_h, n_blocks_w, device=similarity_matrix.device)
        for ci, cj in coarse_alignment:
            r0 = max(ci - width, 0);  r1 = min(ci + width + 1, n_blocks_h)
            c0 = max(cj - width, 0);  c1 = min(cj + width + 1, n_blocks_w)
            coarse_mask[r0:r1, c0:c1] = 1.0

        empty_rows = coarse_mask.sum(dim=1) == 0
        empty_cols = coarse_mask.sum(dim=0) == 0
        for ci, cj in (empty_rows[:, None] & empty_cols[None, :]).nonzero(as_tuple=False).tolist():
            r0 = max(ci - width, 0);  r1 = min(ci + width + 1, n_blocks_h)
            c0 = max(cj - width, 0);  c1 = min(cj + width + 1, n_blocks_w)
            coarse_mask[r0:r1, c0:c1] = 1.0

        mask = F.interpolate(
            coarse_mask[None, None],
            size=(n_blocks_h * block_h, n_blocks_w * block_w),
            mode='nearest'
        ).squeeze(0, 1)[:m, :n]
        similarity_matrix = similarity_matrix * mask

        block_h = max(block_h // 2, 1)
        block_w = max(block_w // 2, 1)
