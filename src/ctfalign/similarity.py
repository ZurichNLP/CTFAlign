"""Cosine similarity matrix between two sets of token embeddings."""
import torch
import torch.nn.functional as F


def get_sim_matrix(embeddings1, embeddings2):
    """Return the (m x n) cosine similarity matrix for L2-normalised rows."""
    E1 = F.normalize(embeddings1, dim=-1)
    E2 = F.normalize(embeddings2, dim=-1)
    return E1 @ E2.T
