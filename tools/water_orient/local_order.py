from __future__ import annotations

from typing import Dict

import numpy as np

from .geometry import minimum_image, pairwise_displacements_pbc, pairwise_distances_pbc
from .orientation import dipole_vectors

ArrayLike = np.ndarray


def tetrahedral_q(O: ArrayLike, box: ArrayLike) -> ArrayLike:
    """Compute the tetrahedral order parameter q for each molecule.

    q = 1 - (3/8) * sum_{j<k} (cos psi_jk + 1/3)^2
    where j,k are the four nearest oxygen neighbors of molecule i.
    """
    O = np.asarray(O, dtype=float)
    n = O.shape[0]
    distances = pairwise_distances_pbc(O, box)
    np.fill_diagonal(distances, np.inf)
    disp = pairwise_displacements_pbc(O, box)

    q = np.full(n, np.nan, dtype=float)
    for i in range(n):
        nbrs = np.argsort(distances[i])[:4]
        vecs = disp[i, nbrs, :]
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        vecs = vecs / np.where(norms == 0.0, 1.0, norms)
        acc = 0.0
        for a in range(4):
            for b in range(a + 1, 4):
                cospsi = np.clip(np.dot(vecs[a], vecs[b]), -1.0, 1.0)
                acc += (cospsi + 1.0 / 3.0) ** 2
        q[i] = 1.0 - (3.0 / 8.0) * acc
    return q


def pair_orientation_metrics(
    O: ArrayLike,
    H1: ArrayLike,
    H2: ArrayLike,
    box: ArrayLike,
    cutoff: float = 3.5,
) -> Dict[str, ArrayLike]:
    """Neighbor-based orientation metrics.

    Returns per-molecule averages over neighbors within ``cutoff``:
      - mean_mu_mu: mean dipole-dipole alignment
      - mean_mu_r: mean alignment between dipole and O-O bond direction
      - mean_abs_mu_r: absolute variant of the above
    """
    O = np.asarray(O, dtype=float)
    mu = dipole_vectors(O, H1, H2, box)
    disp = pairwise_displacements_pbc(O, box)
    dist = np.linalg.norm(disp, axis=-1)
    np.fill_diagonal(dist, np.inf)

    bond_dir = disp / np.where(dist[..., None] == 0.0, 1.0, dist[..., None])
    mask = dist <= cutoff

    mu_mu = np.einsum("id,jd->ij", mu, mu)
    mu_r = np.einsum("id,ijd->ij", mu, bond_dir)

    def masked_mean(x: ArrayLike) -> ArrayLike:
        out = np.full(x.shape[0], np.nan)
        for i in range(x.shape[0]):
            vals = x[i, mask[i]]
            if vals.size:
                out[i] = np.mean(vals)
        return out

    return {
        "mean_mu_mu": masked_mean(mu_mu),
        "mean_mu_r": masked_mean(mu_r),
        "mean_abs_mu_r": masked_mean(np.abs(mu_r)),
    }
