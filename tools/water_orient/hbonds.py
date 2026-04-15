from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .geometry import minimum_image, pairwise_displacements_pbc, pairwise_distances_pbc

ArrayLike = np.ndarray


@dataclass
class HBondResult:
    """Hydrogen-bond assignment for one frame.

    Attributes
    ----------
    arm_acceptors
        Shape (n_mol, 2). ``arm_acceptors[i, a]`` is the acceptor molecule index for
        donor molecule ``i`` and arm ``a`` (0=H1, 1=H2), or -1 if none.
    adjacency
        Symmetric shape (n_mol, n_mol). True if a hydrogen bond exists in either
        direction between molecules.
    donor_mask
        Boolean mask of shape (n_mol, 2, n_mol). ``donor_mask[i, a, j]`` indicates that
        arm ``a`` of donor ``i`` donates to acceptor ``j``.
    oo_distances
        Pairwise oxygen-oxygen distance matrix.
    linearity_cos
        Shape (n_mol, 2, n_mol). Cosines of the donor-arm linearity angle.
    """

    arm_acceptors: ArrayLike
    adjacency: ArrayLike
    donor_mask: ArrayLike
    oo_distances: ArrayLike
    linearity_cos: ArrayLike


def detect_hbonds(
    O: ArrayLike,
    H1: ArrayLike,
    H2: ArrayLike,
    box: ArrayLike,
    r_oo_cut: float = 3.5,
    angle_cut_deg: float = 30.0,
) -> HBondResult:
    """Detect hydrogen bonds for a rigid-water frame.

    Criterion: O_d--O_a <= r_oo_cut and angle(H_d - O_d, O_a - O_d) <= angle_cut_deg.
    For each donor arm, at most one acceptor is assigned (the best-aligned candidate).
    """
    O = np.asarray(O, dtype=float)
    H1 = np.asarray(H1, dtype=float)
    H2 = np.asarray(H2, dtype=float)
    n = O.shape[0]

    oo_disp = pairwise_displacements_pbc(O, box)
    oo_dist = np.linalg.norm(oo_disp, axis=-1)
    np.fill_diagonal(oo_dist, np.inf)

    oh1 = minimum_image(H1 - O, box)
    oh2 = minimum_image(H2 - O, box)
    oh = np.stack([oh1, oh2], axis=1)
    oh = oh / np.where(np.linalg.norm(oh, axis=-1, keepdims=True) == 0.0, 1.0, np.linalg.norm(oh, axis=-1, keepdims=True))

    oo_dir = oo_disp / np.where(oo_dist[..., None] == 0.0, 1.0, oo_dist[..., None])
    linearity_cos = np.einsum("iad,ijd->iaj", oh, oo_dir)

    cos_thresh = float(np.cos(np.deg2rad(angle_cut_deg)))
    donor_mask = np.zeros((n, 2, n), dtype=bool)
    arm_acceptors = -np.ones((n, 2), dtype=int)

    for i in range(n):
        for arm in range(2):
            candidates = np.flatnonzero((oo_dist[i] <= r_oo_cut) & (linearity_cos[i, arm] >= cos_thresh))
            if candidates.size == 0:
                continue
            best = candidates[np.argmax(linearity_cos[i, arm, candidates])]
            donor_mask[i, arm, best] = True
            arm_acceptors[i, arm] = int(best)

    adjacency = np.any(donor_mask, axis=1)
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)

    return HBondResult(
        arm_acceptors=arm_acceptors,
        adjacency=adjacency,
        donor_mask=donor_mask,
        oo_distances=oo_dist,
        linearity_cos=linearity_cos,
    )


def arm_acceptor_switch_mask(arm_acceptors_t0: ArrayLike, arm_acceptors_t1: ArrayLike) -> ArrayLike:
    """Return a boolean mask of donor-arm acceptor changes between two frames."""
    a0 = np.asarray(arm_acceptors_t0, dtype=int)
    a1 = np.asarray(arm_acceptors_t1, dtype=int)
    if a0.shape != a1.shape:
        raise ValueError("Input arrays must have the same shape.")
    return a0 != a1
