from __future__ import annotations

from typing import List, Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback if scipy unavailable
    cKDTree = None

ArrayLike = np.ndarray


def _as_box(box: Sequence[float] | np.ndarray) -> np.ndarray:
    box_arr = np.asarray(box, dtype=float)
    if box_arr.shape != (3,):
        raise ValueError("Only orthorhombic boxes with shape (3,) are supported.")
    return box_arr


def wrap_positions(positions: ArrayLike, box: Sequence[float] | np.ndarray) -> ArrayLike:
    """Wrap positions into [0, L) for an orthorhombic periodic box."""
    box_arr = _as_box(box)
    return np.mod(np.asarray(positions, dtype=float), box_arr)


def minimum_image(displacements: ArrayLike, box: Sequence[float] | np.ndarray) -> ArrayLike:
    """Apply the minimum-image convention to displacement vectors."""
    box_arr = _as_box(box)
    disp = np.asarray(displacements, dtype=float)
    return disp - box_arr * np.round(disp / box_arr)


def pairwise_displacements_pbc(positions: ArrayLike, box: Sequence[float] | np.ndarray) -> ArrayLike:
    """Return pairwise displacement matrix with minimum image.

    Parameters
    ----------
    positions
        Shape (n, 3).
    box
        Orthorhombic box lengths, shape (3,).

    Returns
    -------
    disp
        Shape (n, n, 3), where disp[i, j] = r_j - r_i under minimum image.
    """
    pos = np.asarray(positions, dtype=float)
    disp = pos[None, :, :] - pos[:, None, :]
    return minimum_image(disp, box)


def pairwise_distances_pbc(positions: ArrayLike, box: Sequence[float] | np.ndarray) -> ArrayLike:
    """Return pairwise distance matrix with periodic boundaries."""
    disp = pairwise_displacements_pbc(positions, box)
    return np.linalg.norm(disp, axis=-1)


def neighbor_list(
    positions: ArrayLike,
    cutoff: float,
    box: Sequence[float] | np.ndarray,
    include_self: bool = False,
) -> List[np.ndarray]:
    """Build a periodic neighbor list.

    Uses scipy.spatial.cKDTree when available, otherwise falls back to an O(N^2)
    brute-force distance search.
    """
    pos = wrap_positions(np.asarray(positions, dtype=float), box)
    n = pos.shape[0]
    if cKDTree is not None:
        tree = cKDTree(pos, boxsize=_as_box(box))
        neighbors = tree.query_ball_point(pos, r=float(cutoff))
        out: List[np.ndarray] = []
        for i, nbrs in enumerate(neighbors):
            arr = np.asarray(nbrs, dtype=int)
            if not include_self:
                arr = arr[arr != i]
            out.append(arr)
        return out

    distances = pairwise_distances_pbc(pos, box)
    out = []
    for i in range(n):
        mask = distances[i] <= cutoff
        if not include_self:
            mask[i] = False
        out.append(np.flatnonzero(mask))
    return out
