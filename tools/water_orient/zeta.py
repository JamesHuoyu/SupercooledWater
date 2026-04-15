from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .geometry import neighbor_list, pairwise_distances_pbc
from .hbonds import HBondResult

ArrayLike = np.ndarray


def _get_adjacency(hbonds: HBondResult | ArrayLike) -> ArrayLike:
    if isinstance(hbonds, HBondResult):
        return hbonds.adjacency
    return np.asarray(hbonds, dtype=bool)


def compute_zeta(
    O: ArrayLike,
    hbonds: HBondResult | ArrayLike,
    box: ArrayLike,
    require_hbond_within: float | None = 3.5,
) -> ArrayLike:
    """Compute Tanaka-style zeta for one frame.

    Definition used here:
      zeta_i = r(first non-HB neighbor beyond the farthest HB neighbor)
               - r(farthest HB neighbor)

    Neighbors are ranked by O-O distance. A pair is HB-connected if a hydrogen bond
    exists in either direction.
    """
    O = np.asarray(O, dtype=float)
    adjacency = _get_adjacency(hbonds)
    dist = pairwise_distances_pbc(O, box)
    np.fill_diagonal(dist, np.inf)
    n = O.shape[0]

    zeta = np.full(n, np.nan, dtype=float)
    r_last_hb = np.full(n, np.nan, dtype=float)
    r_first_nonhb = np.full(n, np.nan, dtype=float)

    for i in range(n):
        nbrs = np.argsort(dist[i])
        hb_flags = adjacency[i, nbrs]
        d_sorted = dist[i, nbrs]

        hb_idx = np.flatnonzero(hb_flags)
        if hb_idx.size == 0:
            continue

        last_idx = hb_idx[-1]
        if require_hbond_within is not None and d_sorted[last_idx] > require_hbond_within:
            continue

        nonhb_after = np.flatnonzero(~hb_flags & (np.arange(nbrs.size) > last_idx))
        if nonhb_after.size == 0:
            continue

        first_nonhb_idx = nonhb_after[0]
        r_last_hb[i] = d_sorted[last_idx]
        r_first_nonhb[i] = d_sorted[first_nonhb_idx]
        zeta[i] = r_first_nonhb[i] - r_last_hb[i]

    return zeta


def compute_zeta_series(
    O_series: ArrayLike,
    hbond_series: Sequence[HBondResult | ArrayLike],
    box: ArrayLike,
    require_hbond_within: float | None = 3.5,
) -> ArrayLike:
    """Compute zeta for each frame in a trajectory segment."""
    O_series = np.asarray(O_series, dtype=float)
    if O_series.shape[0] != len(hbond_series):
        raise ValueError("Length mismatch between O_series and hbond_series.")
    out = np.empty((O_series.shape[0], O_series.shape[1]), dtype=float)
    for t in range(O_series.shape[0]):
        out[t] = compute_zeta(O_series[t], hbond_series[t], box, require_hbond_within=require_hbond_within)
    return out


def coarse_grain_zeta(
    zeta: ArrayLike,
    O: ArrayLike,
    box: ArrayLike,
    neighbor_cut: float = 3.5,
    include_self: bool = True,
) -> ArrayLike:
    """Spatially coarse-grain zeta using first-shell oxygen neighbors."""
    z = np.asarray(zeta, dtype=float)
    nbrs = neighbor_list(O, cutoff=neighbor_cut, box=box, include_self=False)
    out = np.full_like(z, np.nan, dtype=float)
    for i, nn in enumerate(nbrs):
        group = nn
        if include_self:
            group = np.concatenate([np.asarray([i], dtype=int), nn])
        vals = z[group]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = np.mean(vals)
    return out


def hysteretic_states(
    series: ArrayLike,
    low: float,
    high: float,
    initial_state: int = 0,
) -> ArrayLike:
    """Assign binary states with hysteresis.

    Returns integer states with values:
      0 = disordered-like
      1 = structured-like

    The interval [low, high] is a memory region where the previous state is retained.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    states = np.full(x.shape, initial_state, dtype=int)
    states[0] = np.where(x[0] >= high, 1, np.where(x[0] <= low, 0, initial_state))
    for t in range(1, x.shape[0]):
        states[t] = states[t - 1]
        states[t][x[t] >= high] = 1
        states[t][x[t] <= low] = 0
    return states[:, 0] if squeeze else states
