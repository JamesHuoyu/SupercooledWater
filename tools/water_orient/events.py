from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from .geometry import pairwise_distances_pbc
from .hbonds import arm_acceptor_switch_mask
from .orientation import angular_displacement, angular_displacement_from_frames

ArrayLike = np.ndarray


def lead_lag_average(series: ArrayLike, event_mask: ArrayLike, window: int) -> Dict[str, ArrayLike]:
    """Average a time series around per-particle event times.

    Parameters
    ----------
    series
        Shape (n_frames, n_mol).
    event_mask
        Boolean array with the same shape as ``series``.
    window
        Number of frames before/after the event.

    Returns
    -------
    dict with keys ``tau``, ``mean``, ``count``.
    """
    x = np.asarray(series, dtype=float)
    m = np.asarray(event_mask, dtype=bool)
    if x.shape != m.shape:
        raise ValueError("series and event_mask must have the same shape.")

    n_frames, n_mol = x.shape
    tau = np.arange(-window, window + 1)
    acc = np.zeros_like(tau, dtype=float)
    cnt = np.zeros_like(tau, dtype=int)

    event_frames, event_mols = np.where(m)
    for t0, i in zip(event_frames, event_mols):
        s = max(0, t0 - window)
        e = min(n_frames, t0 + window + 1)
        tau_slice = np.arange(s - t0, e - t0)
        vals = x[s:e, i]
        valid = np.isfinite(vals)
        inds = tau_slice + window
        acc[inds[valid]] += vals[valid]
        cnt[inds[valid]] += 1

    mean = np.full_like(acc, np.nan, dtype=float)
    nz = cnt > 0
    mean[nz] = acc[nz] / cnt[nz]
    return {"tau": tau, "mean": mean, "count": cnt}


def cumulative_disorder_exposure(state_or_indicator: ArrayLike, dt: float = 1.0) -> ArrayLike:
    """Cumulative time spent in a disordered state up to each frame.

    Input should be 0/1-like with 1 meaning disordered exposure.
    """
    x = np.asarray(state_or_indicator, dtype=float)
    return np.cumsum(x, axis=0) * dt


def conditional_event_probability(state_mask: ArrayLike, event_mask: ArrayLike) -> Dict[str, float]:
    """Estimate P(event | state) and P(event | not state)."""
    s = np.asarray(state_mask, dtype=bool)
    e = np.asarray(event_mask, dtype=bool)
    if s.shape != e.shape:
        raise ValueError("state_mask and event_mask must have the same shape.")

    p1 = float(np.mean(e[s])) if np.any(s) else np.nan
    p0 = float(np.mean(e[~s])) if np.any(~s) else np.nan
    odds_ratio = np.nan
    if np.isfinite(p1) and np.isfinite(p0) and 0.0 < p1 < 1.0 and 0.0 < p0 < 1.0:
        odds_ratio = (p1 / (1.0 - p1)) / (p0 / (1.0 - p0))
    return {"p_event_given_state": p1, "p_event_given_not_state": p0, "odds_ratio": odds_ratio}


def classify_zeta_change_cause(
    zeta_t0: ArrayLike,
    zeta_t1: ArrayLike,
    O_t0: ArrayLike,
    O_t1: ArrayLike,
    frames_t0: ArrayLike,
    frames_t1: ArrayLike,
    arm_acceptors_t0: ArrayLike,
    arm_acceptors_t1: ArrayLike,
    box: ArrayLike,
    radial_shell_k: int = 6,
    zeta_thresh: float = 0.15,
    rot_thresh_deg: float = 20.0,
    radial_thresh: float = 0.12,
) -> Dict[str, ArrayLike]:
    """Classify likely causes of per-molecule zeta changes between two frames.

    Heuristic categories:
      - stable:            |Δzeta| < zeta_thresh
      - orientational:     HB-arm switch and/or large rotation, but small radial-shell change
      - translational:     large radial-shell change with no HB-arm switch and small rotation
      - mixed:             both orientational/topological and translational signatures
      - unresolved:        large Δzeta but weak evidence for all above

    The goal is not to prove causality, but to separate obvious candidates for
    non-translational irreversible zeta changes.
    """
    z0 = np.asarray(zeta_t0, dtype=float)
    z1 = np.asarray(zeta_t1, dtype=float)
    dz = z1 - z0

    rot = angular_displacement_from_frames(frames_t0, frames_t1, degrees=True)
    switch_mask = arm_acceptor_switch_mask(arm_acceptors_t0, arm_acceptors_t1)
    any_switch = np.any(switch_mask, axis=1)

    d0 = pairwise_distances_pbc(O_t0, box)
    d1 = pairwise_distances_pbc(O_t1, box)
    np.fill_diagonal(d0, np.inf)
    np.fill_diagonal(d1, np.inf)

    n = z0.shape[0]
    radial_rms = np.full(n, np.nan)
    for i in range(n):
        idx0 = np.argsort(d0[i])
        idx0 = idx0[np.isfinite(d0[i, idx0])][:radial_shell_k]
        idx1 = np.argsort(d1[i])
        idx1 = idx1[np.isfinite(d1[i, idx1])][:radial_shell_k]
        union = np.unique(np.concatenate([idx0, idx1]))
        if union.size == 0:
            continue
        radial_rms[i] = np.sqrt(np.mean((d1[i, union] - d0[i, union]) ** 2))

    large_dz = np.abs(dz) >= zeta_thresh
    orient_sig = any_switch | (rot >= rot_thresh_deg)
    trans_sig = radial_rms >= radial_thresh

    category = np.full(n, "stable", dtype=object)
    category[large_dz & orient_sig & ~trans_sig] = "orientational"
    category[large_dz & ~orient_sig & trans_sig] = "translational"
    category[large_dz & orient_sig & trans_sig] = "mixed"
    category[large_dz & ~orient_sig & ~trans_sig] = "unresolved"

    return {
        "delta_zeta": dz,
        "rotation_deg": rot,
        "arm_switch": any_switch,
        "radial_shell_rms": radial_rms,
        "category": category,
    }
