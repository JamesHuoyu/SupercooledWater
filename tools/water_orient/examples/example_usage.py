"""Example notebook-style workflow for water_orient.

Replace the coordinate-loading block with your own trajectory reader.
"""

from __future__ import annotations

import numpy as np

from water_orient import (
    body_frames,
    dipole_vectors,
    tetrahedral_q,
    detect_hbonds,
    compute_zeta_series,
    coarse_grain_zeta,
    identify_donor_arm_jumps,
    jumps_to_frame_mask,
    lead_lag_average,
    classify_zeta_change_cause,
)


# -----------------------------------------------------------------------------
# 1. Load arrays.
# -----------------------------------------------------------------------------
# Expected shapes:
#   O_series:  (n_frames, n_mol, 3)
#   H1_series: (n_frames, n_mol, 3)
#   H2_series: (n_frames, n_mol, 3)
#   box:       (3,)
#
# Example placeholders only:
n_frames, n_mol = 10, 32
box = np.array([20.0, 20.0, 20.0])
O_series = np.zeros((n_frames, n_mol, 3))
H1_series = np.zeros((n_frames, n_mol, 3))
H2_series = np.zeros((n_frames, n_mol, 3))


# -----------------------------------------------------------------------------
# 2. Framewise hydrogen bonds and zeta.
# -----------------------------------------------------------------------------
hb_series = [
    detect_hbonds(O_series[t], H1_series[t], H2_series[t], box, r_oo_cut=3.5, angle_cut_deg=30.0)
    for t in range(n_frames)
]

zeta_series = compute_zeta_series(O_series, hb_series, box)
zeta_cg_series = np.stack(
    [coarse_grain_zeta(zeta_series[t], O_series[t], box, neighbor_cut=3.5) for t in range(n_frames)],
    axis=0,
)


# -----------------------------------------------------------------------------
# 3. Orientation metrics.
# -----------------------------------------------------------------------------
frames_series = np.stack(
    [body_frames(O_series[t], H1_series[t], H2_series[t], box) for t in range(n_frames)],
    axis=0,
)
q_series = np.stack([tetrahedral_q(O_series[t], box) for t in range(n_frames)], axis=0)
dipole_series = np.stack(
    [dipole_vectors(O_series[t], H1_series[t], H2_series[t], box) for t in range(n_frames)],
    axis=0,
)


# -----------------------------------------------------------------------------
# 4. Hydrogen-bond jump events.
# -----------------------------------------------------------------------------
arm_series = np.stack([hb.arm_acceptors for hb in hb_series], axis=0)
jumps = identify_donor_arm_jumps(arm_series, min_dwell=3)
jump_mask = jumps_to_frame_mask(jumps, n_frames=n_frames, n_mol=n_mol, mark="start")


# -----------------------------------------------------------------------------
# 5. Event-triggered average of zeta around HB jumps.
# -----------------------------------------------------------------------------
lag_data = lead_lag_average(zeta_cg_series, jump_mask, window=3)
print("tau:", lag_data["tau"])
print("mean zeta_cg around HB jumps:", lag_data["mean"])


# -----------------------------------------------------------------------------
# 6. Frame-to-frame cause classification for zeta changes.
# -----------------------------------------------------------------------------
# Example for one time step t -> t+1.
t = 0
cause = classify_zeta_change_cause(
    zeta_t0=zeta_series[t],
    zeta_t1=zeta_series[t + 1],
    O_t0=O_series[t],
    O_t1=O_series[t + 1],
    frames_t0=frames_series[t],
    frames_t1=frames_series[t + 1],
    arm_acceptors_t0=arm_series[t],
    arm_acceptors_t1=arm_series[t + 1],
    box=box,
)

for key, value in cause.items():
    if key == "category":
        uniq, cnt = np.unique(value, return_counts=True)
        print(dict(zip(uniq, cnt)))
    else:
        print(key, value.shape)
