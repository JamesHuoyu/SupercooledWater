"""Utilities for orientation–zeta coupling analysis in supercooled water.

This package is designed for TIP4P/Ice-like rigid water trajectories and focuses on
three coupled aspects:

1. Molecular orientation and rotational change.
2. Hydrogen-bond topology and donor-arm jump events.
3. Tanaka-style zeta order parameter and its coupling to the above.

All core functions operate on NumPy arrays so they can be imported directly from a
Jupyter notebook without a heavy trajectory framework dependency.
"""

from .geometry import minimum_image, wrap_positions, pairwise_distances_pbc, neighbor_list
from .orientation import (
    body_frames,
    dipole_vectors,
    angular_displacement,
    angular_displacement_from_frames,
    rotational_correlation,
)
from .local_order import tetrahedral_q, pair_orientation_metrics
from .hbonds import HBondResult, detect_hbonds, arm_acceptor_switch_mask
from .zeta import compute_zeta, compute_zeta_series, coarse_grain_zeta, hysteretic_states
from .jumps import JumpEvent, identify_donor_arm_jumps, jumps_to_frame_mask
from .events import (
    lead_lag_average,
    cumulative_disorder_exposure,
    conditional_event_probability,
    classify_zeta_change_cause,
)

__all__ = [
    "minimum_image",
    "wrap_positions",
    "pairwise_distances_pbc",
    "neighbor_list",
    "body_frames",
    "dipole_vectors",
    "angular_displacement",
    "angular_displacement_from_frames",
    "rotational_correlation",
    "tetrahedral_q",
    "pair_orientation_metrics",
    "HBondResult",
    "detect_hbonds",
    "arm_acceptor_switch_mask",
    "compute_zeta",
    "compute_zeta_series",
    "coarse_grain_zeta",
    "hysteretic_states",
    "JumpEvent",
    "identify_donor_arm_jumps",
    "jumps_to_frame_mask",
    "lead_lag_average",
    "cumulative_disorder_exposure",
    "conditional_event_probability",
    "classify_zeta_change_cause",
]
