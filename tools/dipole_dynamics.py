"""
Dipole Orientational Dynamics Analysis Pipeline
================================================
System : Supercooled TIP4P/Ice water at 225 K under SLLOD xy-plane shear
Purpose: Compute orientational correlation functions, angular displacements,
         translational displacements, their joint distributions, and the
         coupling between rotational and translational mobility.

Physical background
-------------------
In supercooled water, water molecules undergo slow cooperative reorientations
coupled to cage-breaking translational events.  This code quantifies the
coupling using:

  * C1(t) = ⟨u(t0+t) · u(t0)⟩          — first-rank dipole ACF
  * C2(t) = ⟨P2(u(t0+t) · u(t0))⟩       — second-rank (infrared-active) ACF
  * ⟨θ²(t)⟩                              — mean-squared angular displacement
  * P(Δr, θ; t)                          — joint translational-rotational PDF
  * R(t) = P_overlap / P_rand             — correlated jump ratio

Units
-----
Distances : Ångström (Å)
Times     : femtoseconds (fs) internally; picoseconds (ps) in output
Angles    : radians internally; degrees in output plots

Dependencies
------------
numpy, scipy, matplotlib, MDAnalysis ≥ 2.0

Usage
-----
  python dipole_dynamics.py  topology.pdb  trajectory.lammpsdump
  python dipole_dynamics.py  --test          # synthetic self-test
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.base import AnalysisBase
except ImportError:
    raise ImportError("MDAnalysis is required.  pip install MDAnalysis")


# ============================================================
#  SECTION 0 – Simulation and analysis parameters
# ============================================================

class SimParams:
    """
    Central repository of all physical and numerical parameters.
    Edit here; nothing else in the code needs changing.
    """
    # ---- Simulation ----
    L: float          = 50.0    # Box length (Å), assumed cubic at t=0
    T: float          = 225.0   # Temperature (K)
    dt_dump: float    = 200.0    # Dump interval (fs)
    shear_rate: float = 0    # γ̇  (Å^{-1} fs^{-1} in LAMMPS real units)

    # ---- Molecule topology ----
    # TIP4P/Ice: 1 O + 2 H per molecule. Atom ordering in dump assumed:
    # O H H  O H H  O H H  ...  (one block per molecule)
    atoms_per_molecule: int = 3
    O_index: int  = 0   # local index of oxygen within molecule
    H1_index: int = 1
    H2_index: int = 2

    # ---- Time origins ----
    n_t0: int         = 100      # Number of reference time origins
    t0_start_frac: float = 0.10 # Skip first 20 % as equilibration/transient

    # ---- Jump thresholds (adjustable) ----
    theta_c_deg: float = 48.0   # Rotational jump threshold (degrees)
    r_c_angstrom: float = 0.81   # Translational jump threshold (Å)
    # Alternative: set use_percentile_thresholds = True to use
    # the (e.g.) 85th percentile of each distribution at t*
    use_percentile_thresholds: bool = False
    threshold_percentile: float = 85.0

    # ---- Histograms ----
    n_theta_bins: int = 60      # Bins for θ in [0, π]
    n_r_bins: int     = 60      # Bins for Δr

    # ---- Shear ----
    remove_affine: bool = True  # Subtract affine displacement from Δr

    # ---- Output ----
    fig_dpi: int = 300


P = SimParams()   # global instance — override fields as needed


# ============================================================
#  SECTION 1 – Dipole computation
# ============================================================

def compute_dipoles(
    pos_O:  np.ndarray,   # (N_mol, 3)  oxygen positions
    pos_H1: np.ndarray,   # (N_mol, 3)  first hydrogen
    pos_H2: np.ndarray,   # (N_mol, 3)  second hydrogen
) -> np.ndarray:
    """
    Compute normalised molecular dipole unit vectors.

    Physical definition
    -------------------
    The TIP4P/Ice dipole points from the oxygen toward the
    mid-point of the two hydrogens:

        u_i  =  (r_H1 + r_H2)/2  −  r_O

    After computing this raw vector we normalise it to a unit
    vector.  The direction (not magnitude) is what enters all
    correlation functions below.

    Numerical safety
    ----------------
    If for any molecule the two hydrogens are exactly at the same
    position as the oxygen (degenerate case), the norm is zero.
    We guard against division-by-zero by replacing zero norms
    with 1 (the direction becomes ill-defined; a warning is issued).

    Parameters
    ----------
    pos_O, pos_H1, pos_H2 : ndarray (N_mol, 3)

    Returns
    -------
    u : ndarray (N_mol, 3)  unit dipole vectors
    """
    midH = 0.5 * (pos_H1 + pos_H2)       # mid-point of H1, H2
    raw = midH - pos_O                     # raw dipole vector

    norms = np.linalg.norm(raw, axis=1)    # (N_mol,)
    zero_mask = norms < 1e-12
    if zero_mask.any():
        n_bad = zero_mask.sum()
        warnings.warn(
            f"compute_dipoles: {n_bad} molecule(s) have near-zero dipole "
            "magnitude.  Check atom ordering / unwrapping."
        )
        norms[zero_mask] = 1.0             # avoid division by zero

    u = raw / norms[:, np.newaxis]         # (N_mol, 3)  unit vectors
    return u


# ============================================================
#  SECTION 2 – Orientational correlation functions
# ============================================================

def compute_orientation_correlation(
    dipoles: np.ndarray,    # (n_frames, N_mol, 3)
    lag_steps: np.ndarray,  # 1-D int array of lag frame counts
    t0_indices: list,       # list of int, reference frame indices
) -> tuple:
    """
    Compute C1(t) and C2(t) averaged over molecules and time origins.

    C1(t) = ⟨u(t0+t) · u(t0)⟩_{i, t0}
    C2(t) = ⟨P2(u(t0+t) · u(t0))⟩_{i, t0}     P2(x) = (3x²−1)/2

    Physical meaning
    ----------------
    C1(t) decays with the single-molecule reorientation time τ1
    (related to dielectric relaxation).  C2(t) decays faster (τ2 < τ1)
    and is probed by NMR and Raman spectroscopy.  In supercooled water
    both functions show stretched-exponential behaviour.

    Numerical details
    -----------------
    Dot products are clipped to [−1, 1] to prevent arccos domain errors
    and to absorb floating-point rounding.  The average is over all
    N_mol molecules for each t0, then over all t0.

    Parameters
    ----------
    dipoles   : (n_frames, N_mol, 3)  unit dipole vectors
    lag_steps : 1-D int array         lag times in frame units
    t0_indices: list of int           frame indices of time origins

    Returns
    -------
    C1 : ndarray (n_lags,)   first-rank ACF
    C2 : ndarray (n_lags,)   second-rank ACF
    """
    n_lags = len(lag_steps)
    C1 = np.zeros(n_lags)
    C2 = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)

    for li, lag in enumerate(lag_steps):
        c1_sum = 0.0
        c2_sum = 0.0
        n = 0
        for t0 in t0_indices:
            t1 = t0 + lag
            if t1 >= dipoles.shape[0]:
                continue
            # dot product per molecule: (N_mol,)
            dot = np.einsum("ij,ij->i", dipoles[t0], dipoles[t1])
            dot = np.clip(dot, -1.0, 1.0)   # numerical safety
            c1_sum += dot.mean()
            c2_sum += (0.5 * (3.0 * dot ** 2 - 1.0)).mean()
            n += 1
        if n > 0:
            C1[li] = c1_sum / n
            C2[li] = c2_sum / n
            counts[li] = n

    return C1, C2


# ============================================================
#  SECTION 3 – Angular displacement
# ============================================================

def compute_angular_displacement(
    dipoles: np.ndarray,    # (n_frames, N_mol, 3)
    lag_steps: np.ndarray,
    t0_indices: list,
) -> tuple:
    """
    Compute the per-molecule angular displacement θ_i(t0, t).

    Definition
    ----------
        θ_i(t0, t) = arccos( u_i(t0+t) · u_i(t0) )

    θ lives in [0, π] (radians).

    Returns
    -------
    theta_msad : ndarray (n_lags,)         mean squared angular displacement ⟨θ²⟩
    theta_all  : list of ndarray (N_mol,)  per-molecule θ for each lag time
                 (used by joint distribution and jump analysis)
    """
    n_lags = len(lag_steps)
    theta_msad = np.zeros(n_lags)    # ⟨θ²(t)⟩
    theta_all = [None] * n_lags      # store for joint distribution

    for li, lag in enumerate(lag_steps):
        theta_accum = []
        for t0 in t0_indices:
            t1 = t0 + lag
            if t1 >= dipoles.shape[0]:
                continue
            dot = np.einsum("ij,ij->i", dipoles[t0], dipoles[t1])
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)           # (N_mol,) in [0, π]
            theta_accum.append(theta)

        if theta_accum:
            stacked = np.concatenate(theta_accum)   # (n_t0 * N_mol,)
            theta_msad[li] = np.mean(stacked ** 2)
            # Store per-molecule arrays (one per t0) for joint use
            theta_all[li] = np.array(theta_accum)   # (n_valid_t0, N_mol)

    return theta_msad, theta_all


def compute_theta_distribution(
    theta_all: list,
    n_bins: int = P.n_theta_bins,
) -> tuple:
    """
    Compute the distribution P(θ, t) for each lag time.

    The distribution is normalised such that ∫ P(θ) dθ = 1.
    Note: for random orientations in 3D, the marginal distribution
    of θ is sin(θ), not uniform.  We do NOT impose this prior —
    we compute the empirical distribution.

    Returns
    -------
    bin_centres : ndarray (n_bins,)          in radians
    P_theta     : ndarray (n_lags, n_bins)   normalised P(θ, t)
    """
    bin_edges = np.linspace(0.0, np.pi, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_lags = len(theta_all)
    P_theta = np.zeros((n_lags, n_bins))

    for li, th_arr in enumerate(theta_all):
        if th_arr is None:
            continue
        flat = th_arr.ravel()
        counts, _ = np.histogram(flat, bins=bin_edges)
        total = counts.sum()
        if total > 0:
            dtheta = bin_edges[1] - bin_edges[0]
            P_theta[li] = counts / (total * dtheta)   # normalised PDF

    return bin_centres, P_theta


# ============================================================
#  SECTION 4 – Translational displacement
# ============================================================

def compute_displacements(
    pos_O: np.ndarray,         # (n_frames, N_mol, 3)  oxygen positions (unwrapped)
    lag_steps: np.ndarray,
    t0_indices: list,
    shear_rate: float = P.shear_rate,
    dt_dump: float = P.dt_dump,
    remove_affine: bool = P.remove_affine,
) -> tuple:
    """
    Compute translational displacement magnitudes |Δr_i(t0, t)|.

    We use oxygen positions as the molecular centre.  Unwrapped
    coordinates are passed in, so no minimum-image convention is needed.

    Affine correction (shear case)
    ------------------------------
    Under simple shear in the xy-plane (SLLOD):
        Δr^{affine}_{i,x} = γ̇ · (\sum y_i(\tau)) · dt
    This is subtracted when remove_affine=True so that only the
    genuine cage-breaking (non-affine) motion is analysed.

    Returns
    -------
    dr_all   : list of ndarray  per-molecule |Δr| for each lag
               each entry shape (n_valid_t0, N_mol)
    dr_msd   : ndarray (n_lags,)  mean-squared displacement ⟨|Δr|²⟩
    """
    n_lags = len(lag_steps)
    dr_all = [None] * n_lags
    dr_msd = np.zeros(n_lags)

    for li, lag in enumerate(lag_steps):
        dr_accum = []
        for t0 in t0_indices:
            t1 = t0 + lag
            if t1 >= pos_O.shape[0]:
                continue
            dr_vec = pos_O[t1] - pos_O[t0]        # (N_mol, 3)

            if remove_affine:
                # fix it into sum form by everystep
                delta_t = lag * dt_dump            # elapsed time (fs)
                y_avg = pos_O[t0:t0+lag, :, 1].mean(axis=0) # shape like (N_mol,)
                # todo: index a magic number for system: y_ref = 25 (half box length)
                dr_vec[:, 0] -= shear_rate * (y_avg - 25.0) * delta_t
                
                # y_t0 = pos_O[t0, :, 1]            # y-coords at t0 (N_mol,)
                # dr_vec[:, 0] -= shear_rate * y_t0 * delta_t  # subtract affine x

            dr_mag = np.linalg.norm(dr_vec, axis=1)   # (N_mol,)
            dr_accum.append(dr_mag)

        if dr_accum:
            stacked = np.array(dr_accum)           # (n_t0, N_mol)
            dr_all[li] = stacked
            dr_msd[li] = np.mean(stacked ** 2)

    return dr_all, dr_msd


# ============================================================
#  SECTION 5 – Joint distribution P(Δr, θ; t)
# ============================================================

def compute_joint_distribution(
    dr_all: list,       # list of ndarray (n_t0, N_mol)
    theta_all: list,    # list of ndarray (n_t0, N_mol)
    n_r_bins: int = P.n_r_bins,
    n_theta_bins: int = P.n_theta_bins,
) -> tuple:
    """
    Compute the joint PDF P(Δr, θ; t) at each lag time.

    The joint distribution reveals whether large angular jumps coincide
    with large translational displacements (coupling) or are independent.
    In a perfectly decoupled system, P(Δr, θ) = P_r(Δr) × P_θ(θ).

    Normalisation: ∫∫ P(Δr, θ) d(Δr) d(θ) = 1

    Returns
    -------
    r_edges     : ndarray (n_r_bins+1,)
    theta_edges : ndarray (n_theta_bins+1,)
    P_joint     : list of ndarray (n_r_bins, n_theta_bins)  one per lag
    """
    n_lags = len(dr_all)
    P_joint = [None] * n_lags

    # Determine global r range from all lag times for consistent axes
    r_max_global = 0.0
    for dr_arr in dr_all:
        if dr_arr is not None:
            r_max_global = max(r_max_global, dr_arr.max())
    r_max_global = max(r_max_global, 1e-3)

    r_edges = np.linspace(0.0, r_max_global * 1.05, n_r_bins + 1)
    theta_edges = np.linspace(0.0, np.pi, n_theta_bins + 1)

    for li in range(n_lags):
        if dr_all[li] is None or theta_all[li] is None:
            continue
        # Flatten over (t0, molecule) axis
        r_flat = dr_all[li].ravel()
        t_flat = theta_all[li].ravel()

        # 2-D histogram: rows = Δr bins, cols = θ bins
        H, _, _ = np.histogram2d(r_flat, t_flat,
                                  bins=[r_edges, theta_edges])
        # Normalise: divide by total weight and bin areas
        total = H.sum()
        if total > 0:
            dr_bin = r_edges[1] - r_edges[0]
            dt_bin = theta_edges[1] - theta_edges[0]
            P_joint[li] = H / (total * dr_bin * dt_bin)
        else:
            P_joint[li] = H

    return r_edges, theta_edges, P_joint


# ============================================================
#  SECTION 6 & 7 – Jump definitions and mobility fields
# ============================================================

def define_thresholds(
    dr_all: list,
    theta_all: list,
    lag_idx_peak: int,
    params: SimParams = P,
) -> tuple:
    """
    Determine jump thresholds θ_c and r_c.

    Two modes:
    1. Fixed (default): use params.theta_c_deg and params.r_c_angstrom
    2. Percentile-based: use the given percentile of the joint distribution
       at the peak lag time (where coupling is expected to be strongest).

    Returns
    -------
    theta_c : float  (radians)
    r_c     : float  (Å)
    """
    if params.use_percentile_thresholds:
        pct = params.threshold_percentile
        dr_peak = dr_all[lag_idx_peak]
        th_peak = theta_all[lag_idx_peak]
        if dr_peak is not None and th_peak is not None:
            r_c = np.percentile(dr_peak.ravel(), pct)
            theta_c = np.percentile(th_peak.ravel(), pct)
        else:
            warnings.warn("Percentile thresholds: peak lag has no data. Using fixed.")
            theta_c = np.deg2rad(params.theta_c_deg)
            r_c = params.r_c_angstrom
        print(f"  Percentile thresholds ({pct}th pct): "
              f"θ_c = {np.rad2deg(theta_c):.1f}°, r_c = {r_c:.2f} Å")
    else:
        theta_c = np.deg2rad(params.theta_c_deg)
        r_c = params.r_c_angstrom
        print(f"  Fixed thresholds: θ_c = {params.theta_c_deg:.1f}°, r_c = {r_c:.2f} Å")

    return theta_c, r_c


def compute_mobility_fields(
    dr_arr: np.ndarray,     # (n_t0, N_mol)  translational displacement
    theta_arr: np.ndarray,  # (n_t0, N_mol)  angular displacement
    theta_c: float,         # radians
    r_c: float,             # Å
) -> tuple:
    """
    Compute binary jump mobility fields.

        m_i^rot   = 1 if θ_i > θ_c else 0
        m_i^trans = 1 if Δr_i > r_c else 0

    Returns
    -------
    m_rot   : ndarray (n_t0, N_mol)  float binary
    m_trans : ndarray (n_t0, N_mol)  float binary
    """
    m_rot   = (theta_arr > theta_c).astype(float)
    m_trans = (dr_arr    > r_c    ).astype(float)
    return m_rot, m_trans


# ============================================================
#  SECTION 8 – Rotation-translation coupling
# ============================================================

def compute_jump_correlation(
    dr_all: list,
    theta_all: list,
    theta_c: float,
    r_c: float,
) -> dict:
    """
    Compute all coupling statistics between rotational and translational jumps.

    Statistics computed (for each lag time)
    ----------------------------------------
    1. P_overlap  = ⟨m^rot · m^trans⟩     fraction of doubly-jumping molecules
    2. P_rand     = ⟨m^rot⟩ · ⟨m^trans⟩   random expectation if independent
    3. R(t)       = P_overlap / P_rand      coupling ratio (R=1 → independent)
    4. Pearson r  = corr(θ, Δr)            linear correlation
    5. <θ | Δr>   conditional mean angle    for each Δr bin
    6. <Δr | θ>   conditional mean distance  for each θ bin

    Physical interpretation
    -----------------------
    R(t) > 1 signals that rotational and translational jumps are NOT
    independent — molecules that jump translationally also preferentially
    undergo large angular reorientations.  This coupling is a hallmark
    of cooperative dynamics in supercooled liquids.

    Returns
    -------
    stats : dict with keys:
        P_overlap, P_rand, R, pearson_r, pearson_p,
        cond_theta_given_dr, cond_dr_given_theta  (all arrays of shape n_lags)
    """
    n_lags = len(dr_all)
    P_overlap = np.full(n_lags, np.nan)
    P_rand    = np.full(n_lags, np.nan)
    R         = np.full(n_lags, np.nan)
    pearson_r = np.full(n_lags, np.nan)
    pearson_p = np.full(n_lags, np.nan)

    for li in range(n_lags):
        if dr_all[li] is None or theta_all[li] is None:
            continue
        dr_flat = dr_all[li].ravel()
        th_flat = theta_all[li].ravel()

        # ----- binary fields -----
        m_rot   = (th_flat > theta_c).astype(float)
        m_trans = (dr_flat > r_c    ).astype(float)

        f_rot   = m_rot.mean()     # fraction of rotational jumpers
        f_trans = m_trans.mean()   # fraction of translational jumpers
        f_both  = (m_rot * m_trans).mean()

        P_overlap[li] = f_both
        P_rand[li]    = f_rot * f_trans
        if P_rand[li] > 1e-15:
            R[li] = f_both / (f_rot * f_trans)

        # ----- Pearson correlation -----
        if np.std(dr_flat) > 1e-12 and np.std(th_flat) > 1e-12:
            pr, pp = pearsonr(dr_flat, th_flat)
            pearson_r[li] = pr
            pearson_p[li] = pp

    return dict(
        P_overlap=P_overlap,
        P_rand=P_rand,
        R=R,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
    )


def compute_conditional_averages(
    dr_flat: np.ndarray,
    theta_flat: np.ndarray,
    n_bins: int = 20,
) -> tuple:
    """
    Compute conditional means ⟨θ | Δr⟩ and ⟨Δr | θ⟩.

    Returns
    -------
    dr_bins    : ndarray (n_bins,)   bin centres in Å
    theta_bins : ndarray (n_bins,)   bin centres in radians
    mean_theta_given_dr : ndarray (n_bins,)   ⟨θ | Δr⟩
    mean_dr_given_theta : ndarray (n_bins,)   ⟨Δr | θ⟩
    """
    # ⟨θ | Δr⟩
    r_edges = np.linspace(0, dr_flat.max() * 1.01, n_bins + 1)
    dr_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    mean_theta_given_dr = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (dr_flat >= r_edges[b]) & (dr_flat < r_edges[b + 1])
        if mask.sum() > 5:
            mean_theta_given_dr[b] = theta_flat[mask].mean()

    # ⟨Δr | θ⟩
    t_edges = np.linspace(0, np.pi, n_bins + 1)
    theta_centres = 0.5 * (t_edges[:-1] + t_edges[1:])
    mean_dr_given_theta = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (theta_flat >= t_edges[b]) & (theta_flat < t_edges[b + 1])
        if mask.sum() > 5:
            mean_dr_given_theta[b] = dr_flat[mask].mean()

    return dr_centres, theta_centres, mean_theta_given_dr, mean_dr_given_theta


# ============================================================
#  SECTION 9 – Time-scale identification
# ============================================================

def identify_timescales(
    lag_times_ps: np.ndarray,
    chi4_or_R: np.ndarray,
    dr_msd: np.ndarray,
    dt_dump: float = P.dt_dump,
) -> dict:
    """
    Identify the three dynamical regimes and the time of peak coupling.

    Regimes identified
    ------------------
    * Ballistic / librational:  t ≲ 0.1 ps (sub-cage)
    * Cage / β-relaxation:      0.1–10 ps (plateau of MSD)
    * Diffusive / α-relaxation: t ≫ 10 ps (linear MSD growth)

    The coupling is expected to be strongest at intermediate times
    (cage-breaking events).

    Returns
    -------
    dict with keys: t_short, t_inter, t_long, t_peak_coupling (all in ps)
    """
    good = np.isfinite(chi4_or_R)
    if good.sum() < 3:
        return dict(t_short=lag_times_ps[0], t_inter=lag_times_ps[len(lag_times_ps)//2],
                    t_long=lag_times_ps[-1], t_peak=np.nan)

    t_peak_idx = np.nanargmax(chi4_or_R[good])
    t_peak = lag_times_ps[good][t_peak_idx]

    # Simple regime boundaries in ps
    t_short = lag_times_ps[lag_times_ps < 0.5][-1] if (lag_times_ps < 0.5).any() else lag_times_ps[0]
    t_long  = lag_times_ps[lag_times_ps > 20.0][0]  if (lag_times_ps > 20.0).any() else lag_times_ps[-1]
    t_inter = t_peak

    print(f"\n  Identified timescales:")
    print(f"    Short  (ballistic): t < {t_short:.2f} ps")
    print(f"    Inter  (coupling peak): t* = {t_inter:.2f} ps")
    print(f"    Long   (diffusive): t > {t_long:.2f} ps")

    return dict(t_short=t_short, t_inter=t_inter, t_long=t_long, t_peak=t_peak)


# ============================================================
#  SECTION 10 – Shear-box utilities
# ============================================================

def get_box_tilt(universe: "mda.Universe") -> np.ndarray:
    """
    Extract the xy box tilt factor (relevant for SLLOD shear) at each frame.

    For a LAMMPS triclinic box after deformation, the tilt xy accumulates
    as xy = γ̇ · Lx · t.  We read it from the trajectory's dimensions array.

    Returns
    -------
    tilt_xy : ndarray (n_frames,)   xy tilt factor in Å
    """
    n_frames = len(universe.trajectory)
    tilt_xy = np.zeros(n_frames)
    for fi, ts in enumerate(universe.trajectory):
        dims = ts.dimensions   # [lx, ly, lz, alpha, beta, gamma]
        # MDAnalysis stores triclinic info; for simple shear xy tilt:
        if hasattr(ts, "_unitcell"):
            # Low-level: _unitcell[1] = xy in LAMMPS convention
            try:
                tilt_xy[fi] = ts._unitcell[1]
            except (IndexError, TypeError):
                pass
    return tilt_xy


# ============================================================
#  SECTION 11 – MDAnalysis trajectory loading
# ============================================================

def load_trajectory_mda(
    topology_file: str,
    trajectory_file: str,
    params: SimParams = P,
) -> tuple:
    """
    Load trajectory using MDAnalysis and extract:
        - Oxygen positions (unwrapped)
        - H1, H2 positions (unwrapped)
        - Dipole vectors at every frame

    Atom selection
    --------------
    We select oxygens and hydrogens separately.  For TIP4P/Ice the
    standard naming convention is OW (oxygen) and HW1, HW2 (hydrogens).
    For LAMMPS dump files without atom names you may need to set atom
    types in the topology or use atom-index-based selection.

    Returns
    -------
    pos_O   : (n_frames, N_mol, 3)   oxygen positions
    pos_H1  : (n_frames, N_mol, 3)
    pos_H2  : (n_frames, N_mol, 3)
    dipoles : (n_frames, N_mol, 3)   unit dipole vectors
    """
    print(f"\nLoading trajectory: {trajectory_file}")
    u = mda.Universe(topology_file, trajectory_file, format="LAMMPSDUMP")
    n_frames = len(u.trajectory)

    # ---- Atom selection ----
    # Attempt standard TIP4P/Ice naming; fall back to type-based
    try:
        ox_sel  = u.select_atoms("name OW")
        h1_sel  = u.select_atoms("name HW1")
        h2_sel  = u.select_atoms("name HW2")
        if len(ox_sel) == 0:
            raise ValueError("No OW atoms found.")
    except Exception:
        # Fallback: assume O H H order in LAMMPS dump (type 1 = O, type 2 = H)
        ox_sel = u.select_atoms("type 1")
        h1_sel = u.select_atoms("type 2")[0::2]   # every other H
        h2_sel = u.select_atoms("type 2")[1::2]

    N_mol = len(ox_sel)
    print(f"  Molecules: {N_mol},  Frames: {n_frames}")

    # ---- Allocate arrays ----
    pos_O  = np.zeros((n_frames, N_mol, 3), dtype=np.float32)
    pos_H1 = np.zeros((n_frames, N_mol, 3), dtype=np.float32)
    pos_H2 = np.zeros((n_frames, N_mol, 3), dtype=np.float32)

    for fi, ts in enumerate(u.trajectory):
        pos_O[fi]  = ox_sel.positions.copy()
        pos_H1[fi] = h1_sel.positions.copy()
        pos_H2[fi] = h2_sel.positions.copy()

    pos_O  = pos_O.astype(float)
    pos_H1 = pos_H1.astype(float)
    pos_H2 = pos_H2.astype(float)

    # ---- Dipoles at every frame ----
    print("  Computing dipoles...")
    dipoles = np.zeros_like(pos_O)
    for fi in range(n_frames):
        dipoles[fi] = compute_dipoles(pos_O[fi], pos_H1[fi], pos_H2[fi])

    return pos_O, pos_H1, pos_H2, dipoles


def choose_lag_steps(n_frames: int, dt_dump: float, n_t0: int) -> tuple:
    """
    Choose lag steps (geometrically spaced) and t0 indices.

    Returns
    -------
    lag_steps    : ndarray (n_lags,) int
    lag_times_fs : ndarray (n_lags,)
    t0_indices   : list of int
    """
    max_lag = n_frames // 2
    lag_steps = np.unique(
        np.round(np.geomspace(1, max_lag, 30)).astype(int)
    )
    lag_steps = lag_steps[lag_steps < n_frames]

    # t0 indices: steady-state portion of trajectory
    frame_start = int(P.t0_start_frac * n_frames)
    frame_end   = n_frames - lag_steps.max() - 1
    frame_end   = max(frame_end, frame_start + 1)
    t0_indices  = sorted(set(
        np.linspace(frame_start, frame_end, n_t0, dtype=int).tolist()
    ))
    lag_times_fs = lag_steps * dt_dump
    return lag_steps, lag_times_fs, t0_indices


# ============================================================
#  SECTION 12 – Visualisation
# ============================================================

def make_all_plots(
    lag_times_ps: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    theta_msad: np.ndarray,
    theta_bins: np.ndarray,
    P_theta: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    P_joint: list,
    jump_stats: dict,
    dr_all: list,
    theta_all: list,
    tstar_idx: int,
    output_dir: Path,
    params: SimParams = P,
):
    """Generate and save all diagnostic figures."""
    dpi = params.fig_dpi
    out = output_dir

    # ------------------------------------------------------------------
    # Figure 1 — Orientational ACFs  C1(t) and C2(t) on log-time axis
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    t_ps = lag_times_ps
    valid = t_ps > 0
    ax.semilogx(t_ps[valid], C1[valid], "o-", lw=2, ms=4, label="$C_1(t)$")
    ax.semilogx(t_ps[valid], C2[valid], "s--", lw=2, ms=4, label="$C_2(t)$")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("$t$  (ps)")
    ax.set_ylabel("$C_{1,2}(t)$")
    ax.set_title("Dipole reorientation ACF")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)

    ax = axes[1]
    ax.loglog(t_ps[valid & (theta_msad > 0)], theta_msad[valid & (theta_msad > 0)],
              "^-", lw=2, ms=5, color="darkorange")
    ax.set_xlabel("$t$  (ps)")
    ax.set_ylabel(r"$\langle \theta^2 \rangle$  (rad²)")
    ax.set_title("Mean-squared angular displacement")
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out / "acf_and_msad.png", dpi=dpi)
    plt.close(fig)
    print("  Saved acf_and_msad.png")

    # ------------------------------------------------------------------
    # Figure 2 — P(θ, t) at three representative lag times
    # ------------------------------------------------------------------
    n_lags = len(lag_times_ps)
    idx_short = max(0, int(0.05 * n_lags))
    idx_inter = tstar_idx
    idx_long  = min(n_lags - 1, int(0.90 * n_lags))
    plot_indices = sorted({idx_short, idx_inter, idx_long})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    theta_deg = np.rad2deg(theta_bins)
    colors_p = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_indices)))
    for ci, li in enumerate(plot_indices):
        if P_theta[li].sum() > 0:
            lab = f"$t$ = {lag_times_ps[li]:.2f} ps"
            ax.plot(theta_deg, P_theta[li], lw=2, color=colors_p[ci], label=lab)
    ax.set_xlabel(r"$\theta$  (degrees)")
    ax.set_ylabel(r"$P(\theta, t)$  (rad$^{-1}$)")
    ax.set_title("Angular displacement distribution")
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "P_theta_t.png", dpi=dpi)
    plt.close(fig)
    print("  Saved P_theta_t.png")

    # ------------------------------------------------------------------
    # Figure 3 — Joint distribution P(Δr, θ) at t*   (heatmap)
    # ------------------------------------------------------------------
    Pj = P_joint[tstar_idx]
    if Pj is not None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        theta_deg_e = np.rad2deg(theta_edges)
        r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
        theta_deg_c = np.rad2deg(0.5 * (theta_edges[:-1] + theta_edges[1:]))

        # Smooth slightly for display
        Pj_smooth = gaussian_filter(Pj, sigma=1.0)
        im = ax.pcolormesh(
            theta_deg_e, r_edges, Pj_smooth,
            cmap="hot_r",
            norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=Pj_smooth.max()),
        )
        # Mark thresholds
        theta_c_deg_val = np.rad2deg(np.deg2rad(params.theta_c_deg))
        ax.axhline(params.r_c_angstrom, color="cyan", lw=1.5, ls="--",
                   label=f"$r_c$ = {params.r_c_angstrom:.1f} Å")
        ax.axvline(theta_c_deg_val, color="lime", lw=1.5, ls="--",
                   label=f"$\\theta_c$ = {params.theta_c_deg:.0f}°")
        plt.colorbar(im, ax=ax, label=r"$P(\Delta r,\,\theta;\,t^*)$")
        ax.set_xlabel(r"$\theta$  (degrees)")
        ax.set_ylabel(r"$\Delta r$  (Å)")
        ax.set_title(rf"Joint distribution at $t^*$ = {lag_times_ps[tstar_idx]:.2f} ps")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "joint_distribution.png", dpi=dpi)
        plt.close(fig)
        print("  Saved joint_distribution.png")

    # ------------------------------------------------------------------
    # Figure 4 — Coupling ratio R(t) and Pearson correlation
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    R_vals = jump_stats["R"]
    pr_vals = jump_stats["pearson_r"]

    ax = axes[0]
    good = np.isfinite(R_vals) & (lag_times_ps > 0)
    ax.semilogx(lag_times_ps[good], R_vals[good], "o-", lw=2, ms=5, color="crimson")
    ax.axhline(1.0, color="grey", lw=1.2, ls="--", label="$R=1$ (independent)")
    tstar_ps = lag_times_ps[tstar_idx]
    ax.axvline(tstar_ps, color="blue", lw=1.2, ls=":", label=f"$t^*$={tstar_ps:.2f} ps")
    ax.set_xlabel("$t$  (ps)")
    ax.set_ylabel("$R(t) = P_\\mathrm{overlap} / P_\\mathrm{rand}$")
    ax.set_title("Rotational–translational coupling")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)

    ax = axes[1]
    good2 = np.isfinite(pr_vals) & (lag_times_ps > 0)
    ax.semilogx(lag_times_ps[good2], pr_vals[good2], "s-", lw=2, ms=5, color="darkorange")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("$t$  (ps)")
    ax.set_ylabel("Pearson $r(\\theta, \\Delta r)$")
    ax.set_title("Linear correlation: rotation vs translation")
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out / "coupling_R_and_pearson.png", dpi=dpi)
    plt.close(fig)
    print("  Saved coupling_R_and_pearson.png")

    # ------------------------------------------------------------------
    # Figure 5 — Conditional averages at t*
    # ------------------------------------------------------------------
    if dr_all[tstar_idx] is not None and theta_all[tstar_idx] is not None:
        dr_flat = dr_all[tstar_idx].ravel()
        th_flat = theta_all[tstar_idx].ravel()
        dr_c, th_c, mean_th_dr, mean_dr_th = compute_conditional_averages(
            dr_flat, th_flat, n_bins=25
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ok = np.isfinite(mean_th_dr)
        ax.plot(dr_c[ok], np.rad2deg(mean_th_dr[ok]), "o-", lw=2, ms=5, color="purple")
        ax.axhline(params.theta_c_deg, color="grey", ls="--",
                   label=f"$\\theta_c$ = {params.theta_c_deg:.0f}°")
        ax.set_xlabel(r"$\Delta r$  (Å)")
        ax.set_ylabel(r"$\langle \theta \mid \Delta r \rangle$  (degrees)")
        ax.set_title(rf"Conditional mean angle at $t^*$")
        ax.legend(); ax.grid(True, alpha=0.25)

        ax = axes[1]
        ok2 = np.isfinite(mean_dr_th)
        ax.plot(np.rad2deg(th_c[ok2]), mean_dr_th[ok2], "s-", lw=2, ms=5, color="teal")
        ax.axvline(params.theta_c_deg, color="grey", ls="--",
                   label=f"$\\theta_c$ = {params.theta_c_deg:.0f}°")
        ax.set_xlabel(r"$\theta$  (degrees)")
        ax.set_ylabel(r"$\langle \Delta r \mid \theta \rangle$  (Å)")
        ax.set_title(rf"Conditional mean displacement at $t^*$")
        ax.legend(); ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(out / "conditional_averages.png", dpi=dpi)
        plt.close(fig)
        print("  Saved conditional_averages.png")


# ============================================================
#  SECTION 13 – Main analysis driver
# ============================================================

def run_analysis(
    topology_file: str,
    trajectory_file: str,
    output_dir: str = "dipole_output",
    params: SimParams = P,
) -> dict:
    """
    Full analysis pipeline for dipole orientational dynamics.

    Parameters
    ----------
    topology_file   : MDAnalysis-readable topology (PDB, GRO, PSF …)
    trajectory_file : trajectory file (DCD, XTC, LAMMPSDUMP …)
    output_dir      : output directory for figures and data
    params          : SimParams object

    Returns
    -------
    results : dict (see end of function)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # STEP 1 – Load trajectory and compute dipoles
    # ================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading trajectory and computing dipoles")
    print("="*60)
    pos_O, pos_H1, pos_H2, dipoles = load_trajectory_mda(
        topology_file, trajectory_file, params
    )
    n_frames, N_mol, _ = pos_O.shape

    # ================================================================
    # STEP 2 – Lag steps and t0 indices
    # ================================================================
    lag_steps, lag_times_fs, t0_indices = choose_lag_steps(
        n_frames, params.dt_dump, params.n_t0
    )
    lag_times_ps = lag_times_fs / 1000.0   # fs → ps
    n_lags = len(lag_steps)
    print(f"  Lag times: {lag_times_ps[0]:.3f} — {lag_times_ps[-1]:.3f} ps  ({n_lags} points)")
    print(f"  Time origins: {len(t0_indices)}")

    # ================================================================
    # STEP 3 – Orientational correlation functions
    # ================================================================
    print("\n" + "="*60)
    print("STEP 3: Orientational ACFs  C1(t), C2(t)")
    print("="*60)
    C1, C2 = compute_orientation_correlation(dipoles, lag_steps, t0_indices)

    # ================================================================
    # STEP 4 – Angular displacement
    # ================================================================
    print("\n" + "="*60)
    print("STEP 4: Angular displacement  ⟨θ²(t)⟩  and  P(θ,t)")
    print("="*60)
    theta_msad, theta_all = compute_angular_displacement(dipoles, lag_steps, t0_indices)
    theta_bins, P_theta = compute_theta_distribution(theta_all, params.n_theta_bins)

    # ================================================================
    # STEP 5 – Translational displacement
    # ================================================================
    print("\n" + "="*60)
    print("STEP 5: Translational displacement  ⟨|Δr|²(t)⟩")
    print("="*60)
    dr_all, dr_msd = compute_displacements(
        pos_O, lag_steps, t0_indices,
        shear_rate=params.shear_rate,
        dt_dump=params.dt_dump,
        remove_affine=params.remove_affine,
    )
    print(f"  Affine correction: {params.remove_affine}")

    # ================================================================
    # STEP 6 – Joint distribution
    # ================================================================
    print("\n" + "="*60)
    print("STEP 6: Joint distribution  P(Δr, θ; t)")
    print("="*60)
    r_edges, theta_edges, P_joint = compute_joint_distribution(
        dr_all, theta_all, params.n_r_bins, params.n_theta_bins
    )

    # ================================================================
    # STEP 7 – Jump thresholds
    # ================================================================
    print("\n" + "="*60)
    print("STEP 7: Jump thresholds")
    print("="*60)
    # Peak of ⟨θ²⟩ growth rate as a proxy for cage-breaking time
    diffs = np.diff(np.log(theta_msad + 1e-30))
    start_idx = len(diffs) // 4
    end_idx = len(diffs) - start_idx
    max_idx = np.argmax(diffs[start_idx:end_idx]) + start_idx
    tstar_idx = max(1, int(max_idx)) if len(diffs) > 0 else n_lags // 2
    tstar_idx = min(tstar_idx, n_lags - 1)
    theta_c, r_c = define_thresholds(dr_all, theta_all, tstar_idx, params)

    # ================================================================
    # STEP 8 – Coupling statistics
    # ================================================================
    print("\n" + "="*60)
    print("STEP 8: Rotation–translation coupling")
    print("="*60)
    jump_stats = compute_jump_correlation(dr_all, theta_all, theta_c, r_c)

    # Update tstar_idx to be the peak of R(t) if available
    R_vals = jump_stats["R"]
    # if np.isfinite(R_vals).any():
    #     tstar_idx = int(np.nanargmax(R_vals))
    #     print(f"  Peak R(t) at t* = {lag_times_ps[tstar_idx]:.3f} ps, "
    #           f"R = {R_vals[tstar_idx]:.3f}")

    # ================================================================
    # STEP 9 – Timescale identification
    # ================================================================
    print("\n" + "="*60)
    print("STEP 9: Timescale identification")
    print("="*60)
    timescales = identify_timescales(lag_times_ps, R_vals, dr_msd, params.dt_dump)

    # ================================================================
    # STEP 10 – Print summary table
    # ================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  {'Lag (ps)':>10}  {'C1':>8}  {'C2':>8}  "
          f"{'⟨θ²⟩(°²)':>12}  {'R(t)':>8}  {'Pearson r':>10}")
    every = max(1, n_lags // 10)
    for li in range(0, n_lags, every):
        th2_deg2 = np.rad2deg(np.sqrt(theta_msad[li])) ** 2 if theta_msad[li] > 0 else np.nan
        print(f"  {lag_times_ps[li]:>10.3f}  {C1[li]:>8.4f}  {C2[li]:>8.4f}  "
              f"{th2_deg2:>12.3f}  {R_vals[li]:>8.3f}  "
              f"{jump_stats['pearson_r'][li]:>10.4f}")

    # ================================================================
    # STEP 11 – Save numerical results
    # ================================================================
    print("\n" + "="*60)
    print("STEP 11: Saving data")
    print("="*60)
    np.savez(
        out / "dipole_results.npz",
        lag_times_ps=lag_times_ps,
        C1=C1, C2=C2,
        theta_msad=theta_msad,
        theta_bins=theta_bins,
        P_theta=P_theta,
        dr_msd=dr_msd,
        r_edges=r_edges,
        theta_edges=theta_edges,
        R=R_vals,
        P_overlap=jump_stats["P_overlap"],
        P_rand=jump_stats["P_rand"],
        pearson_r=jump_stats["pearson_r"],
        tstar_ps=np.array([lag_times_ps[tstar_idx]]),
    )
    print("  Saved dipole_results.npz")

    # ================================================================
    # STEP 12 – Plots
    # ================================================================
    print("\n" + "="*60)
    print("STEP 12: Generating figures")
    print("="*60)
    make_all_plots(
        lag_times_ps=lag_times_ps,
        C1=C1, C2=C2,
        theta_msad=theta_msad,
        theta_bins=theta_bins,
        P_theta=P_theta,
        r_edges=r_edges,
        theta_edges=theta_edges,
        P_joint=P_joint,
        jump_stats=jump_stats,
        dr_all=dr_all,
        theta_all=theta_all,
        tstar_idx=tstar_idx,
        output_dir=out,
        params=params,
    )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Output directory: {out.resolve()}")
    print("="*60)

    return dict(
        lag_times_ps=lag_times_ps,
        C1=C1, C2=C2,
        theta_msad=theta_msad,
        theta_bins=theta_bins,
        P_theta=P_theta,
        dr_msd=dr_msd,
        r_edges=r_edges,
        theta_edges=theta_edges,
        P_joint=P_joint,
        jump_stats=jump_stats,
        timescales=timescales,
        tstar_idx=tstar_idx,
        tstar_ps=lag_times_ps[tstar_idx],
        theta_c=theta_c,
        r_c=r_c,
    )


# ============================================================
#  SECTION 14 – Synthetic trajectory generator (testing)
# ============================================================

def generate_synthetic_water(
    N_mol: int = 800,
    n_frames: int = 120,
    L: float = 40.0,
    dt: float = 25.0,
    gamma_dot: float = 5e-6,
    seed: int = 42,
    outfile: str = "synthetic_water.npz",
) -> str:
    """
    Generate a synthetic TIP4P/Ice-like trajectory for unit testing.

    Model
    -----
    * Oxygens perform a heterogeneous random walk:
      fast (mobile) and slow (caged) subpopulations.
    * H1 and H2 are placed at fixed bond length r_OH = 0.96 Å with
      bond angle HOH = 104.5° from a rotating dipole.
    * Dipole reorients via Brownian rotation on the unit sphere with
      two populations: slow (D_rot = 0.001 rad²/fs) and fast (0.02).
    * Affine shear is added: Δx += γ̇ · y(t0) · dt per step.
    """
    rng = np.random.default_rng(seed)
    r_OH = 0.96    # Å
    half_angle = np.deg2rad(104.5 / 2.0)   # half of HOH bond angle

    # ---- Initial positions and dipoles ----
    pos_O = rng.uniform(0.5, L - 0.5, (N_mol, 3))
    # Random unit dipoles
    u = rng.normal(0, 1, (N_mol, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    # Population labels: 0 = slow (caged), 1 = fast (mobile)
    mobile = rng.random(N_mol) < 0.2   # 20 % mobile
    D_trans_arr = np.where(mobile, 0.05, 0.002)    # Å²/fs
    D_rot_arr   = np.where(mobile, 0.02, 0.0005)   # rad²/fs

    # ---- Storage ----
    traj_O  = np.zeros((n_frames, N_mol, 3), dtype=np.float32)
    traj_H1 = np.zeros_like(traj_O)
    traj_H2 = np.zeros_like(traj_O)
    traj_O[0] = pos_O

    def _place_hydrogens(pos_ox, dipole):
        """Place H1 and H2 given oxygen position and dipole unit vector."""
        # Build local frame: dipole direction and two perpendicular axes
        d = dipole  # (N,3)
        arb = np.where(np.abs(d[:, 0:1]) < 0.9,
                       np.array([[1,0,0]]), np.array([[0,1,0]]))
        e1 = np.cross(d, arb); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(d, e1); e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
        # H positions at ± half_angle from dipole in the d-e1 plane
        h1_dir = np.cos(half_angle) * d + np.sin(half_angle) * e1
        h2_dir = np.cos(half_angle) * d - np.sin(half_angle) * e1
        H1 = pos_ox + r_OH * h1_dir
        H2 = pos_ox + r_OH * h2_dir
        return H1, H2

    H1_0, H2_0 = _place_hydrogens(pos_O, u)
    traj_H1[0] = H1_0; traj_H2[0] = H2_0

    pos_O_unwrap = pos_O.copy()

    for fi in range(1, n_frames):
        # Translational step
        sigma_t = np.sqrt(2 * D_trans_arr * dt)[:, None]
        dr = rng.normal(0, sigma_t, (N_mol, 3))
        dr[:, 0] += gamma_dot * pos_O_unwrap[:, 1] * dt   # affine shear
        pos_O_unwrap = pos_O_unwrap + dr

        # Rotational step: Rodrigues rotation by small random angle
        sigma_r = np.sqrt(2 * D_rot_arr * dt)
        angle = rng.normal(0, sigma_r, N_mol)
        axis = rng.normal(0, 1, (N_mol, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-15
        # Rodrigues: u' = u cos(a) + (axis × u) sin(a) + axis (axis·u)(1-cos(a))
        c = np.cos(angle)[:, None]; s = np.sin(angle)[:, None]
        axu = np.cross(axis, u)
        au = np.einsum("ij,ij->i", axis, u)[:, None]
        u = u * c + axu * s + axis * au * (1 - c)
        norms = np.linalg.norm(u, axis=1, keepdims=True) + 1e-15
        u = u / norms

        H1, H2 = _place_hydrogens(pos_O_unwrap, u)
        traj_O[fi]  = pos_O_unwrap.astype(np.float32)
        traj_H1[fi] = H1.astype(np.float32)
        traj_H2[fi] = H2.astype(np.float32)

    np.savez(outfile, pos_O=traj_O, pos_H1=traj_H1, pos_H2=traj_H2,
             L=L, dt=dt, gamma_dot=gamma_dot, N_mol=N_mol)
    print(f"Saved synthetic trajectory: {outfile}  (N={N_mol}, F={n_frames})")
    return outfile


class SyntheticAnalysis:
    """
    Run the full analysis on a synthetic trajectory without MDAnalysis.
    Useful for testing and development.
    """
    def __init__(self, npz_file: str):
        data = np.load(npz_file)
        self.pos_O  = data["pos_O"].astype(float)   # (n_frames, N_mol, 3)
        self.pos_H1 = data["pos_H1"].astype(float)
        self.pos_H2 = data["pos_H2"].astype(float)
        self.L = float(data["L"])
        self.dt = float(data["dt"])
        self.gamma_dot = float(data["gamma_dot"])
        self.N_mol = int(data["N_mol"])
        self.n_frames = self.pos_O.shape[0]
        print(f"SyntheticAnalysis: {self.n_frames} frames, {self.N_mol} molecules")

    def run(self, output_dir: str = "dipole_output_test") -> dict:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        p = SimParams()
        p.L = self.L; p.dt_dump = self.dt; p.shear_rate = self.gamma_dot

        # Dipoles
        print("Computing dipoles...")
        n_frames = self.n_frames; N_mol = self.N_mol
        dipoles = np.zeros((n_frames, N_mol, 3))
        for fi in range(n_frames):
            dipoles[fi] = compute_dipoles(self.pos_O[fi], self.pos_H1[fi], self.pos_H2[fi])

        # Lag steps / t0
        lag_steps, lag_times_fs, t0_indices = choose_lag_steps(
            n_frames, p.dt_dump, p.n_t0
        )
        lag_times_ps = lag_times_fs / 1000.0
        n_lags = len(lag_steps)

        # ACFs
        print("Computing ACFs...")
        C1, C2 = compute_orientation_correlation(dipoles, lag_steps, t0_indices)

        # Angular displacements
        print("Computing angular displacements...")
        theta_msad, theta_all = compute_angular_displacement(dipoles, lag_steps, t0_indices)
        theta_bins, P_theta = compute_theta_distribution(theta_all, p.n_theta_bins)

        # Translational displacements
        print("Computing translational displacements...")
        dr_all, dr_msd = compute_displacements(
            self.pos_O, lag_steps, t0_indices,
            shear_rate=p.shear_rate, dt_dump=p.dt_dump,
            remove_affine=p.remove_affine,
        )

        # Joint distribution
        print("Computing joint distribution...")
        r_edges, theta_edges, P_joint = compute_joint_distribution(
            dr_all, theta_all, p.n_r_bins, p.n_theta_bins
        )

        # Thresholds and coupling
        diffs = np.diff(np.log(theta_msad + 1e-30))
        tstar_idx = max(1, int(np.argmax(diffs))) if len(diffs) > 0 else n_lags // 2
        tstar_idx = min(tstar_idx, n_lags - 1)
        theta_c, r_c = define_thresholds(dr_all, theta_all, tstar_idx, p)

        print("Computing coupling statistics...")
        jump_stats = compute_jump_correlation(dr_all, theta_all, theta_c, r_c)

        R_vals = jump_stats["R"]
        if np.isfinite(R_vals).any():
            tstar_idx = int(np.nanargmax(R_vals))

        timescales = identify_timescales(lag_times_ps, R_vals, dr_msd, p.dt_dump)

        make_all_plots(
            lag_times_ps, C1, C2, theta_msad, theta_bins, P_theta,
            r_edges, theta_edges, P_joint, jump_stats, dr_all, theta_all,
            tstar_idx, out, p,
        )

        # Validation assertions (lag_steps[0] = 1 frame = dt_dump fs, not lag=0)
        assert C1[0] > 0.3, f"C1 at first lag should be positive, got {C1[0]:.4f}"
        assert C2[0] > -0.5, f"C2 at first lag should not be strongly negative, got {C2[0]:.4f}"
        assert theta_msad[0] < theta_msad[-1], "MSAD should increase with lag time"
        assert np.isfinite(R_vals).any(), "R(t) has no finite values"

        print(f"\n  t* = {lag_times_ps[tstar_idx]:.3f} ps,  R(t*) = {R_vals[tstar_idx]:.3f}")
        print(f"  C1(t=0) = {C1[0]:.4f},  C2(t=0) = {C2[0]:.4f}")
        print(f"  Output: {out.resolve()}")

        return dict(C1=C1, C2=C2, theta_msad=theta_msad, dr_msd=dr_msd,
                    jump_stats=jump_stats, lag_times_ps=lag_times_ps,
                    tstar_ps=lag_times_ps[tstar_idx], timescales=timescales)


# ============================================================
#  SECTION 15 – Entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) == 3:
        topology_file   = sys.argv[1]
        trajectory_file = sys.argv[2]
        results = run_analysis(topology_file, trajectory_file)

    elif len(sys.argv) == 2 and sys.argv[1] == "--test":
        print("\n*** Self-test with synthetic TIP4P/Ice-like trajectory ***\n")
        npz = generate_synthetic_water(
            N_mol=600, n_frames=100, L=35.0, dt=25.0, gamma_dot=5e-6, seed=7
        )
        sa = SyntheticAnalysis(npz)
        results = sa.run(output_dir="dipole_output_test")
        print("\nAll validation checks PASSED.")

    else:
        print(__doc__)
        print("\nUsage:")
        print("  python dipole_dynamics.py  topology.pdb  trajectory.lammpsdump")
        print("  python dipole_dynamics.py  --test")
