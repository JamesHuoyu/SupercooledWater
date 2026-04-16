"""
Four-Point Structure Factor S4(k,t) Analysis Pipeline
======================================================
System : Supercooled water at 225 K under xy-plane SLLOD shear
Purpose: Compute S4(k,t), χ4(t), dynamic correlation length ξ4,
         and diagnose shear banding from an unwrapped MD trajectory.

Physical background
-------------------
Dynamical heterogeneity in supercooled liquids is quantified by the
four-point susceptibility χ4(t) and its k-resolved generalisation S4(k,t).

    S4(k, t) = (1/N) ⟨ |ρ_w(k,t0)|^2 ⟩_{t0}

where the "mobility-weighted density" is

    ρ_w(k, t0) = Σ_i  w_i(t0,t)  exp(i k · r_i(t0))

and the overlap mobility indicator is

    w_i(t0,t) = 1  if |Δr_i^{NA}(t0,t)| < a   (a = 1.0 Å)
              = 0  otherwise

Key physics notes
-----------------
* Positions used in the phase factor are those at the *reference* time t0
  (not t0+t).  This ensures we are probing spatial correlations of the
  *initial* positions of mobile particles, which is the standard definition.
* Non-affine (NA) displacements are used so that the trivial affine shear
  flow does not mask genuine cage-breaking dynamics.
* Because SLLOD shear breaks isotropy (the velocity gradient is along y,
  the flow is along x), S4 must NOT be angularly averaged without first
  checking anisotropy.  We therefore compute S4 separately along z and y,
  and in the full yz plane (kx=0).

Fitting ξ4
----------
At t = t* (peak of χ4) we fit the small-k behaviour:

    1/S4(k) = A + B k^2

=> ξ4 = sqrt(B/A)

This is the Ornstein–Zernike form appropriate for the envelope of
correlated mobile-particle clusters.

Units
-----
All distances: Ångström (Å)
All times    : femtoseconds (fs)
All k vectors: Å^{-1}

Dependencies
------------
numpy, scipy, matplotlib, MDAnalysis (>=2.0)

Usage
-----
    python s4_analysis.py  trajectory.dcd  topology.pdb

Or import and call run_analysis() programmatically.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.base import AnalysisBase
except ImportError:
    raise ImportError("MDAnalysis is required.  pip install MDAnalysis")


# ============================================================
#  SECTION 0 – Simulation / analysis parameters (edit here)
# ============================================================

class SimParams:
    """Central repository of all simulation and analysis parameters."""

    # ---- Simulation geometry ----
    L: float = 50.0          # Box length (Å), assumed cubic
    T: float = 225.0          # Temperature (K)
    dt_dump: float = 25.0     # Dump interval (fs)
    total_time: float = 1e5   # Total simulation time (fs) = 100 ps
    shear_rate: float = 5e-6  # LAMMPS real units (1/fs effectively dimensionless here)

    # ---- Overlap threshold ----
    a_threshold: float = 1.0  # Ångström; defines mobile vs immobile

    # ---- Time origins ----
    n_t0: int = 500            # Number of time origins for ensemble average
    t0_start_frac: float = 0.1  # Skip first 10 % as transient / equilibration

    # ---- k-space ----
    n_kmax: int = 6           # Maximum integer for each k-component (kn ≤ n_kmax)
    n_k_fit: int = 8          # Number of smallest |k| modes used in OZ fit

    # ---- Shear banding ----
    n_ybins: int = 20         # Number of y-bins for vx(y) profile

    # ---- Plotting ----
    fig_dpi: int = 300


P = SimParams()   # global parameter object; replace fields as needed


# ============================================================
#  SECTION 1 – Utility functions
# ============================================================

def build_kvectors(L: float, n_kmax: int):
    """
    Construct all reciprocal-lattice vectors compatible with PBC.

    k_n = (2π/L) * (nx, ny, nz),   |nx|,|ny|,|nz| ≤ n_kmax

    Returns
    -------
    kvecs : ndarray (M, 3)  full k-vector set
    kmags : ndarray (M,)    |k|
    k0    : float           minimum non-zero |k| = 2π/L
    """
    dk = 2.0 * np.pi / L
    ns = np.arange(-n_kmax, n_kmax + 1)
    nx, ny, nz = np.meshgrid(ns, ns, ns, indexing="ij")
    nx = nx.ravel(); ny = ny.ravel(); nz = nz.ravel()
    kvecs = dk * np.column_stack([nx, ny, nz])  # (M,3)
    kmags = np.linalg.norm(kvecs, axis=1)
    return kvecs, kmags, dk


def build_kvectors_plane(L: float, n_kmax: int, fixed_axis: int = 0):
    """
    k-vectors in a 2-D plane with one component fixed at zero.

    Parameters
    ----------
    fixed_axis : 0=kx, 1=ky, 2=kz  → the axis forced to zero

    Returns
    -------
    kvecs : ndarray (M, 3)
    kmags : ndarray (M,)
    """
    dk = 2.0 * np.pi / L
    ns = np.arange(-n_kmax, n_kmax + 1)
    n1, n2 = np.meshgrid(ns, ns, indexing="ij")
    n1 = n1.ravel(); n2 = n2.ravel()
    axes = [0, 1, 2]
    free = [a for a in axes if a != fixed_axis]
    kvecs = np.zeros((len(n1), 3))
    kvecs[:, free[0]] = dk * n1
    kvecs[:, free[1]] = dk * n2
    kmags = np.linalg.norm(kvecs, axis=1)
    return kvecs, kmags


def build_kvectors_axis(L: float, n_kmax: int, axis: int):
    """
    1-D set of k-vectors along a single Cartesian axis.

    Parameters
    ----------
    axis : 0=x, 1=y, 2=z

    Returns
    -------
    kvecs : ndarray (2*n_kmax+1, 3)  includes k=0
    kmags : ndarray (2*n_kmax+1,)
    """
    dk = 2.0 * np.pi / L
    ns = np.arange(-n_kmax, n_kmax + 1)
    kvecs = np.zeros((len(ns), 3))
    kvecs[:, axis] = dk * ns
    kmags = np.abs(dk * ns)
    return kvecs, kmags


def compute_overlap_field(
    dr_na: np.ndarray,
    threshold: float = P.a_threshold,
) -> np.ndarray:
    """
    Compute the binary overlap field w_i for a single (t0, t) pair.

    Physical meaning
    ----------------
    w_i = 1 marks particles that have NOT moved beyond 'a' in the
    non-affine sense — i.e. they are "immobile" / still caged.
    Alternatively one can define mobility as w_i=1 for movers; both
    choices give the same χ4 because χ4 measures *variance* of the
    average, which is symmetric.  Here we follow the standard convention
    (cage-persistent particles = 1).

    Parameters
    ----------
    dr_na : ndarray (N, 3)  non-affine displacements Δr^NA_i(t0, t0+t)
    threshold : float       cage radius a (Å)

    Returns
    -------
    w : ndarray (N,)  binary float array
    """
    disp_sq = np.sum(dr_na ** 2, axis=1)          # |Δr^NA|^2  (N,)
    return (disp_sq < threshold ** 2).astype(float)


def mobility_density_fourier(
    pos_t0: np.ndarray,
    w: np.ndarray,
    kvecs: np.ndarray,
) -> np.ndarray:
    """
    Compute ρ_w(k) = Σ_i w_i exp(i k · r_i(t0)) for all k at once.

    Numerical implementation
    ------------------------
    Uses vectorised broadcasting: phases (N,M) = pos_t0 @ kvecs.T
    Memory note: for N~15000 and M~1000 this is a 15M complex array
    (~120 MB).  If memory is a concern reduce n_kmax or chunk over k.

    Parameters
    ----------
    pos_t0 : ndarray (N, 3)  positions at reference time t0  (Å)
    w      : ndarray (N,)    binary overlap weights
    kvecs  : ndarray (M, 3)  k-vectors  (Å^{-1})

    Returns
    -------
    rho_w : ndarray (M,)  complex  mobility-weighted Fourier modes
    """
    # phases[i, m] = k_m · r_i,  shape (N, M)
    phases = pos_t0 @ kvecs.T           # pure numpy dot, no Python loop
    exp_ikr = np.exp(1j * phases)       # complex exponentials
    # Weight each particle by w_i then sum over i
    rho_w = (w[:, np.newaxis] * exp_ikr).sum(axis=0)   # (M,)
    return rho_w


# ============================================================
#  SECTION 2 – Core S4(k,t) computation
# ============================================================
def compute_S4_single_t(
    positions: np.ndarray,     # (n_frames, N, 3)
    non_affine_disp: np.ndarray,  # (n_frames-1, N, 3) or similar
    kvecs: np.ndarray,         # (M, 3)
    t0_indices: list,
    lag_index: int,
    threshold: float = P.a_threshold,
) -> np.ndarray:
    """
    Compute connected S4(k, t) for a single lag time.

    S4(k, t) = (1/N) [ <|ρ_w(k)|^2> - |<ρ_w(k)>|^2 ]
    """

    N = positions.shape[1]
    M = len(kvecs)

    S4_raw = np.zeros(M, dtype=float)
    rho_mean = np.zeros(M, dtype=complex)
    count = 0

    for t0_idx in t0_indices:
        t1_idx = t0_idx + lag_index
        if t1_idx >= positions.shape[0]:
            continue

        # --- NA displacement ---
        dr_na = non_affine_disp[t0_idx, :, :]

        # --- mobility ---
        w = compute_overlap_field(dr_na, threshold)

        # --- positions ---
        r_t0 = positions[t0_idx, :, :]

        # --- Fourier ---
        rho_w = mobility_density_fourier(r_t0, w, kvecs)

        # --- accumulate ---
        S4_raw += np.abs(rho_w) ** 2
        rho_mean += rho_w
        count += 1

    if count == 0:
        raise RuntimeError(f"No valid t0 frames for lag_index={lag_index}.")

    S4_raw /= count
    rho_mean /= count

    S4 = (S4_raw - np.abs(rho_mean) ** 2) / N
    return S4


def compute_S4_time_series(
    positions: np.ndarray,
    na_disp_cumulative: np.ndarray,
    kvecs: np.ndarray,
    lag_indices: np.ndarray,
    t0_indices: list,
    threshold: float = P.a_threshold,
) -> np.ndarray:
    """
    Compute connected S4(k, t) for all lag times.
    """

    n_lags = len(lag_indices)
    M = len(kvecs)
    S4_t = np.zeros((n_lags, M), dtype=float)

    for li, lag_idx in enumerate(lag_indices):
        print(f"  Computing S4 at lag {lag_idx} frames ({li+1}/{n_lags})...")
        S4_t[li] = _compute_S4_for_lag(
            positions,
            na_disp_cumulative,
            kvecs,
            t0_indices,
            li,
            threshold
        )

    return S4_t


def _compute_S4_for_lag(
    positions: np.ndarray,
    na_disp_cumulative: np.ndarray,
    kvecs: np.ndarray,
    t0_indices: list,
    lag_li: int,
    threshold: float,
):
    """
    Internal: compute connected S4 at a single lag index.

    S4(k) = (1/N) [ <|ρ_w(k)|^2> - |<ρ_w(k)>|^2 ]
    """

    N = positions.shape[1]
    M = len(kvecs)

    S4_raw = np.zeros(M, dtype=float)
    rho_mean = np.zeros(M, dtype=complex)
    count = 0

    for ti, t0 in enumerate(t0_indices):

        # --- bounds check ---
        if ti >= na_disp_cumulative.shape[0]:
            continue
        if lag_li >= na_disp_cumulative.shape[1]:
            continue

        # --- NA displacement ---
        dr_na = na_disp_cumulative[ti, lag_li, :, :]   # (N,3)

        # --- mobility field ---
        w = compute_overlap_field(dr_na, threshold)

        # --- positions at t0 ---
        r_t0 = positions[t0, :, :]

        # --- Fourier transform ---
        rho_w = mobility_density_fourier(r_t0, w, kvecs)

        # --- accumulate ---
        S4_raw += np.abs(rho_w) ** 2
        rho_mean += rho_w
        count += 1

    if count == 0:
        return np.zeros(M)

    # --- averages ---
    S4_raw /= count
    rho_mean /= count

    # --- connected S4 ---
    S4 = (S4_raw - np.abs(rho_mean) ** 2) / N

    return S4

# ============================================================
#  SECTION 3 – χ4(t): k→0 susceptibility
# ============================================================

def compute_chi4(
    positions: np.ndarray,
    na_disp_cumulative: np.ndarray,
    lag_indices: np.ndarray,
    t0_indices: list,
    threshold: float = P.a_threshold,
) -> np.ndarray:
    """
    Compute χ4(t) = S4(k=0, t) = (1/N) [ ⟨Q^2⟩ - ⟨Q⟩^2 ] * N

    where Q(t0,t) = Σ_i w_i(t0,t) is the total overlap.

    At k=0 the Fourier sum reduces to just Σ_i w_i, so:
        |ρ_w(k=0)|^2 = (Σ_i w_i)^2 = Q^2

    and the ensemble average gives:
        S4(k=0) = (1/N) ⟨Q^2⟩  (using single-t0 estimator)

    The *variance* form χ4 = (1/N)(⟨Q^2⟩ - ⟨Q⟩^2) is the
    proper fluctuation susceptibility.  We compute both.

    Returns
    -------
    chi4_var : ndarray (n_lags,)  variance-based χ4
    chi4_s4  : ndarray (n_lags,)  S4(k=0) (mean of |ρ_w(0)|^2 / N)
    Q_mean   : ndarray (n_lags,)  mean overlap ⟨Q⟩/N (fraction of immobile)
    """
    n_lags = len(lag_indices)
    chi4_var = np.zeros(n_lags)
    chi4_s4 = np.zeros(n_lags)
    Q_mean = np.zeros(n_lags)

    for li, lag_idx in enumerate(lag_indices):
        Q_vals = []   # Q(t0) for each origin
        # Enumerate with ti = array index into na_disp_cumulative first axis
        for ti, t0 in enumerate(t0_indices):
            if ti >= na_disp_cumulative.shape[0]:
                continue
            if li >= na_disp_cumulative.shape[1]:
                continue
            dr_na = na_disp_cumulative[ti, li, :, :]   # indexed by (ti, li)
            w = compute_overlap_field(dr_na, threshold)
            Q_vals.append(w.sum())

        N = na_disp_cumulative.shape[2]
        if len(Q_vals) == 0:
            Q_mean[li] = np.nan; chi4_s4[li] = np.nan; chi4_var[li] = np.nan
            continue
        Q_arr = np.array(Q_vals, dtype=float)
        Q_mean[li] = Q_arr.mean() / N
        chi4_s4[li] = (Q_arr ** 2).mean() / N          # ⟨Q^2⟩/N
        chi4_var[li] = ((Q_arr ** 2).mean() - Q_arr.mean() ** 2) / N  # variance / N

    return chi4_var, chi4_s4, Q_mean


# ============================================================
#  SECTION 4 – Ornstein–Zernike fit and ξ4 extraction
# ============================================================

def oz_fit_function(k2, A, B):
    """
    Ornstein–Zernike model for 1/S4(k).

    1/S4(k) = A + B * k^2
    => S4(k) = S4(0) / (1 + ξ4^2 k^2)
    with ξ4 = sqrt(B/A)
    """
    return A + B * k2


def extract_xi4(
    S4_k: np.ndarray,
    kvecs: np.ndarray,
    kmags: np.ndarray,
    n_k_fit: int = P.n_k_fit,
    label: str = "",
) -> dict:
    """
    Fit 1/S4(k) = A + B k^2 to extract dynamic correlation length ξ4.

    Only the `n_k_fit` smallest non-zero |k| values are used.
    k=0 is excluded (S4(k=0) = χ4 diverges in principle; fit is infrared).

    Parameters
    ----------
    S4_k   : ndarray (M,)  S4 values (at a fixed t, usually t*)
    kvecs  : ndarray (M,3)
    kmags  : ndarray (M,)
    n_k_fit: int  number of distinct |k| shells to include in fit
    label  : str  identifier for printout

    Returns
    -------
    result : dict with keys
        xi4, A, B, k_fit, inv_S4_fit, inv_S4_data,
        R2, fit_successful
    """
    # ---- Select non-zero k points ----
    nonzero = kmags > 1e-10
    k_nz = kmags[nonzero]
    S4_nz = S4_k[nonzero]

    # ---- Keep only the n_k_fit smallest distinct |k| magnitudes ----
    unique_kmags = np.unique(np.round(k_nz, decimals=6))
    if len(unique_kmags) < 3:
        warnings.warn(f"[{label}] Too few distinct k-shells for OZ fit.")
        return _empty_fit_result()

    k_shells = unique_kmags[:n_k_fit]
    k_use, S4_use = [], []
    for kshell in k_shells:
        mask = np.abs(k_nz - kshell) < 1e-6
        # Average over degenerate directions at same |k|
        k_use.append(kshell)
        S4_use.append(S4_nz[mask].mean())

    k_arr = np.array(k_use)
    S4_arr = np.array(S4_use)

    # ---- Guard against zeros / negatives in S4 ----
    valid = S4_arr > 1e-12
    if valid.sum() < 3:
        warnings.warn(f"[{label}] Not enough positive S4 values for fit.")
        return _empty_fit_result()
    k_arr = k_arr[valid]; S4_arr = S4_arr[valid]

    k2_arr = k_arr ** 2
    inv_S4 = 1.0 / S4_arr

    # ---- Least-squares fit ----
    try:
        popt, pcov = curve_fit(
            oz_fit_function, k2_arr, inv_S4,
            p0=[inv_S4[0], (inv_S4[-1] - inv_S4[0]) / (k2_arr[-1] - k2_arr[0] + 1e-15)],
            maxfev=5000,
        )
    except RuntimeError as e:
        warnings.warn(f"[{label}] OZ fit failed: {e}")
        return _empty_fit_result()

    A, B = popt
    perr = np.sqrt(np.diag(pcov))

    # ---- Correlation length ----
    if A <= 0 or B <= 0:
        warnings.warn(f"[{label}] OZ fit gave non-physical A={A:.4f}, B={B:.4f}.")
        xi4 = np.nan
    else:
        xi4 = np.sqrt(B / A)

    # ---- Fit quality R^2 ----
    inv_S4_fit_vals = oz_fit_function(k2_arr, A, B)
    ss_res = np.sum((inv_S4 - inv_S4_fit_vals) ** 2)
    ss_tot = np.sum((inv_S4 - inv_S4.mean()) ** 2)
    R2 = 1.0 - ss_res / (ss_tot + 1e-30)

    print(f"\n  [{label}] OZ fit results:")
    print(f"    A = {A:.6f} ± {perr[0]:.6f}  (= 1/S4(k→0))")
    print(f"    B = {B:.6f} ± {perr[1]:.6f}  (slope in k^2)")
    print(f"    ξ4 = {xi4:.3f} Å")
    print(f"    R² = {R2:.6f}")

    return dict(
        xi4=xi4, A=A, B=B,
        perr_A=perr[0], perr_B=perr[1],
        k_fit=k_arr, k2_fit=k2_arr,
        inv_S4_data=inv_S4,
        inv_S4_fit=inv_S4_fit_vals,
        S4_data=S4_arr,
        R2=R2,
        fit_successful=True,
        label=label,
    )


def _empty_fit_result():
    return dict(xi4=np.nan, A=np.nan, B=np.nan, fit_successful=False,
                k_fit=np.array([]), inv_S4_data=np.array([]),
                inv_S4_fit=np.array([]), R2=np.nan, label="")

# ============================================================
#  SECTION X – Consistent t* determination via S4(k→0)
# ============================================================

def extract_chi4_from_S4(
    S4_t: np.ndarray,
    kvecs: np.ndarray,
    k_tolerance: float = 1e-6,
):
    """
    Extract χ_S4(t) ≈ S4(k→0, t)

    Parameters
    ----------
    S4_t : (n_lags, M)
    kvecs : (M,3)
    k_tolerance : float

    Returns
    -------
    chi_s4 : (n_lags,)
    k0_index : int
    """

    # Compute |k|
    k_magnitudes = np.linalg.norm(kvecs, axis=1)

    # Find smallest k (approx k→0)
    k0_index = np.argmin(k_magnitudes)

    print("\n[INFO] k→0 selection:")
    print(f"  k_min = {k_magnitudes[k0_index]:.6f} Å⁻¹ (index={k0_index})")

    chi_s4 = S4_t[:, k0_index]

    return chi_s4, k0_index


def determine_t_star(
    chi_var: np.ndarray,
    chi_s4: np.ndarray,
    lag_times_fs: np.ndarray,
):
    """
    Determine t* from both χ_var and χ_S4
    """

    lag_idx_var = np.argmax(chi_var)
    lag_idx_s4  = np.argmax(chi_s4)

    t_var = lag_times_fs[lag_idx_var]
    t_s4  = lag_times_fs[lag_idx_s4]

    print("\n============================================================")
    print("STEP X: t* determination")
    print("============================================================")

    print(f"  χ_var peak: lag={lag_idx_var} → t = {t_var:.2f} fs = {t_var/1000:.2f} ps")
    print(f"  χ_S4  peak: lag={lag_idx_s4} → t = {t_s4:.2f} fs = {t_s4/1000:.2f} ps")

    return lag_idx_var, lag_idx_s4

# ============================================================
#  SECTION 5 – Shear banding diagnostic
# ============================================================

def compute_velocity_profile(
    positions: np.ndarray,
    dt_fs: float,
    n_ybins: int = P.n_ybins,
    L: float = P.L,
    frame_start: int = 0,
    frame_end: int = -1,
) -> tuple:
    """
    Compute the mean streamwise velocity profile vx(y) to diagnose shear banding.

    Method
    ------
    vx(y) ≈ ⟨ Δx_i / Δt ⟩_{particles in bin, time average}

    We use consecutive-frame displacements averaged over the steady-state
    window, then bin by y coordinate.

    Physical relevance
    ------------------
    For homogeneous simple shear the profile must be linear: vx(y) = γ̇ · y.
    Deviation from linearity signals shear banding — a flow instability
    that would invalidate the assumption of a uniform shear field and
    corrupt the S4 analysis.

    Returns
    -------
    y_bins  : ndarray (n_ybins,)  bin centres (Å)
    vx_mean : ndarray (n_ybins,)  mean vx (Å/fs)
    vx_std  : ndarray (n_ybins,)  standard deviation within bin
    linear_fit : dict  slope, intercept, R^2 of linear regression
    """
    if frame_end == -1:
        frame_end = positions.shape[0]

    frames = positions[frame_start:frame_end]   # (F, N, 3)
    # Δx / Δt for each particle between consecutive frames
    dx = np.diff(frames[:, :, 0], axis=0)       # (F-1, N)
    dy = frames[:-1, :, 1]                       # y positions at earlier frame (F-1, N)

    # Flatten time axis
    dx_flat = dx.ravel()         # (F-1)*N
    dy_flat = dy.ravel()

    # Bin by y
    bin_edges = np.linspace(0.0, L, n_ybins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Map y into [0, L) using PBC
    dy_pbc = dy_flat % L

    vx_mean = np.zeros(n_ybins)
    vx_std = np.zeros(n_ybins)
    for b in range(n_ybins):
        mask = (dy_pbc >= bin_edges[b]) & (dy_pbc < bin_edges[b + 1])
        if mask.sum() > 5:
            vx_vals = dx_flat[mask] / dt_fs
            vx_mean[b] = vx_vals.mean()
            vx_std[b] = vx_vals.std()
        else:
            vx_mean[b] = np.nan
            vx_std[b] = np.nan

    # Linear regression to check homogeneity
    good = np.isfinite(vx_mean)
    y_g = bin_centres[good]; vx_g = vx_mean[good]
    coeffs = np.polyfit(y_g, vx_g, 1)
    slope, intercept = coeffs
    vx_linear = np.polyval(coeffs, y_g)
    ss_res = np.sum((vx_g - vx_linear) ** 2)
    ss_tot = np.sum((vx_g - vx_g.mean()) ** 2)
    R2_lin = 1.0 - ss_res / (ss_tot + 1e-30)

    linear_fit = dict(slope=slope, intercept=intercept, R2=R2_lin)
    print(f"\n  Shear banding check:")
    print(f"    Linear fit slope = {slope:.6g} Å/fs per Å  (expected ≈ {P.shear_rate:.2e})")
    print(f"    R² of linear vx(y) = {R2_lin:.6f}  (should be > 0.95 for no banding)")

    return bin_centres, vx_mean, vx_std, linear_fit


# ============================================================
#  SECTION 6 – Data loading helpers (MDAnalysis)
# ============================================================

def load_trajectory(
    topology_file: str,
    trajectory_file: str,
    params: SimParams = P,
) -> tuple:
    """
    Load trajectory using MDAnalysis and return:
        positions            : ndarray (n_frames, N, 3) in Å (unwrapped)
        na_disp_cumulative   : ndarray (n_frames, n_lags_max, N, 3)
        lag_times_fs         : ndarray (n_lags_max,)
        t0_indices           : list of int

    Non-affine displacements
    -------------------------
    We assume that `non_affine_disp` has already been computed (i.e., the
    affine shear contribution γ̇·y·t has been subtracted) and is stored as
    a secondary field in the trajectory, OR we compute it on-the-fly using
    the unwrapped positions (for a cubic box with simple shear):

        Δr^{affine}_{i,x}(t0, t) = ∫γ̇ · y_i(τ) · dτ
        Δr^{affine}_{i,y} = 0
        Δr^{affine}_{i,z} = 0

    The NA displacement is then:
        Δr^{NA}_i(t0, t) = [r_i(t) - r_i(t0)] - Δr^{affine}_i(t0, t)

    Parameters
    ----------
    topology_file   : PDB/GRO/TPR file
    trajectory_file : DCD/XTC/TRR/LAMMPSDUMP file

    Returns
    -------
    See header
    """
    print(f"\nLoading trajectory: {trajectory_file}")
    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    atoms = u.select_atoms("type 1")
    N = len(atoms)
    n_frames = len(u.trajectory)
    print(f"  Atoms: {N},  Frames: {n_frames}")

    # ---- Load all positions ----
    positions = np.zeros((n_frames, N, 3), dtype=np.float32)
    for fi, ts in enumerate(u.trajectory):
        positions[fi] = atoms.positions.copy()

    positions = positions.astype(float)  # promote to float64

    # ---- Determine lag times ----
    # Use a geometric-ish spread of lag times from 1 dump interval to
    # half the trajectory, sampling ~20 points.
    max_lag_frames = n_frames // 2
    lag_steps = np.unique(
        np.round(np.geomspace(1, max_lag_frames, 30)).astype(int)
    )
    lag_steps = lag_steps[lag_steps < n_frames]
    n_lags = len(lag_steps)
    print(f"  Lag times (frames): {lag_steps}")

    # ---- Determine t0 indices ----
    # Use steady-state portion of trajectory only.
    frame_start = int(params.t0_start_frac * n_frames)
    frame_end = n_frames - lag_steps.max() - 1
    if frame_end <= frame_start:
        warnings.warn("Trajectory too short for requested t0 range.")
        frame_end = n_frames // 2
    t0_candidates = np.linspace(frame_start, frame_end, params.n_t0, dtype=int)
    t0_indices = sorted(set(t0_candidates.tolist()))
    print(f"  t0 origins (frames): {t0_indices[0]} to {t0_indices[-1]}, n={len(t0_indices)}")

    # ---- Build cumulative NA displacements ----
    # na_disp_cumulative[t0, lag_idx, i, :] = Δr^NA_i(t0, t0+lag_steps[lag_idx])
    #
    # Memory estimate: n_t0 × n_lags × N × 3 × 8 bytes
    # For n_t0=20, n_lags=30, N=15000: ~216 MB (acceptable)
    print("  Computing non-affine displacements...")
    na_disp = np.zeros((len(t0_indices), n_lags, N, 3), dtype=np.float32)

    dt_per_frame = params.dt_dump   # fs per frame
    gamma_dot = params.shear_rate   # 1/fs in real units

    for ti, t0 in enumerate(t0_indices):  # ti = array index, t0 = frame index
        r_t0 = positions[t0]   # (N,3)
        y_t0 = r_t0[:, 1]     # y-coordinates at t0 (N,)

        for li, lag in enumerate(lag_steps):  # li = array index, lag = frame count
            t1 = t0 + lag
            if t1 >= n_frames:
                na_disp[ti, li] = 0.0
                continue

            r_t1 = positions[t1]   # (N,3)
            dr_total = r_t1 - r_t0  # total displacement (N,3)

            # Affine contribution in x: Δx^aff = ∑γ̇ · y(t_n) · Δt
            y_traj = positions[t0:t1, :, 1]
            dr_affine = np.zeros_like(dr_total)
            dr_affine[:, 0] = gamma_dot * np.sum(y_traj, axis=0) * dt_per_frame

            na_disp[ti, li] = (dr_total - dr_affine).astype(np.float32)

    print("  Done building NA displacements.")

    lag_times_fs = lag_steps * params.dt_dump
    return positions, na_disp, lag_times_fs, lag_steps, t0_indices, N


# ============================================================
#  SECTION 7 – Directional anisotropy check
# ============================================================

def check_anisotropy(
    S4_ky: np.ndarray,   # S4 values along y-axis k-points
    S4_kz: np.ndarray,   # S4 values along z-axis k-points
    kmags_1d: np.ndarray,
) -> bool:
    """
    Compare S4(ky) vs S4(kz) to decide if angular averaging is safe.

    Returns True if the two directions agree within 10% at each k shell,
    False if significant anisotropy is detected.
    """
    # Use only non-zero k points
    nz = kmags_1d > 1e-10
    ky_nz = S4_ky[nz]; kz_nz = S4_kz[nz]; k_nz = kmags_1d[nz]

    n_compare = min(len(ky_nz), len(kz_nz))
    if n_compare == 0:
        return True

    ratio = ky_nz[:n_compare] / (kz_nz[:n_compare] + 1e-30)
    frac_diff = np.abs(ratio - 1.0)
    max_anisotropy = frac_diff.max()
    mean_anisotropy = frac_diff.mean()

    print(f"\n  Anisotropy check  (S4_ky vs S4_kz):")
    print(f"    Max |S4_ky/S4_kz - 1| = {max_anisotropy:.4f}")
    print(f"    Mean |S4_ky/S4_kz - 1| = {mean_anisotropy:.4f}")

    if max_anisotropy < 0.10:
        print("    → ISOTROPIC (within 10%).  yz-plane angular average OK.")
        return True
    else:
        print("    → ANISOTROPIC.  Keeping ky, kz directions separate.")
        return False


# ============================================================
#  SECTION 8 – Plotting
# ============================================================
def plot_chi4_comparison(
    lag_times_fs: np.ndarray,
    chi_var: np.ndarray,
    chi_s4: np.ndarray,
    save_path: str = None,
    dpi: int=300
):
    """
    Plot χ_var vs χ_S4 comparison
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))

    plt.plot(lag_times_fs / 1000, chi_var, 'o-', label=r'$\chi_4^{var}(t)$')
    plt.plot(lag_times_fs / 1000, chi_s4, 's-', label=r'$S_4(k\to0, t)$')

    plt.xlabel("t (ps)")
    plt.ylabel("Dynamic susceptibility")
    plt.legend()
    plt.title("χ₄(t) consistency check")

    # mark peaks
    i1 = np.argmax(chi_var)
    i2 = np.argmax(chi_s4)

    plt.axvline(lag_times_fs[i1]/1000, linestyle='--', alpha=0.5)
    plt.axvline(lag_times_fs[i2]/1000, linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
        print(f"[Saved] χ4 comparison plot → {save_path}")

    plt.close()

def make_all_plots(
    lag_times_fs: np.ndarray,
    chi4_var: np.ndarray,
    chi4_s4: np.ndarray,
    Q_mean: np.ndarray,
    S4_z_vs_k: dict,   # dict keyed by lag_idx: S4 along kz
    S4_y_vs_k: dict,
    S4_yz_vs_k: dict,
    kmags_z: np.ndarray,
    kmags_y: np.ndarray,
    kmags_yz: np.ndarray,
    fit_z: dict,       # OZ fit result along z
    fit_y: dict,
    fit_yz: dict,
    tstar_idx: int,
    vx_profile: tuple,
    output_dir: Path = Path("."),
    params: SimParams = P,
):
    """Generate and save all diagnostic plots."""
    fig_dpi = params.fig_dpi

    # ------------------------------------------------------------------
    # Figure 1: χ4(t) and mean overlap Q(t)
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(lag_times_fs / 1e3, chi4_var, "o-", color="blue", label=r"$\chi_4$ (variance)")
    ax.set_xlabel("lag time  $t$  (ps)")
    ax.set_ylabel(r"$\chi_4$ (variance)", color="blue")
    ax.tick_params(axis='y', labelcolor="blue")

    ax2 = ax.twinx()
    ax2.plot(lag_times_fs / 1e3, chi4_s4, "s--", color="orange", label=r"$S_4(k=0)$")
    ax2.set_ylabel(r"$S_4(k=0)$", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    tstar = lag_times_fs[tstar_idx]
    ax.axvline(tstar / 1e3, color="red", lw=1.5, ls=":", label=f"$t^*$ = {tstar/1e3:.1f} ps")
    ax.set_title("Four-point susceptibility")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(lag_times_fs / 1e3, Q_mean, "^-", color="green")
    ax.axvline(tstar / 1e3, color="red", lw=1.5, ls=":")
    ax.set_xlabel("lag time  $t$  (ps)")
    ax.set_ylabel(r"$\langle Q \rangle / N$  (immobile fraction)")
    ax.set_title("Mean overlap function")
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "chi4_vs_t.png", dpi=fig_dpi)
    plt.close(fig1)
    print("  Saved chi4_vs_t.png")

    # ------------------------------------------------------------------
    # Figure 2: S4(k) along z and y at t*
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lag_labels = {tstar_idx: rf"$t^*$={tstar/1e3:.1f} ps"}
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(S4_z_vs_k)))

    for ci, (li, S4_vals) in enumerate(S4_z_vs_k.items()):
        nz = kmags_z > 1e-10
        lab = rf"$t$={lag_times_fs[li]/1e3:.1f} ps"
        lw = 2.5 if li == tstar_idx else 1.0
        axes[0].plot(kmags_z[nz], S4_vals[nz], "o-",
                     color=colors[ci], lw=lw, ms=4, label=lab)

    for ci, (li, S4_vals) in enumerate(S4_y_vs_k.items()):
        nz = kmags_y > 1e-10
        lab = rf"$t$={lag_times_fs[li]/1e3:.1f} ps"
        lw = 2.5 if li == tstar_idx else 1.0
        axes[1].plot(kmags_y[nz], S4_vals[nz], "s--",
                     color=colors[ci], lw=lw, ms=4, label=lab)

    for ax, direction in zip(axes, ["$k_z$", "$k_y$"]):
        ax.set_xlabel(rf"$|k|$  (Å$^{{-1}}$)")
        ax.set_ylabel(r"$S_4(k, t)$")
        ax.set_title(f"$S_4$ along {direction}")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "S4_k_directional.png", dpi=fig_dpi)
    plt.close(fig2)
    print("  Saved S4_k_directional.png")

    # ------------------------------------------------------------------
    # Figure 3: OZ fit  1/S4 vs k^2  at t*
    # ------------------------------------------------------------------
    fig3, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, fit, title in zip(axes, [fit_z, fit_y, fit_yz],
                               ["Along $k_z$", "Along $k_y$", "yz plane"]):
        if not fit.get("fit_successful", False):
            ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes, ha="center")
            continue
        ax.plot(fit["k2_fit"], fit["inv_S4_data"], "o", ms=7,
                label="Data $1/S_4$")
        k2_fine = np.linspace(0, fit["k2_fit"].max() * 1.1, 200)
        ax.plot(k2_fine, oz_fit_function(k2_fine, fit["A"], fit["B"]),
                "-", lw=2, label=f"OZ fit\n$\\xi_4$={fit['xi4']:.2f} Å\n$R^2$={fit['R2']:.4f}")
        ax.set_xlabel(r"$k^2$  (Å$^{-2}$)")
        ax.set_ylabel(r"$1/S_4(k, t^*)$")
        ax.set_title(f"OZ fit — {title}")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig3.suptitle(rf"Ornstein–Zernike fit at $t^*$ = {tstar/1e3:.1f} ps", y=1.01)
    fig3.tight_layout()
    fig3.savefig(output_dir / "OZ_fit.png", dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved OZ_fit.png")

    # ------------------------------------------------------------------
    # Figure 4: S4(k) in yz plane (2-D heat map at t*)
    # ------------------------------------------------------------------
    _plot_S4_yz_heatmap(
        S4_yz_vs_k.get(tstar_idx, None),
        kmags_yz, params.L, tstar,
        output_dir / "S4_yz_plane.png", fig_dpi
    )
    # --- plot comparison ---
    # plot_chi4_comparison(lag_times_fs, chi4, chi_s4, output_dir / "chi_4_comparison.ong", fig_dpi)
    # ------------------------------------------------------------------
    # Figure 5: Shear banding diagnostic
    # ------------------------------------------------------------------
    y_bins, vx_mean, vx_std, lin_fit = vx_profile
    fig5, ax = plt.subplots(figsize=(6, 4.5))
    good = np.isfinite(vx_mean)
    ax.errorbar(y_bins[good], vx_mean[good] * 1e3,
                yerr=vx_std[good] * 1e3,
                fmt="o", ms=5, capsize=3, label="$v_x(y)$ data")
    # Expected linear profile
    y_fine = np.linspace(0, params.L, 200)
    ax.plot(y_fine, params.shear_rate * y_fine * 1e3,
            "--", lw=2, color="red",
            label=rf"Expected: $\dot{{\gamma}} y$ (γ̇={params.shear_rate:.1e})")
    ax.set_xlabel("$y$  (Å)")
    ax.set_ylabel("$v_x$  (mÅ/fs)")
    ax.set_title(f"Shear banding check  (R² = {lin_fit['R2']:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(output_dir / "shear_banding.png", dpi=fig_dpi)
    plt.close(fig5)
    print("  Saved shear_banding.png")


def _plot_S4_yz_heatmap(S4_vals, kmags_yz, L, tstar_fs, outpath, dpi):
    """Heat map of S4(ky, kz) in the yz plane."""
    if S4_vals is None:
        return
    dk = 2.0 * np.pi / L
    # Reconstruct integer k components from magnitudes — approximate
    fig, ax = plt.subplots(figsize=(5.5, 5))
    # Scatter plot of |k| vs S4 coloured by magnitude
    ax.scatter(kmags_yz, S4_vals, c=S4_vals, cmap="hot_r", s=20, alpha=0.7)
    ax.set_xlabel(r"$|\mathbf{k}|$  (Å$^{-1}$)")
    ax.set_ylabel(r"$S_4(k_x=0, k_y, k_z, t^*)$")
    ax.set_title(rf"$S_4$ in yz plane at $t^*$ = {tstar_fs/1e3:.1f} ps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print("  Saved S4_yz_plane.png")


# ============================================================
#  SECTION 9 – Main analysis driver
# ============================================================

def run_analysis(
    topology_file: str,
    trajectory_file: str,
    output_dir: str = "s4_output",
    params: SimParams = P,
) -> dict:
    """
    Full analysis pipeline.

    Parameters
    ----------
    topology_file    : MDAnalysis-readable topology (PDB, GRO, PSF, ...)
    trajectory_file  : trajectory file (DCD, XTC, LAMMPSDUMP, ...)
    output_dir       : directory where figures and data are saved
    params           : SimParams object with all physical parameters

    Returns
    -------
    results : dict containing:
        chi4_var       ndarray (n_lags,)
        chi4_s4        ndarray (n_lags,)
        Q_mean         ndarray (n_lags,)
        lag_times_fs   ndarray (n_lags,)
        tstar_fs       float
        S4_kz          dict  {lag_idx: ndarray(M_z,)}
        S4_ky          dict
        S4_kyz         dict
        fit_z          dict  OZ fit along kz
        fit_y          dict
        fit_yz         dict
        xi4_z          float (Å)
        xi4_y          float (Å)
        xi4_yz         float (Å)
        vx_profile     tuple (y_bins, vx_mean, vx_std, linear_fit)
        anisotropic    bool
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # STEP 1 – Load trajectory
    # ================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading trajectory")
    print("="*60)
    (positions, na_disp_cumulative,
     lag_times_fs, lag_steps,
     t0_indices, N) = load_trajectory(topology_file, trajectory_file, params)

    n_lags = len(lag_steps)

    # ================================================================
    # STEP 2 – Shear banding check (early, so we can warn fast)
    # ================================================================
    print("\n" + "="*60)
    print("STEP 2: Shear banding diagnostic")
    print("="*60)
    # Use a subset of frames for speed
    frame_start_band = int(params.t0_start_frac * positions.shape[0])
    vx_prof = compute_velocity_profile(
        positions, params.dt_dump,
        n_ybins=params.n_ybins, L=params.L,
        frame_start=frame_start_band,
    )
    if vx_prof[3]["R2"] < 0.90:
        warnings.warn(
            "Shear banding may be present (R² of linear vx(y) < 0.90). "
            "S4 analysis may be compromised."
        )

    # ================================================================
    # STEP 3 – Build k-vector sets
    # ================================================================
    print("\n" + "="*60)
    print("STEP 3: Building k-space grids")
    print("="*60)
    # 1-D along z (kx=ky=0)
    kvecs_z, kmags_z = build_kvectors_axis(params.L, params.n_kmax, axis=2)
    # 1-D along y (kx=kz=0)
    kvecs_y, kmags_y = build_kvectors_axis(params.L, params.n_kmax, axis=1)
    # yz plane (kx=0)
    kvecs_yz, kmags_yz = build_kvectors_plane(params.L, params.n_kmax, fixed_axis=0)

    print(f"  k along z: {len(kvecs_z)} vectors  (k_min = {2*np.pi/params.L:.4f} Å⁻¹)")
    print(f"  k along y: {len(kvecs_y)} vectors")
    print(f"  k in yz plane: {len(kvecs_yz)} vectors")

    # ================================================================
    # STEP 4 – Compute χ4(t) and χS4
    # ================================================================
    print("\n" + "="*60)
    print("STEP 4: Computing χ4(t) and χS4")
    print("="*60)
    chi4_var, chi4_s4, Q_mean = compute_chi4(
        positions, na_disp_cumulative, lag_steps, t0_indices, params.a_threshold
    )
    # chi_s4, k0_idx = extract_chi4_from_S4(S4_t, kvecs)
    # lag_var, lag_s4 = determine_t_star(chi4, chi_s4, lag_times_fs)
    # Peak of χ4 → t*
    tstar_idx = int(np.argmax(chi4_var))
    
    # using χS4 instead of χ4 to determine tstar_idx
    # tstar_idx = lag_s4
    tstar_fs = lag_times_fs[tstar_idx]
    print(f"\n  χ4 peaks at lag index {tstar_idx} → t* = {tstar_fs:.1f} fs = {tstar_fs/1e3:.2f} ps")

    # ================================================================
    # STEP 5 – Compute S4(k, t) along z and y
    # ================================================================
    print("\n" + "="*60)
    print("STEP 5: Computing S4(k,t) — directional")
    print("="*60)

    # Compute at all lag times (for time evolution plots)
    print("  Direction: kz")
    S4_kz_all = compute_S4_time_series(
        positions, na_disp_cumulative, kvecs_z, lag_steps, t0_indices, params.a_threshold
    )   # (n_lags, M_z)

    print("\n  Direction: ky")
    S4_ky_all = compute_S4_time_series(
        positions, na_disp_cumulative, kvecs_y, lag_steps, t0_indices, params.a_threshold
    )   # (n_lags, M_y)

    print("\n  Direction: yz plane (kx=0)")
    S4_kyz_all = compute_S4_time_series(
        positions, na_disp_cumulative, kvecs_yz, lag_steps, t0_indices, params.a_threshold
    )  # (n_lags, M_yz)

    # Build dicts keyed by lag index for convenient access
    S4_kz_dict = {li: S4_kz_all[li] for li in range(n_lags)}
    S4_ky_dict = {li: S4_ky_all[li] for li in range(n_lags)}
    S4_kyz_dict = {li: S4_kyz_all[li] for li in range(n_lags)}

    # ================================================================
    # STEP 6 – Anisotropy check
    # ================================================================
    print("\n" + "="*60)
    print("STEP 6: Anisotropy check")
    print("="*60)
    is_isotropic = check_anisotropy(
        S4_ky_all[tstar_idx], S4_kz_all[tstar_idx], kmags_z
    )

    # ================================================================
    # STEP 7 – Extract ξ4 via OZ fit at t*
    # ================================================================
    print("\n" + "="*60)
    print("STEP 7: Extracting ξ4 via OZ fit at t*")
    print("="*60)

    fit_z = extract_xi4(
        S4_kz_all[tstar_idx], kvecs_z, kmags_z,
        n_k_fit=params.n_k_fit, label="kz"
    )
    fit_y = extract_xi4(
        S4_ky_all[tstar_idx], kvecs_y, kmags_y,
        n_k_fit=params.n_k_fit, label="ky"
    )
    fit_yz = extract_xi4(
        S4_kyz_all[tstar_idx], kvecs_yz, kmags_yz,
        n_k_fit=params.n_k_fit, label="yz-plane"
    )

    xi4_z = fit_z["xi4"]
    xi4_y = fit_y["xi4"]
    xi4_yz = fit_yz["xi4"]
    print(f"\n  ξ4 summary:")
    print(f"    Along kz  : {xi4_z:.3f} Å")
    print(f"    Along ky  : {xi4_y:.3f} Å")
    print(f"    yz plane  : {xi4_yz:.3f} Å")

    # ================================================================
    # STEP 8 – Save numerical results
    # ================================================================
    print("\n" + "="*60)
    print("STEP 8: Saving data arrays")
    print("="*60)
    np.savez(
        out / "s4_results.npz",
        lag_times_fs=lag_times_fs,
        chi4_var=chi4_var,
        chi4_s4=chi4_s4,
        Q_mean=Q_mean,
        S4_kz_all=S4_kz_all,
        S4_ky_all=S4_ky_all,
        S4_kyz_all=S4_kyz_all,
        kmags_z=kmags_z,
        kmags_y=kmags_y,
        kmags_yz=kmags_yz,
        tstar_idx=np.array([tstar_idx]),
        tstar_fs=np.array([tstar_fs]),
        xi4_z=np.array([xi4_z]),
        xi4_y=np.array([xi4_y]),
        xi4_yz=np.array([xi4_yz]),
    )
    print(f"  Saved s4_results.npz")

    # ================================================================
    # STEP 9 – Plotting
    # ================================================================
    print("\n" + "="*60)
    print("STEP 9: Generating figures")
    print("="*60)
    # Subsample lag times for cleaner plots (every 3rd)
    plot_lags = list(range(0, n_lags, max(1, n_lags // 8))) + [tstar_idx]
    plot_lags = sorted(set(plot_lags))

    make_all_plots(
        lag_times_fs=lag_times_fs,
        chi4_var=chi4_var,
        chi4_s4=chi4_s4,
        Q_mean=Q_mean,
        S4_z_vs_k={li: S4_kz_dict[li] for li in plot_lags},
        S4_y_vs_k={li: S4_ky_dict[li] for li in plot_lags},
        S4_yz_vs_k={li: S4_kyz_dict[li] for li in plot_lags},
        kmags_z=kmags_z,
        kmags_y=kmags_y,
        kmags_yz=kmags_yz,
        fit_z=fit_z,
        fit_y=fit_y,
        fit_yz=fit_yz,
        tstar_idx=tstar_idx,
        vx_profile=vx_prof,
        output_dir=out,
        params=params,
    )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"  Output directory: {out.resolve()}")

    return dict(
        chi4_var=chi4_var,
        chi4_s4=chi4_s4,
        Q_mean=Q_mean,
        lag_times_fs=lag_times_fs,
        tstar_fs=tstar_fs,
        tstar_idx=tstar_idx,
        S4_kz=S4_kz_dict,
        S4_ky=S4_ky_dict,
        S4_kyz=S4_kyz_dict,
        kmags_z=kmags_z,
        kmags_y=kmags_y,
        kmags_yz=kmags_yz,
        fit_z=fit_z,
        fit_y=fit_y,
        fit_yz=fit_yz,
        xi4_z=xi4_z,
        xi4_y=xi4_y,
        xi4_yz=xi4_yz,
        vx_profile=vx_prof,
        anisotropic=(not is_isotropic),
    )


# ============================================================
#  SECTION 10 – Synthetic trajectory generator (for testing)
# ============================================================

def generate_synthetic_trajectory(
    N: int = 2000,
    n_frames: int = 200,
    L: float = 50.0,
    dt: float = 25.0,
    gamma_dot: float = 5e-6,
    seed: int = 42,
    outfile: str = "synthetic_traj.npz",
) -> str:
    """
    Generate a synthetic supercooled-like trajectory for unit testing.

    Physics model
    -------------
    * Particles start on a random lattice.
    * Each particle has a cage-escape time drawn from a log-normal
      distribution (mimicking heterogeneous dynamics).
    * Before escape: particles diffuse slowly (D_cage = 0.001 Å²/fs).
    * After escape : particles diffuse fast  (D_free = 0.05 Å²/fs).
    * Affine shear is added: Δx += γ̇ · y(t) · dt at each step.
    * The trajectory is saved as unwrapped positions.

    Returns
    -------
    Path to saved npz file.
    """
    rng = np.random.default_rng(seed)
    print(f"\nGenerating synthetic trajectory: N={N}, frames={n_frames}")

    # Initial positions
    pos = rng.uniform(0, L, (N, 3))
    pos_unwrapped = pos.copy()

    # Cage-escape frame for each particle
    mean_escape = n_frames * 0.4   # peak around 40% of trajectory
    std_escape = n_frames * 0.2
    escape_frame = (rng.lognormal(np.log(mean_escape), 0.5, N)).astype(int)
    escape_frame = np.clip(escape_frame, 1, n_frames - 1)

    D_cage = 0.001  # Å²/fs
    D_free = 0.05

    traj = np.zeros((n_frames, N, 3), dtype=np.float32)
    traj[0] = pos

    for fi in range(1, n_frames):
        # Decide diffusion coefficient per particle
        D = np.where(fi < escape_frame, D_cage, D_free)    # (N,)
        sigma = np.sqrt(2 * D * dt)                         # (N,)

        # Random displacement
        dr = rng.normal(0, sigma[:, np.newaxis], (N, 3))

        # Affine shear contribution in x
        dr[:, 0] += gamma_dot * pos_unwrapped[:, 1] * dt

        pos_unwrapped = pos_unwrapped + dr
        traj[fi] = pos_unwrapped.astype(np.float32)

    np.savez(outfile, positions=traj, L=L, dt=dt,
             gamma_dot=gamma_dot, escape_frame=escape_frame)
    print(f"  Saved synthetic trajectory to {outfile}")
    return outfile


class SyntheticUniverse:
    """
    Minimal MDAnalysis-like interface wrapping the synthetic npz trajectory.
    Allows run_analysis() to be tested without a real MD file.

    Usage
    -----
    su = SyntheticUniverse("synthetic_traj.npz")
    # Then pass su.topology and su.trajectory to run_analysis()
    # OR call su.run_direct_analysis()
    """

    def __init__(self, npz_file: str):
        data = np.load(npz_file)
        self.positions = data["positions"].astype(float)  # (n_frames, N, 3)
        self.L = float(data["L"])
        self.dt = float(data["dt"])
        self.gamma_dot = float(data["gamma_dot"])
        self.n_frames, self.N, _ = self.positions.shape
        print(f"SyntheticUniverse: {self.n_frames} frames, {self.N} particles, L={self.L} Å")

    def run_direct_analysis(self, output_dir: str = "s4_output_synthetic") -> dict:
        """Run the full analysis on the synthetic trajectory without MDAnalysis."""
        params = SimParams()
        params.L = self.L
        params.dt_dump = self.dt
        params.shear_rate = self.gamma_dot
        params.total_time = self.n_frames * self.dt

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        positions = self.positions
        n_frames = self.n_frames
        N = self.N

        # ---- Build lag steps ----
        max_lag = n_frames // 2

        lag_linear = np.arange(1, 200)
        lag_geom = np.unique(np.round(np.geomspace(200, max_lag, 40)).astype(int))
        lag_steps = np.unique(np.concatenate([lag_linear, lag_geom]))
        lag_steps = lag_steps[lag_steps < n_frames]
        
        # lag_steps = np.unique(np.round(np.geomspace(1, max_lag, 25)).astype(int))
        # lag_steps = lag_steps[lag_steps < n_frames]
        lag_times_fs = lag_steps * params.dt_dump

        # ---- t0 indices ----
        frame_start = int(params.t0_start_frac * n_frames)
        frame_end = n_frames - lag_steps.max() - 1
        t0_indices = sorted(set(
            np.linspace(frame_start, max(frame_start + 1, frame_end),
                        params.n_t0, dtype=int).tolist()
        ))

        # ---- Build NA displacements (subtract affine) ----
        n_lags = len(lag_steps)
        print("Computing NA displacements for synthetic trajectory...")
        na_disp = np.zeros((len(t0_indices), n_lags, N, 3), dtype=np.float32)

        for ti, t0 in enumerate(t0_indices):
            r_t0 = positions[t0]
            y_t0 = r_t0[:, 1]
            for li, lag in enumerate(lag_steps):
                t1 = t0 + lag
                if t1 >= n_frames:
                    continue
                dr_total = positions[t1] - r_t0
                delta_t = lag * params.dt_dump
                dr_affine = np.zeros_like(dr_total)
                dr_affine[:, 0] = params.shear_rate * y_t0 * delta_t
                na_disp[ti, li] = (dr_total - dr_affine).astype(np.float32)

        # ---- χ4(t) ----
        print("Computing χ4(t)...")
        chi4_var, chi4_s4, Q_mean = compute_chi4(
            positions, na_disp, lag_steps, t0_indices, params.a_threshold
        )
        tstar_idx = int(np.argmax(chi4_var)) + 1
        tstar_fs = lag_times_fs[tstar_idx]
        print(f"  t* = {tstar_fs:.1f} fs")

        # ---- k-vectors ----
        kvecs_z, kmags_z = build_kvectors_axis(params.L, params.n_kmax, axis=2)
        kvecs_y, kmags_y = build_kvectors_axis(params.L, params.n_kmax, axis=1)
        kvecs_yz, kmags_yz = build_kvectors_plane(params.L, params.n_kmax, fixed_axis=0)

        # ---- S4(k,t) ----
        print("Computing S4(k,t) along kz...")
        S4_kz_all = compute_S4_time_series(
            positions, na_disp, kvecs_z, lag_steps, t0_indices, params.a_threshold)
        print("Computing S4(k,t) along ky...")
        S4_ky_all = compute_S4_time_series(
            positions, na_disp, kvecs_y, lag_steps, t0_indices, params.a_threshold)
        print("Computing S4(k,t) in yz plane...")
        S4_kyz_all = compute_S4_time_series(
            positions, na_disp, kvecs_yz, lag_steps, t0_indices, params.a_threshold)

        # ---- OZ fits ----
        fit_z = extract_xi4(S4_kz_all[tstar_idx], kvecs_z, kmags_z,
                            n_k_fit=params.n_k_fit, label="kz")
        fit_y = extract_xi4(S4_ky_all[tstar_idx], kvecs_y, kmags_y,
                            n_k_fit=params.n_k_fit, label="ky")
        fit_yz = extract_xi4(S4_kyz_all[tstar_idx], kvecs_yz, kmags_yz,
                             n_k_fit=params.n_k_fit, label="yz-plane")

        # ---- Shear banding ----
        vx_prof = compute_velocity_profile(
            positions, params.dt_dump, params.n_ybins, params.L,
            frame_start=int(0.3 * n_frames))

        # ---- Anisotropy ----
        check_anisotropy(S4_ky_all[tstar_idx], S4_kz_all[tstar_idx], kmags_z)

        # ---- Plots ----
        n_lags = len(lag_steps)
        plot_lags = list(range(0, n_lags, max(1, n_lags // 8))) + [tstar_idx]
        plot_lags = sorted(set(plot_lags))

        S4_kz_dict = {li: S4_kz_all[li] for li in range(n_lags)}
        S4_ky_dict = {li: S4_ky_all[li] for li in range(n_lags)}
        S4_kyz_dict = {li: S4_kyz_all[li] for li in range(n_lags)}

        make_all_plots(
            lag_times_fs=lag_times_fs,
            chi4_var=chi4_var, chi4_s4=chi4_s4, Q_mean=Q_mean,
            S4_z_vs_k={li: S4_kz_dict[li] for li in plot_lags},
            S4_y_vs_k={li: S4_ky_dict[li] for li in plot_lags},
            S4_yz_vs_k={li: S4_kyz_dict[li] for li in plot_lags},
            kmags_z=kmags_z, kmags_y=kmags_y, kmags_yz=kmags_yz,
            fit_z=fit_z, fit_y=fit_y, fit_yz=fit_yz,
            tstar_idx=tstar_idx, vx_profile=vx_prof,
            output_dir=out, params=params,
        )

        np.savez(out / "s4_results.npz",
                 lag_times_fs=lag_times_fs,
                 chi4_var=chi4_var, chi4_s4=chi4_s4, Q_mean=Q_mean,
                 S4_kz_all=S4_kz_all, S4_ky_all=S4_ky_all, S4_kyz_all=S4_kyz_all,
                 kmags_z=kmags_z, kmags_y=kmags_y, kmags_yz=kmags_yz,
                 tstar_fs=np.array([tstar_fs]),
                 xi4_z=np.array([fit_z["xi4"]]),
                 xi4_y=np.array([fit_y["xi4"]]),
                 xi4_yz=np.array([fit_yz["xi4"]]))

        print(f"\nResults saved to {out.resolve()}")
        return dict(chi4_var=chi4_var, chi4_s4=chi4_s4, Q_mean=Q_mean,
                    lag_times_fs=lag_times_fs, tstar_fs=tstar_fs,
                    xi4_z=fit_z["xi4"], xi4_y=fit_y["xi4"], xi4_yz=fit_yz["xi4"],
                    fit_z=fit_z, fit_y=fit_y, fit_yz=fit_yz)


# ============================================================
#  SECTION 11 – Entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Real trajectory mode
        topology_file = sys.argv[1]
        trajectory_file = sys.argv[2]
        results = run_analysis(topology_file, trajectory_file)

    elif len(sys.argv) == 2 and sys.argv[1] == "--test":
        # Self-test with synthetic trajectory
        print("\n*** Running self-test with synthetic trajectory ***\n")
        npz = generate_synthetic_trajectory(
            N=3000, n_frames=160, L=50.0, dt=25.0,
            gamma_dot=5e-6, seed=42,
        )
        su = SyntheticUniverse(npz)
        results = su.run_direct_analysis(output_dir="s4_output_test")
        print(f"\nTest complete.  ξ4 (kz) = {results['xi4_z']:.3f} Å")

    else:
        print(__doc__)
        print("\nUsage:")
        print("  python s4_analysis.py  topology.pdb  trajectory.dcd")
        print("  python s4_analysis.py  --test          (synthetic self-test)")
