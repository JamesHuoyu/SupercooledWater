# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Density-Field Cluster Analysis
================================

Identifies mesoscopic dynamic clusters and their boundaries from a
smoothed propensity density field, then characterises each cluster
by its initial ζ_cg structural order (from ZetaOrderParameter).

Pipeline (following the document's prescription)
-------------------------------------------------

  Step 1 – Build smoothed propensity density field
    ρ_P(r) = Σ_i  w_i · K(r − r_i^0)
    where  w_i = max(P_i − P_thr_mol, 0)   (active-molecule weight)
    and    K   = Gaussian kernel, σ ~ 2–3 Å

  Step 2 – Iterative background threshold
    ρ_0 = ρ_max
    repeat:  ρ_0 ← mean of all grid values ≤ ρ_0
    until convergence → ρ_th

  Step 3 – Connected-component labelling
    mask = ρ_P(r) ≥ ρ_th
    clusters = scipy.ndimage.label(mask)

  Step 4 – Optional finer-grid boundary refinement
    re-run on grid with half the spacing inside each cluster's bounding box

  Step 5 – Characterise each cluster by initial ζ_cg

  Step 6 – Track clusters across frames
    Jaccard overlap + Hungarian assignment (same scheme as ZetaClusterAnalysis)

  Step 7 – 2D slice visualisation (XY, XZ, or YZ)

Separation of roles (from the document)
-----------------------------------------
  Dynamic propensity  →  defines which molecules are "active"
                          and therefore where cluster boundaries lie
  ζ_cg               →  characterises the initial structural identity
                          of each cluster
  Stress (optional)  →  traces relaxation per cluster domain

Usage
-----
::

    from water.tools.propensity_field_clusters import PropensityFieldClusters as PFC

    # pfc works on one reference frame at a time.
    # propensity shape: (N_ow,)   – one scalar per OW molecule
    # zeta_cg shape:   (N_ow,)   – from zop.results.zeta[frame_row]
    # positions shape: (N_ow, 3) – OW positions at the reference frame

    pfc = PFC(
        positions=pos_ref,
        propensity=prop_arr,
        zeta_cg=zeta_arr,
        box=box_lengths,          # shape (3,) Å
        sigma=3.0,                # Å  – Gaussian kernel width
        grid_spacing=1.0,         # Å  – coarse grid
        propensity_top_frac=0.15, # top 15 % defines active molecules
    )
    pfc.run()

    # Access results
    pfc.results.cluster_labels        # integer label per grid voxel
    pfc.results.cluster_stats         # list of ClusterStats
    pfc.results.rho_field             # smoothed density array
    pfc.results.rho_threshold         # converged ρ_th

    pfc.plot_slice(axis=2, ax=None)   # 2D projection coloured by ζ_cg
"""

import warnings
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy.ndimage import label as nd_label, gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.fft import rfftn, irfftn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cluster statistics container
# ---------------------------------------------------------------------------

@dataclass
class ClusterStats:
    """Statistics for one connected cluster region at one frame."""
    cluster_id:      int
    n_molecules:     int            # OW atoms inside the cluster region
    mean_zeta:       float          # ⟨ζ_cg⟩ of members
    std_zeta:        float
    frac_tetrahedral: float         # fraction with ζ_cg > 0
    mean_propensity: float          # ⟨P_i⟩ of members
    centroid:        np.ndarray     # shape (3,) Å
    rg:              float          # radius of gyration Å
    volume_vox:      int            # number of grid voxels above threshold
    member_indices:  np.ndarray     # column indices into the OW atom array

    # Generic field metadata
    field_kind:      str = "propensity"
    mean_field_weight: float = np.nan

    # Tracking fields (filled by PropensityFieldTracker)
    global_id:       int = -1
    birth_frame:     int = -1
    death_frame:     int = -1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PropensityFieldClusters:
    """Build density-field clusters for one reference frame.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        OW positions in Å.
    propensity : np.ndarray, shape (N,)
        Dynamic propensity P_i for each OW (e.g. isoconfigurational MSD).
    zeta_cg : np.ndarray, shape (N,)
        ζ_cg structural order parameter from ZetaOrderParameter.
        NaN entries are allowed; they are excluded from structural statistics.
    box : array-like, shape (3,) or (6,)
        Box lengths [lx, ly, lz] in Å (orthorhombic).
    sigma : float
        Gaussian kernel width in Å. Default 3.0 Å.
    grid_spacing : float
        Coarse grid voxel size in Å. Default 1.0 Å.
    propensity_top_frac : float
        Fraction of molecules with the highest propensity used as
        active molecules (w_i > 0).  Default 0.15 (top 15 %).
        Alternatively, set propensity_threshold directly.
    propensity_threshold : float or None
        Explicit P_thr_mol.  If set, overrides propensity_top_frac.
    rho_threshold_mode : str
        ``"iterative"`` (default) – iterative background convergence.
        ``"otsu"``      – Otsu's method on the grid histogram.
        ``"percentile"`` – use rho_percentile of the field as threshold.
    rho_percentile : float
        Used when rho_threshold_mode="percentile". Default 90.
    min_cluster_voxels : int
        Clusters smaller than this are discarded. Default 4.
    refine_grid : bool
        If True, refine the boundary of each cluster on a half-spacing grid.
        Default False (expensive for large boxes).
    """

    def __init__(
        self,
        positions:             np.ndarray,
        propensity:            np.ndarray,
        zeta_cg:               np.ndarray,
        box:                   np.ndarray,
        sigma:                 float = 3.0,
        grid_spacing:          float = 1.0,
        propensity_top_frac:   float = 0.15,
        propensity_threshold:  Optional[float] = None,
        rho_threshold_mode:    str   = "iterative",
        rho_percentile:        float = 90.0,
        min_cluster_voxels:    int   = 4,
        refine_grid:           bool  = False,
        exp_kernel_length:     float = 3.0,   # Å; separate from sigma for optional cutoff
        exp_kernel_cutoff:     Optional[float] = None,  # Å; if set, K(r > cutoff) = 0

        # New: choose which scalar defines the density field.
        field_kind:            str   = "propensity",  # "propensity" or "zeta_cg"

        # New: ζ_cg-field clustering controls. No extra convolution is applied.
        zeta_top_frac:         float = 0.15,
        zeta_threshold:        Optional[float] = None,
        zeta_cluster_mode:     str   = "high",       # "high", "low", or "abs"
        zeta_weight_mode:      str   = "excess",     # "excess", "binary", or "raw"
        zeta_deposit:          str   = "ngp",        # "ngp" or "cic"; no convolution

        # Thresholding / connectivity controls.
        threshold_positive_only: Optional[bool] = None,
        periodic_labels:       bool  = False,
    ):
        self.positions            = np.asarray(positions,   dtype=float)
        self.propensity           = np.asarray(propensity,  dtype=float)
        self.zeta_cg              = np.asarray(zeta_cg,     dtype=float)
        self.box                  = np.asarray(box,         dtype=float)[:3]
        self.sigma                = sigma
        self.grid_spacing         = grid_spacing
        self.propensity_top_frac  = propensity_top_frac
        self.propensity_threshold = propensity_threshold
        self.rho_threshold_mode   = rho_threshold_mode
        self.rho_percentile       = rho_percentile
        self.min_cluster_voxels   = min_cluster_voxels
        self.refine_grid          = refine_grid
        self.exp_kernel_length    = exp_kernel_length
        self.exp_kernel_cutoff    = exp_kernel_cutoff

        self.field_kind           = field_kind.lower()
        self.zeta_top_frac        = zeta_top_frac
        self.zeta_threshold       = zeta_threshold
        self.zeta_cluster_mode    = zeta_cluster_mode.lower()
        self.zeta_weight_mode     = zeta_weight_mode.lower()
        self.zeta_deposit         = zeta_deposit.lower()
        self.threshold_positive_only = threshold_positive_only
        self.periodic_labels      = periodic_labels

        valid_field_kinds = {"propensity", "dynamic_propensity", "zeta_cg", "zeta"}
        if self.field_kind not in valid_field_kinds:
            raise ValueError(
                f"Unknown field_kind '{field_kind}'. Choose 'propensity' or 'zeta_cg'."
            )
        if self.zeta_cluster_mode not in {"high", "low", "abs"}:
            raise ValueError("zeta_cluster_mode must be 'high', 'low', or 'abs'.")
        if self.zeta_weight_mode not in {"excess", "binary", "raw"}:
            raise ValueError("zeta_weight_mode must be 'excess', 'binary', or 'raw'.")
        if self.zeta_deposit not in {"ngp", "cic"}:
            raise ValueError("zeta_deposit must be 'ngp' or 'cic'.")

        # Grid dimensions
        self._nx, self._ny, self._nz = [
            max(1, int(np.ceil(self.box[d] / self.grid_spacing)))
            for d in range(3)
        ]
        # Actual voxel sizes (may differ slightly from grid_spacing)
        self._hx = self.box[0] / self._nx
        self._hy = self.box[1] / self._ny
        self._hz = self.box[2] / self._nz

        # Results namespace (plain object)
        class _R: pass
        self.results = _R()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> "PropensityFieldClusters":
        """Execute all five steps and populate self.results."""
        self._step1_build_density_field()
        self._step2_compute_threshold()
        self._step3_label_clusters()
        if self.refine_grid:
            if self.field_kind in {"zeta_cg", "zeta"}:
                warnings.warn(
                    "refine_grid is disabled for field_kind='zeta_cg' because "
                    "the requested ζ_cg field uses direct deposition without extra smoothing."
                )
            else:
                self._step4_refine_boundaries()
        self._step5_characterise_clusters()
        return self

    #-------------------------------------------------------------------
    # Helper: FFT convolution periodic smoothing
    #-------------------------------------------------------------------
    def _build_periodic_exp_kernel(self, L, cutoff=None, normalize=True):
        """
        Build an isotropic periodic exponential kernel:

            K(r) = exp(-r / L)

        where r is the minimum-image distance on the 3D periodic grid.

        Parameters
        ----------
        L : float
            Exponential decay length, in the same length unit as positions.
        cutoff : float or None
            Optional cutoff radius. If given, K(r > cutoff) = 0.
        normalize : bool
            If True, normalize the discrete kernel so that sum(K) = 1.

        Returns
        -------
        K : ndarray, shape (nx, ny, nz)
            Periodic convolution kernel.
        """

        ix = np.arange(self._nx)
        iy = np.arange(self._ny)
        iz = np.arange(self._nz)

        # Minimum-image distances along each grid direction
        dx = np.minimum(ix, self._nx - ix) * self._hx
        dy = np.minimum(iy, self._ny - iy) * self._hy
        dz = np.minimum(iz, self._nz - iz) * self._hz

        r = np.sqrt(
            dx[:, None, None] ** 2
            + dy[None, :, None] ** 2
            + dz[None, None, :] ** 2
        )

        K = np.exp(-r / L)

        if cutoff is not None:
            K[r > cutoff] = 0.0

        if normalize:
            s = K.sum()
            if s > 0:
                K /= s

        return K
    

    # ------------------------------------------------------------------
    # Step 1: build scalar field
    # ------------------------------------------------------------------

    def _step1_build_density_field(self):
        """Build the grid field used to define connected clusters.

        field_kind='propensity' keeps the original dynamic-propensity pipeline:
        threshold molecular propensity, deposit active weights onto a grid,
        then apply the existing periodic exponential convolution.

        field_kind='zeta_cg' builds a ζ_cg-weighted grid by direct deposition
        only. No Gaussian/exponential convolution is applied, because ζ_cg is
        already spatially coarse-grained at the molecular level.
        """
        if self.field_kind in {"propensity", "dynamic_propensity"}:
            self._build_propensity_density_field()
        elif self.field_kind in {"zeta_cg", "zeta"}:
            self._build_zeta_cg_density_field()
        else:
            raise RuntimeError(f"Unsupported field_kind={self.field_kind!r}")

        self.results.grid_shape = self.results.rho_field.shape
        self.results.grid_spacing_actual = np.array([self._hx, self._hy, self._hz])
        self.results.field_kind = self.field_kind

    def _grid_indices(self, positions: Optional[np.ndarray] = None):
        """Return nearest-grid-point indices for particle positions."""
        if positions is None:
            positions = self.positions
        pos = np.asarray(positions, dtype=float)
        x = np.mod(pos[:, 0], self.box[0])
        y = np.mod(pos[:, 1], self.box[1])
        z = np.mod(pos[:, 2], self.box[2])
        ix = np.clip((x / self._hx).astype(int), 0, self._nx - 1)
        iy = np.clip((y / self._hy).astype(int), 0, self._ny - 1)
        iz = np.clip((z / self._hz).astype(int), 0, self._nz - 1)
        return ix, iy, iz

    def _deposit_ngp(self, weights: np.ndarray) -> np.ndarray:
        """Nearest-grid-point deposition. This is not a smoothing kernel."""
        field = np.zeros((self._nx, self._ny, self._nz), dtype=float)
        ix, iy, iz = self._grid_indices()
        valid = np.isfinite(weights) & (weights != 0.0)
        np.add.at(field, (ix[valid], iy[valid], iz[valid]), weights[valid])
        return field

    def _deposit_cic(self, weights: np.ndarray) -> np.ndarray:
        """Cloud-in-cell deposition with periodic wrapping.

        This is mass-conserving interpolation onto the grid, not a physical
        coarse-graining/convolution. Use it only if NGP makes the direct ζ_cg
        field too voxelized.
        """
        field = np.zeros((self._nx, self._ny, self._nz), dtype=float)
        valid = np.isfinite(weights) & (weights != 0.0)
        if not np.any(valid):
            return field

        pos = self.positions[valid]
        w = weights[valid]
        gx = np.mod(pos[:, 0], self.box[0]) / self._hx
        gy = np.mod(pos[:, 1], self.box[1]) / self._hy
        gz = np.mod(pos[:, 2], self.box[2]) / self._hz

        i0 = np.floor(gx).astype(int) % self._nx
        j0 = np.floor(gy).astype(int) % self._ny
        k0 = np.floor(gz).astype(int) % self._nz
        fx = gx - np.floor(gx)
        fy = gy - np.floor(gy)
        fz = gz - np.floor(gz)

        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            ii = (i0 + dx) % self._nx
            for dy, wy in ((0, 1.0 - fy), (1, fy)):
                jj = (j0 + dy) % self._ny
                for dz, wz in ((0, 1.0 - fz), (1, fz)):
                    kk = (k0 + dz) % self._nz
                    np.add.at(field, (ii, jj, kk), w * wx * wy * wz)
        return field

    def _build_propensity_weights(self) -> Tuple[np.ndarray, float]:
        """Original active-propensity molecular weights."""
        if self.propensity_threshold is not None:
            p_thr = float(self.propensity_threshold)
        else:
            p_thr = float(np.nanpercentile(
                self.propensity, 100.0 * (1.0 - self.propensity_top_frac)
            ))
        weights = np.maximum(self.propensity - p_thr, 0.0).astype(float)
        weights[~np.isfinite(weights)] = 0.0
        return weights, p_thr

    def _build_propensity_density_field(self):
        """Original propensity-density field with periodic exponential smoothing."""
        weights, p_thr = self._build_propensity_weights()
        self.results.propensity_mol_threshold = p_thr
        self.results.field_mol_threshold = p_thr
        self.results.n_active = int((weights > 0).sum())
        self.results.molecular_field_weight = weights

        rho0 = self._deposit_ngp(weights)
        kernel = self._build_periodic_exp_kernel(
            L=self.exp_kernel_length,
            cutoff=getattr(self, "exp_kernel_cutoff", None),
            normalize=True,
        )
        rho = irfftn(rfftn(rho0) * rfftn(kernel), s=rho0.shape).real
        self.results.rho_field = rho
        self.results.raw_deposited_field = rho0
        self.results.field_label = "ρ_P"

    def _build_zeta_weights(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """Convert ζ_cg values into non-negative molecular weights.

        high: high-ζ_cg / LDL-like / tetrahedral-rich regions.
        low:  low-ζ_cg / HDL-like / disordered regions.
        abs:  large-|ζ_cg| regions irrespective of sign.
        """
        z = self.zeta_cg.astype(float)
        finite = np.isfinite(z)
        if not np.any(finite):
            raise ValueError("zeta_cg contains no finite values.")

        if self.zeta_threshold is not None:
            z_thr = float(self.zeta_threshold)
        else:
            if self.zeta_cluster_mode == "high":
                z_thr = float(np.nanpercentile(z, 100.0 * (1.0 - self.zeta_top_frac)))
            elif self.zeta_cluster_mode == "low":
                z_thr = float(np.nanpercentile(z, 100.0 * self.zeta_top_frac))
            else:
                z_thr = float(np.nanpercentile(
                    np.abs(z[finite]), 100.0 * (1.0 - self.zeta_top_frac)
                ))

        weights = np.zeros_like(z, dtype=float)
        selected = np.zeros_like(z, dtype=bool)

        if self.zeta_cluster_mode == "high":
            selected = finite & (z >= z_thr)
            if self.zeta_weight_mode == "excess":
                weights[selected] = z[selected] - z_thr
            elif self.zeta_weight_mode == "binary":
                weights[selected] = 1.0
            else:
                weights[selected] = np.maximum(z[selected], 0.0)

        elif self.zeta_cluster_mode == "low":
            selected = finite & (z <= z_thr)
            if self.zeta_weight_mode == "excess":
                weights[selected] = z_thr - z[selected]
            elif self.zeta_weight_mode == "binary":
                weights[selected] = 1.0
            else:
                weights[selected] = np.maximum(-z[selected], 0.0)

        else:
            az = np.abs(z)
            selected = finite & (az >= z_thr)
            if self.zeta_weight_mode == "excess":
                weights[selected] = az[selected] - z_thr
            elif self.zeta_weight_mode == "binary":
                weights[selected] = 1.0
            else:
                weights[selected] = az[selected]

        return weights, z_thr, selected

    def _build_zeta_cg_density_field(self):
        """Build a direct ζ_cg-weighted field without convolution."""
        weights, z_thr, selected = self._build_zeta_weights()

        if self.zeta_deposit == "ngp":
            rho = self._deposit_ngp(weights)
            count = self._deposit_ngp(selected.astype(float))
            zeta_sum = self._deposit_ngp(np.where(selected, self.zeta_cg, 0.0))
        elif self.zeta_deposit == "cic":
            rho = self._deposit_cic(weights)
            count = self._deposit_cic(selected.astype(float))
            zeta_sum = self._deposit_cic(np.where(selected, self.zeta_cg, 0.0))
        else:
            raise RuntimeError(f"Unsupported zeta_deposit={self.zeta_deposit!r}")

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_zeta_grid = zeta_sum / count
        mean_zeta_grid[count <= 0] = np.nan

        self.results.zeta_mol_threshold = z_thr
        self.results.field_mol_threshold = z_thr
        self.results.n_active = int(selected.sum())
        self.results.molecular_field_weight = weights
        self.results.selected_molecule_mask = selected
        self.results.rho_field = rho
        self.results.raw_deposited_field = rho.copy()
        self.results.count_field = count
        self.results.mean_zeta_grid = mean_zeta_grid
        self.results.field_label = f"ρ_ζ({self.zeta_cluster_mode})"
    # ------------------------------------------------------------------
    # Step 2: iterative background threshold
    # ------------------------------------------------------------------

    def _step2_compute_threshold(self):
        """Compute a threshold for the grid field.

        For direct ζ_cg deposition, the grid can be sparse. By default, the
        threshold is computed only over positive/nonzero voxels, avoiding the
        trivial ρ_th≈0 threshold caused by empty grid cells.
        """
        rho = self.results.rho_field
        flat_all = rho.ravel()

        if self.threshold_positive_only is None:
            positive_only = self.field_kind in {"zeta_cg", "zeta"}
        else:
            positive_only = bool(self.threshold_positive_only)

        if positive_only and np.any(flat_all > 0):
            flat = flat_all[flat_all > 0]
        else:
            flat = flat_all

        if flat.size == 0:
            raise ValueError("Cannot determine rho_threshold from an empty field.")

        if self.rho_threshold_mode == "iterative":
            rho_0 = float(flat.max())
            tol = 1.0e-3
            for _ in range(500):
                background = flat[flat <= rho_0]
                if len(background) == 0:
                    break
                rho_new = float(background.mean())
                denom = max(abs(rho_new), np.finfo(float).eps)
                if abs(rho_new - rho_0) / denom < tol:
                    rho_0 = rho_new
                    break
                rho_0 = rho_new
            rho_th = rho_0
        elif self.rho_threshold_mode == "otsu":
            rho_th = _otsu_threshold(flat)
        elif self.rho_threshold_mode == "percentile":
            rho_th = float(np.percentile(flat, self.rho_percentile))
        else:
            raise ValueError(
                f"Unknown rho_threshold_mode '{self.rho_threshold_mode}'. "
                "Choose 'iterative', 'otsu', or 'percentile'."
            )

        self.results.rho_threshold = rho_th
        self.results.threshold_positive_only = positive_only
        logger.debug("ρ_th = %.6f  (mode=%s)", rho_th, self.rho_threshold_mode)

    # ------------------------------------------------------------------
    # Step 3: connected-component labelling
    # ------------------------------------------------------------------

    def _step3_label_clusters(self):
        """Label connected regions where ρ ≥ ρ_th."""
        rho = self.results.rho_field
        rho_th = self.results.rho_threshold
        mask = rho > rho_th if rho_th <= 0 else rho >= rho_th

        struct = np.ones((3, 3, 3), dtype=int)
        labeled, n_raw = nd_label(mask, structure=struct)
        if self.periodic_labels and n_raw > 1:
            labeled, n_raw = _merge_periodic_labels(labeled)

        cleaned = np.zeros_like(labeled)
        new_id = 1
        for cid in range(1, n_raw + 1):
            vox_count = int((labeled == cid).sum())
            if vox_count >= self.min_cluster_voxels:
                cleaned[labeled == cid] = new_id
                new_id += 1

        self.results.cluster_labels = cleaned
        self.results.n_clusters_raw = n_raw
        self.results.n_clusters = new_id - 1
        self.results.cluster_mask = cleaned > 0

    # ------------------------------------------------------------------
    # Step 4 (optional): finer-grid boundary refinement
    # ------------------------------------------------------------------

    def _step4_refine_boundaries(self):
        """
        Re-run the Gaussian KDE at half the grid spacing inside each
        cluster's bounding box to sharpen boundaries.
        Updates self.results.cluster_labels in place.
        """
        if self.field_kind in {"zeta_cg", "zeta"}:
            return

        labels   = self.results.cluster_labels
        n_cl     = self.results.n_clusters
        h_fine   = self.grid_spacing / 2.0
        refined  = labels.copy()

        for cid in range(1, n_cl + 1):
            vox = np.argwhere(labels == cid)
            if len(vox) == 0:
                continue
            # Bounding box (coarse voxels)
            lo = vox.min(axis=0)
            hi = vox.max(axis=0) + 1

            # Convert bounding box to Å, add one-σ margin
            margin_vox = max(1, int(np.ceil(self.sigma / self.grid_spacing)))
            lo_c = np.maximum(lo - margin_vox, 0)
            hi_c = np.minimum(hi + margin_vox,
                              np.array([self._nx, self._ny, self._nz]))

            lo_ang = lo_c * np.array([self._hx, self._hy, self._hz])
            hi_ang = hi_c * np.array([self._hx, self._hy, self._hz])

            # Atoms inside the bounding box
            in_box = np.all(
                (self.positions >= lo_ang) & (self.positions <= hi_ang),
                axis=1,
            )
            if in_box.sum() == 0:
                continue

            pos_sub  = self.positions[in_box]
            p_thr    = self.results.propensity_mol_threshold
            w_sub    = np.maximum(self.propensity[in_box] - p_thr, 0.0)

            # Fine grid inside bounding box
            box_sub  = hi_ang - lo_ang
            nx_f = max(1, int(np.ceil(box_sub[0] / h_fine)))
            ny_f = max(1, int(np.ceil(box_sub[1] / h_fine)))
            nz_f = max(1, int(np.ceil(box_sub[2] / h_fine)))
            hx_f = box_sub[0] / nx_f
            hy_f = box_sub[1] / ny_f
            hz_f = box_sub[2] / nz_f

            rho_f = np.zeros((nx_f, ny_f, nz_f), dtype=float)
            pos_rel = pos_sub - lo_ang
            ixf = np.clip((pos_rel[:, 0] / hx_f).astype(int), 0, nx_f - 1)
            iyf = np.clip((pos_rel[:, 1] / hy_f).astype(int), 0, ny_f - 1)
            izf = np.clip((pos_rel[:, 2] / hz_f).astype(int), 0, nz_f - 1)
            np.add.at(rho_f, (ixf, iyf, izf), w_sub)
            rho_f = gaussian_filter(
                rho_f, sigma=self.sigma / h_fine, mode="nearest"
            )

            # Apply the same ρ_th to the fine grid
            mask_f = rho_f >= self.results.rho_threshold
            # Map fine-grid mask back to coarse voxels
            for fi in np.ndindex(nx_f, ny_f, nz_f):
                if mask_f[fi]:
                    # Coarse-grid voxel containing this fine voxel
                    cx = lo_c[0] + int(fi[0] * h_fine / self._hx)
                    cy = lo_c[1] + int(fi[1] * h_fine / self._hy)
                    cz = lo_c[2] + int(fi[2] * h_fine / self._hz)
                    if (0 <= cx < self._nx and
                            0 <= cy < self._ny and
                            0 <= cz < self._nz):
                        refined[cx, cy, cz] = cid

        self.results.cluster_labels = refined

    # ------------------------------------------------------------------
    # Step 5: characterise clusters by ζ_cg
    # ------------------------------------------------------------------

    def _step5_characterise_clusters(self):
        """
        For each connected cluster region, find which OW atoms fall inside it
        and compute structural/dynamic statistics.
        """
        labels = self.results.cluster_labels
        n_cl = self.results.n_clusters

        ix, iy, iz = self._grid_indices()
        atom_cluster_id = labels[ix, iy, iz]
        mol_field_weight = getattr(
            self.results, "molecular_field_weight", np.zeros(len(self.positions))
        )

        stats_list: List[ClusterStats] = []
        for cid in range(1, n_cl + 1):
            vox_count = int((labels == cid).sum())
            member_idx = np.where(atom_cluster_id == cid)[0]
            if len(member_idx) == 0:
                continue

            pos_m = self.positions[member_idx]
            zeta_m = self.zeta_cg[member_idx]
            prop_m = self.propensity[member_idx]
            weight_m = mol_field_weight[member_idx]

            valid_z = zeta_m[np.isfinite(zeta_m)]
            centroid = pos_m.mean(axis=0)
            delta = pos_m - centroid
            rg = float(np.sqrt((delta ** 2).sum(axis=1).mean()))

            stats = ClusterStats(
                cluster_id=cid,
                n_molecules=len(member_idx),
                mean_zeta=float(np.nanmean(zeta_m)),
                std_zeta=float(np.nanstd(zeta_m)),
                frac_tetrahedral=float((valid_z > 0).mean()) if len(valid_z) else np.nan,
                mean_propensity=float(np.nanmean(prop_m)),
                centroid=centroid,
                rg=rg,
                volume_vox=vox_count,
                member_indices=member_idx,
                field_kind=self.field_kind,
                mean_field_weight=float(np.nanmean(weight_m)),
            )
            stats_list.append(stats)

        # Sort by size and remap labels so stats.cluster_id remains consistent
        # with results.cluster_labels and results.atom_cluster_id.
        stats_list.sort(key=lambda s: s.n_molecules, reverse=True)
        old_to_new = {s.cluster_id: rank + 1 for rank, s in enumerate(stats_list)}
        if old_to_new:
            remapped_labels = np.zeros_like(labels)
            for old_id, new_id in old_to_new.items():
                remapped_labels[labels == old_id] = new_id
            labels = remapped_labels
            atom_cluster_id = labels[ix, iy, iz]
            for s in stats_list:
                s.cluster_id = old_to_new[s.cluster_id]

        self.results.cluster_labels = labels
        self.results.cluster_stats = stats_list
        self.results.atom_cluster_id = atom_cluster_id
        self.results.n_clusters = len(stats_list)
        self.results.cluster_mask = labels > 0

    # ------------------------------------------------------------------
    # Visualisation: 2D projection slice
    # ------------------------------------------------------------------

    def plot_slice(
        self,
        axis:          int   = 2,
        slice_centre:  Optional[float] = None,
        slice_width:   float = 5.0,
        ax=None,
        show_boundaries: bool = True,
        show_propensity: bool = False,
        title:         str   = "",
    ):
        """Plot a 2D slab coloured by ζ_cg with cluster boundaries overlaid.

        Parameters
        ----------
        axis : 0/1/2  → x/y/z as slab normal
        slice_centre : float or None  → Å; None = box centre
        slice_width  : float          → full slab thickness in Å
        ax           : matplotlib Axes or None
        show_boundaries : bool  → draw cluster contours
        show_propensity : bool  → colour by propensity instead of ζ_cg
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch

        ax_x, ax_y = [i for i in range(3) if i != axis]
        centre = self.box[axis] / 2.0 if slice_centre is None else slice_centre

        # Slab mask (PBC-corrected along slab axis)
        z_pos = self.positions[:, axis]
        dz    = z_pos - centre
        dz   -= self.box[axis] * np.round(dz / self.box[axis])
        slab  = np.abs(dz) <= slice_width / 2.0

        pos_s    = self.positions[slab]
        zeta_s   = self.zeta_cg[slab]
        prop_s   = self.propensity[slab]
        cid_s    = self.results.atom_cluster_id[slab]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        # ---- Scatter: colour = ζ_cg or propensity -----------------------
        if show_propensity:
            vmin, vmax = prop_s.min(), prop_s.max()
            sc = ax.scatter(
                pos_s[:, ax_x], pos_s[:, ax_y],
                c=prop_s, cmap="hot_r",
                vmin=vmin, vmax=vmax,
                s=12, linewidths=0, alpha=0.85, zorder=2,
            )
            plt.colorbar(sc, ax=ax, label="Propensity P_i", shrink=0.8)
        else:
            vmin, vmax = np.nanpercentile(zeta_s, [2, 98])
            sc = ax.scatter(
                pos_s[:, ax_x], pos_s[:, ax_y],
                c=zeta_s, cmap="RdBu",
                vmin=vmin, vmax=vmax,
                s=12, linewidths=0, alpha=0.85, zorder=2,
            )
            plt.colorbar(sc, ax=ax, label="ζ_cg (Å)", shrink=0.8)

        # ---- Cluster boundaries (convex hull of members in slab) ---------
        if show_boundaries:
            from scipy.spatial import ConvexHull
            import itertools
            cmap_cl = plt.get_cmap("tab10")
            for stats in self.results.cluster_stats[:10]:   # top 10
                mem_slab = np.intersect1d(stats.member_indices,
                                          np.where(slab)[0])
                if len(mem_slab) < 3:
                    continue
                pts = self.positions[mem_slab][:, [ax_x, ax_y]]
                try:
                    hull  = ConvexHull(pts)
                    verts = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(pts[verts, 0], pts[verts, 1],
                            color=cmap_cl(stats.cluster_id % 10),
                            lw=1.5, alpha=0.8, zorder=3)
                    cx, cy = pts.mean(axis=0)
                    ax.text(cx, cy,
                            f"#{stats.cluster_id}\n"
                            f"ζ={stats.mean_zeta:+.2f}\n"
                            f"N={stats.n_molecules}",
                            fontsize=6, ha="center", va="center",
                            color=cmap_cl(stats.cluster_id % 10),
                            zorder=4)
                except Exception:
                    pass

        axis_labels = ["x", "y", "z"]
        ax.set_xlabel(f"{axis_labels[ax_x]} (Å)")
        ax.set_ylabel(f"{axis_labels[ax_y]} (Å)")
        ax.set_title(title or
                     f"{getattr(self.results, 'field_label', 'Field')} clusters  |  "
                     f"slab {axis_labels[axis]} ± {slice_width/2:.1f} Å  |  "
                     f"N_cl = {self.results.n_clusters}")
        ax.set_aspect("equal")
        return ax

    # ------------------------------------------------------------------
    # 2D density-field overlay
    # ------------------------------------------------------------------


    def plot_density_slice(
        self,
        axis: int = 2,
        slice_centre: Optional[float] = None,
        slice_width: float = 10.0,
        reduce: str = "max",
        ax=None,
        cmap: str = "plasma",
        show_threshold: bool = True,
        title: str = "",
    ):
        """Show a 2D slab slice of the density field.

        Unlike the old version, this function does not project through the
        entire simulation box. It only uses grid voxels whose centres lie inside
        a finite slab along the chosen normal direction:

            slice_centre - slice_width/2 <= r_axis <= slice_centre + slice_width/2

        with periodic wrapping along the slab-normal axis. By default,
        ``slice_centre`` is the box centre and ``slice_width=10.0`` Å, i.e. the
        displayed slab is box centre ± 5 Å.

        Parameters
        ----------
        axis : int, default 2
            Slab-normal axis. ``0``/``1``/``2`` correspond to x/y/z.
            For example, ``axis=2`` shows an XY slab around the box centre in z.
        slice_centre : float or None, default None
            Centre of the slab in Å. If None, use ``box[axis] / 2``.
        slice_width : float, default 10.0
            Full slab thickness in Å. The default corresponds to ±5 Å.
        reduce : {"max", "mean", "sum"}, default "max"
            How to reduce the selected 3D slab to a 2D field.
            ``"max"`` best preserves high-density cluster cores;
            ``"mean"`` gives the average density inside the slab;
            ``"sum"`` gives the integrated slab density.
        ax : matplotlib Axes or None
            Axis to draw on. If None, a new figure and axis are created.
        cmap : str, default "plasma"
            Matplotlib colormap for the density field.
        show_threshold : bool, default True
            If True, draw the rho_threshold contour when the threshold lies
            inside the displayed 2D field range.
        title : str, default ""
            Optional custom title.

        Returns
        -------
        ax : matplotlib Axes
            The axis containing the plot.
        """
        import matplotlib.pyplot as plt

        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        if slice_width <= 0:
            raise ValueError("slice_width must be positive")

        rho = self.results.rho_field
        rho_th = self.results.rho_threshold
        h = self.results.grid_spacing_actual
        ax_x, ax_y = [i for i in range(3) if i != axis]
        axis_labels = ["x", "y", "z"]

        centre = self.box[axis] / 2.0 if slice_centre is None else float(slice_centre)

        # Grid-cell centres along the slab-normal axis.
        n_axis = rho.shape[axis]
        centres_axis = (np.arange(n_axis) + 0.5) * h[axis]

        # Periodic distance from the requested slice centre.
        d = centres_axis - centre
        d -= self.box[axis] * np.round(d / self.box[axis])
        slab_idx = np.where(np.abs(d) <= slice_width / 2.0)[0]

        # Very thin slices may miss all voxel centres. Fall back to the nearest
        # grid layer instead of returning an empty plot.
        if slab_idx.size == 0:
            slab_idx = np.array([int(np.argmin(np.abs(d)))])

        rho_slab = np.take(rho, slab_idx, axis=axis)

        if reduce == "max":
            rho_2d = np.max(rho_slab, axis=axis)
            cbar_label = r"$\rho$ (slab max)"
        elif reduce == "mean":
            rho_2d = np.mean(rho_slab, axis=axis)
            cbar_label = r"$\rho$ (slab mean)"
        elif reduce == "sum":
            rho_2d = np.sum(rho_slab, axis=axis)
            cbar_label = r"$\rho$ (slab sum)"
        else:
            raise ValueError("reduce must be 'max', 'mean', or 'sum'")

        x_edges = np.arange(rho_2d.shape[0] + 1) * h[ax_x]
        y_edges = np.arange(rho_2d.shape[1] + 1) * h[ax_y]
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        pcm = ax.pcolormesh(
            x_edges,
            y_edges,
            rho_2d.T,
            cmap=cmap,
            shading="flat",
        )

        if show_threshold:
            rho_min = float(np.nanmin(rho_2d))
            rho_max = float(np.nanmax(rho_2d))
            if rho_min <= rho_th <= rho_max:
                ax.contour(
                    x_centres,
                    y_centres,
                    rho_2d.T,
                    levels=[rho_th],
                    colors="white",
                    linewidths=1.2,
                )

        plt.colorbar(pcm, ax=ax, label=cbar_label)

        slab_lo = centre - slice_width / 2.0
        slab_hi = centre + slice_width / 2.0
        ax.set(
            xlabel=f"{axis_labels[ax_x]} (Å)",
            ylabel=f"{axis_labels[ax_y]} (Å)",
            title=(
                title
                or f"Density field slab | "
                   f"{axis_labels[axis]} = {centre:.2f} ± {slice_width/2.0:.2f} Å | "
                   f"{reduce}, ρ_th = {rho_th:.4f}"
            ),
        )
        ax.set_aspect("equal")
        return ax


# ---------------------------------------------------------------------------
# Cluster tracker across frames
# ---------------------------------------------------------------------------

class PropensityFieldTracker:
    """Track density-field clusters across multiple frames.

    Each frame's PropensityFieldClusters object is passed in order.
    Jaccard-overlap Hungarian assignment propagates global IDs.

    Parameters
    ----------
    min_overlap : float
        Minimum Jaccard index to link clusters across consecutive frames.
    min_lifetime : int
        Clusters surviving fewer frames are filtered out of the final tracks.

    Usage
    -----
    ::

        tracker = PropensityFieldTracker(min_overlap=0.3, min_lifetime=3)
        for fi, pfc_frame in enumerate(pfc_list):
            tracker.update(pfc_frame, frame_time=fi * dt)
        tracks = tracker.finalise()
    """

    def __init__(self, min_overlap: float = 0.3, min_lifetime: int = 3):
        self.min_overlap  = min_overlap
        self.min_lifetime = min_lifetime

        self._next_gid   = 0
        self._active_map: Dict[int, int] = {}   # local_cluster_id → global_id
        self._tracks:     Dict[int, dict] = {}  # global_id → track dict
        self._prev_member_sets: Dict[int, frozenset] = {}

    def update(self, pfc: PropensityFieldClusters, frame_time: float = 0.0):
        """Ingest one frame's cluster results and extend/create tracks."""
        curr_stats  = pfc.results.cluster_stats
        curr_ids    = [s.cluster_id for s in curr_stats]
        curr_sets   = {
            s.cluster_id: frozenset(s.member_indices.tolist())
            for s in curr_stats
        }

        if not self._active_map:
            # First frame: assign fresh global IDs to all clusters
            for s in curr_stats:
                gid = self._next_gid
                self._next_gid += 1
                s.global_id = gid
                self._tracks[gid] = _new_track(gid, s, frame_time)
                self._active_map[s.cluster_id] = gid
        else:
            # Build Jaccard matrix: rows = prev, cols = curr
            prev_ids  = list(self._active_map.keys())
            if not prev_ids or not curr_ids:
                self._active_map = {}
                self._prev_member_sets = curr_sets
                return

            J = np.zeros((len(prev_ids), len(curr_ids)), dtype=float)
            for pi, pid in enumerate(prev_ids):
                A = self._prev_member_sets.get(pid, frozenset())
                for ci, cid in enumerate(curr_ids):
                    B = curr_sets.get(cid, frozenset())
                    inter = len(A & B)
                    union = len(A | B)
                    J[pi, ci] = inter / union if union > 0 else 0.0

            row_ind, col_ind = linear_sum_assignment(-J)
            matched_curr = set()
            new_map = {}

            for pi, ci in zip(row_ind, col_ind):
                if J[pi, ci] >= self.min_overlap:
                    pid = prev_ids[pi]
                    cid = curr_ids[ci]
                    gid = self._active_map[pid]
                    s   = curr_stats[ci]
                    s.global_id = gid
                    _track_append(self._tracks[gid], s, frame_time)
                    new_map[cid] = gid
                    matched_curr.add(ci)

            # Unmatched: new tracks
            for ci, s in enumerate(curr_stats):
                if ci not in matched_curr:
                    gid = self._next_gid
                    self._next_gid += 1
                    s.global_id = gid
                    self._tracks[gid] = _new_track(gid, s, frame_time)
                    new_map[s.cluster_id] = gid

            self._active_map = new_map

        self._prev_member_sets = curr_sets

    def finalise(self) -> Dict[int, dict]:
        """Return all tracks with lifetime ≥ min_lifetime."""
        return {
            gid: t for gid, t in self._tracks.items()
            if len(t["times"]) >= self.min_lifetime
        }


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _merge_periodic_labels(labeled: np.ndarray) -> Tuple[np.ndarray, int]:
    """Merge connected-component labels across periodic box faces."""
    shape = labeled.shape
    parent = {int(x): int(x) for x in np.unique(labeled) if x != 0}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        if a == 0 or b == 0 or a == b:
            return
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    if not parent:
        return labeled, 0

    for y in range(shape[1]):
        for z in range(shape[2]):
            a = labeled[0, y, z]
            if a == 0:
                continue
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    union(a, labeled[-1, (y + dy) % shape[1], (z + dz) % shape[2]])

    for x in range(shape[0]):
        for z in range(shape[2]):
            a = labeled[x, 0, z]
            if a == 0:
                continue
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    union(a, labeled[(x + dx) % shape[0], -1, (z + dz) % shape[2]])

    for x in range(shape[0]):
        for y in range(shape[1]):
            a = labeled[x, y, 0]
            if a == 0:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    union(a, labeled[(x + dx) % shape[0], (y + dy) % shape[1], -1])

    root_to_new = {}
    new_id = 1
    out = np.zeros_like(labeled)
    for old in np.unique(labeled):
        old = int(old)
        if old == 0:
            continue
        root = find(old)
        if root not in root_to_new:
            root_to_new[root] = new_id
            new_id += 1
        out[labeled == old] = root_to_new[root]
    return out, new_id - 1


def _otsu_threshold(flat: np.ndarray) -> float:
    """1-D Otsu's method on histogram of flat field values."""
    hist, edges = np.histogram(flat, bins=256)
    bin_centres  = 0.5 * (edges[:-1] + edges[1:])
    total        = hist.sum()
    if total == 0:
        return float(flat.mean())
    cumsum   = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centres)
    global_mean = cum_mean[-1] / total
    best_var, best_thr = 0.0, float(bin_centres[0])
    for k in range(1, len(hist)):
        w0 = cumsum[k] / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean[k] / cumsum[k]
        mu1 = (cum_mean[-1] - cum_mean[k]) / (total - cumsum[k])
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thr = float(bin_centres[k])
    return best_thr


def _new_track(gid: int, s: ClusterStats, t: float) -> dict:
    return {
        "global_id":    gid,
        "times":        [t],
        "n_molecules":  [s.n_molecules],
        "mean_zeta":    [s.mean_zeta],
        "std_zeta":     [s.std_zeta],
        "frac_tet":     [s.frac_tetrahedral],
        "mean_prop":    [s.mean_propensity],
        "mean_field_weight": [getattr(s, "mean_field_weight", np.nan)],
        "centroid":     [s.centroid.copy()],
        "rg":           [s.rg],
    }


def _track_append(track: dict, s: ClusterStats, t: float):
    track["times"].append(t)
    track["n_molecules"].append(s.n_molecules)
    track["mean_zeta"].append(s.mean_zeta)
    track["std_zeta"].append(s.std_zeta)
    track["frac_tet"].append(s.frac_tetrahedral)
    track["mean_prop"].append(s.mean_propensity)
    track.setdefault("mean_field_weight", []).append(
        getattr(s, "mean_field_weight", np.nan)
    )
    track["centroid"].append(s.centroid.copy())
    track["rg"].append(s.rg)


def finalise_track_arrays(track: dict) -> dict:
    """Convert lists → numpy arrays in a track dict."""
    out = {"global_id": track["global_id"]}
    for key in ("times", "n_molecules", "mean_zeta", "std_zeta",
                "frac_tet", "mean_prop", "mean_field_weight", "rg"):
        out[key] = np.array(track[key])
    out["centroid"] = np.array(track["centroid"])  # (T, 3)
    out["lifetime"] = len(track["times"])
    if len(out["centroid"]) >= 2:
        out["com_displacement"] = float(
            np.linalg.norm(out["centroid"][-1] - out["centroid"][0])
        )
    else:
        out["com_displacement"] = 0.0
    return out
