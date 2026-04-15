# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Zeta Local Order Parameter
===========================

Computes the zeta (ζ) order parameter for every central molecule at every
trajectory frame, using a completed HydrogenBondAnalysis run from
``custom_hbond_analysis.py`` as input.

Definition (Duboué-Dijon & Laage, J. Phys. Chem. B, 2015)
-----------------------------------------------------------

  ζᵢ(t) = d_nonHB_near(i,t) − d_HB_far(i,t)

where, for central molecule i at frame t:

  d_HB_far   = distance to the *farthest*  hydrogen-bond partner  (OW–OW)
  d_nonHB_near = distance to the *nearest* non-hydrogen-bonded OW
                 within a search shell of radius ``shell_cutoff``

A molecule is an "HB partner" of i if i appears as either the donor or the
acceptor in any bond recorded by HydrogenBondAnalysis.  The OW–OW D-A
distance is stored directly in column 4 of ``hba.results.hbonds``, so no
extra geometry calculation is needed for HB partners.

Physical interpretation
-----------------------
  ζ > 0  →  tetrahedral / ice-like local structure:
            all HB partners lie inside the first coordination shell;
            non-HB molecules are pushed further out.
  ζ < 0  →  distorted / interstitial structure:
            at least one non-HB molecule has intruded closer than the
            outermost HB partner, breaking the shell hierarchy.
  ζ = NaN → molecule has no HB partners *or* no non-HB neighbors within
            the search shell at this frame (excluded from statistics).

Output (``results`` namespace)
-------------------------------
  results.zeta        : float32 array, shape (n_frames, n_central)
  results.d_hb_far    : float32 array, shape (n_frames, n_central)
  results.d_nonhb_near: float32 array, shape (n_frames, n_central)
  results.n_hb        : int16   array, shape (n_frames, n_central)
                        number of HB partners per molecule per frame

Usage
-----
::

    import MDAnalysis as mda
    from custom_hbond_analysis import HydrogenBondAnalysis as HBA
    from zeta_order_parameter import ZetaOrderParameter as ZOP

    u   = mda.Universe("topology.tpr", "trajectory.xtc")

    # Step 1 – run H-bond analysis
    hba = HBA(universe=u,
              donors_sel="resname SOL and name OW",
              hydrogens_sel="resname SOL and name HW1 HW2",
              acceptors_sel="resname SOL and name OW",
              d_a_cutoff=3.5, h_d_a_angle_cutoff=30.0)
    hba.run()

    # Step 2 – compute zeta
    zop = ZOP(hba=hba,
              central_sel="resname SOL and name OW",
              shell_cutoff=4.5)
    zop.run()

    # Step 3 – analysis helpers
    print(zop.mean_zeta())          # scalar – grand mean
    zop.get_zeta_timeseries()       # shape (n_frames, n_central)
    zop.classify_structure()        # per-molecule tetrahedral fraction
"""

import warnings
import logging
from typing import Optional

import numpy as np

from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.lib.distances import capped_distance, self_capped_distance
from MDAnalysis.exceptions import NoDataError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZetaOrderParameter
# ---------------------------------------------------------------------------

class ZetaOrderParameter(AnalysisBase):
    """Compute the ζ local order parameter for each central molecule.

    Parameters
    ----------
    hba : HydrogenBondAnalysis
        A **completed** (``run()`` has been called) HydrogenBondAnalysis
        object from ``custom_hbond_analysis``.  Its ``results.hbonds`` array
        provides the HB partner indices and D-A distances without any
        re-computation.
    central_sel : str
        MDAnalysis selection string that identifies the representative atom
        of each "central molecule".  For water this is the oxygen atom,
        e.g. ``"resname SOL and name OW"``.
    shell_cutoff : float
        Search radius (Å) for finding *all* neighbors (HB and non-HB alike).
        Must be ≥ the ``d_a_cutoff`` used in ``hba`` (default 4.5 Å).
        Increasing this allows more non-HB neighbors to be considered; the
        nearest one is always taken regardless.
    update_central : bool
        Re-select central atoms every frame.  Leave False (default) for
        rigid / homogeneous systems; set True if the selection is dynamic.

    Notes
    -----
    * The analysis iterates over the *same* frames that were used by ``hba``
      (``hba.frames``).  ``start``, ``stop``, ``step`` passed to ``run()``
      further sub-select within those frames.
    * PBC minimum-image distances are handled by MDAnalysis' ``capped_distance``.
    """

    def __init__(
        self,
        hba,
        central_sel: str = "resname SOL and name OW",
        shell_cutoff: float = 4.5,
        update_central: bool = False,
    ):
        # Inherit trajectory from the same universe used by hba
        super().__init__(hba.u.trajectory)

        # Validate that hba has been run
        if hba.results.hbonds is None or len(hba.results.hbonds) == 0:
            raise NoDataError(
                "HydrogenBondAnalysis.results.hbonds is empty. "
                "Call hba.run() before constructing ZetaOrderParameter."
            )

        self.hba            = hba
        self.u              = hba.u
        self.central_sel    = central_sel
        self.shell_cutoff   = shell_cutoff
        self.update_central = update_central

        # Central atom group (OW atoms, one per molecule)
        self._central_ag = self.u.select_atoms(
            central_sel, updating=update_central
        )
        if len(self._central_ag) == 0:
            raise ValueError(
                f"Selection '{central_sel}' matched no atoms.  "
                "Check resname and atom name for your force field."
            )

        # Pre-build the HB partner lookup:
        #   _hb_lookup[frame][central_atom_index] = {(partner_idx, dist), ...}
        # This avoids repeated array scanning inside _single_frame.
        self._hb_lookup = self._build_hb_lookup()

        self.results = Results()

    # ------------------------------------------------------------------
    # Pre-computation: build per-frame HB partner lookup
    # ------------------------------------------------------------------

    def _build_hb_lookup(self):
        """Build a {frame: {ow_idx: {(partner_ow_idx, dist)}}} mapping.

        Both D→A and A→D directions are included so that every molecule
        learns about ALL of its HB partners regardless of whether it donated
        or accepted a given bond.
        """
        hb = self.hba.results.hbonds
        central_indices = set(self._central_ag.indices)

        lookup = {}  # frame → {central_idx → list of (partner_idx, dist)}
        for row in hb:
            frame   = int(row[0])
            d_idx   = int(row[1])
            a_idx   = int(row[3])
            dist    = float(row[4])

            if frame not in lookup:
                lookup[frame] = {}

            # Register from the donor's perspective (central = donor)
            if d_idx in central_indices:
                lookup[frame].setdefault(d_idx, []).append((a_idx, dist))

            # Register from the acceptor's perspective (central = acceptor)
            if a_idx in central_indices:
                lookup[frame].setdefault(a_idx, []).append((d_idx, dist))

        return lookup

    # ------------------------------------------------------------------
    # AnalysisBase protocol
    # ------------------------------------------------------------------

    def _prepare(self):
        n_frames  = len(self.frames)
        n_central = len(self._central_ag)
        dtype_f   = np.float32
        dtype_i   = np.int16

        self.results.zeta         = np.full((n_frames, n_central), np.nan, dtype=dtype_f)
        self.results.d_hb_far     = np.full((n_frames, n_central), np.nan, dtype=dtype_f)
        self.results.d_nonhb_near = np.full((n_frames, n_central), np.nan, dtype=dtype_f)
        self.results.n_hb         = np.zeros((n_frames, n_central),        dtype=dtype_i)

        # Simple counter incremented by _single_frame.
        # We CANNOT build a frame-number→row-index dict here because
        # self.frames is only pre-allocated (all zeros) at _prepare() time in
        # MDAnalysis >= 2.7; the actual frame numbers are written into it
        # during the loop inside _compute().  Using a counter avoids the
        # KeyError that results from mapping against stale zero values.
        self._frame_counter = 0

    def _single_frame(self):
        frame     = self._ts.frame
        row_idx   = self._frame_counter   # 0-based index into result arrays
        box       = self._ts.dimensions

        if self.update_central:
            central_ag = self.u.select_atoms(self.central_sel)
        else:
            central_ag = self._central_ag

        central_pos     = central_ag.positions          # (N, 3)
        central_indices = central_ag.indices            # (N,)
        # Fast lookup: atom index → position in central_ag array
        central_idx_to_pos = {idx: i for i, idx in enumerate(central_indices)}

        # Retrieve HB partners for this frame
        frame_hb = self._hb_lookup.get(frame, {})

        # ---- All pairwise OW–OW distances within shell_cutoff ---------------
        # self_capped_distance returns pairs (i, j) with i < j, and distances.
        # We need every ordered pair for the per-molecule lookup below.
        pairs, dists = self_capped_distance(
            central_pos,
            max_cutoff=self.shell_cutoff,
            min_cutoff=0.5,        # exclude self (distance = 0)
            box=box,
            return_distances=True,
        )
        # pairs shape: (M, 2), indexing INTO central_pos (0-based within central_ag)

        # Build per-central-atom dict of {neighbor_atom_idx: dist}
        # Store BOTH directions (i→j and j→i) for symmetric access.
        neighbor_dist = [{} for _ in range(len(central_indices))]
        for (pi, pj), d in zip(pairs, dists):
            neighbor_dist[pi][central_indices[pj]] = d
            neighbor_dist[pj][central_indices[pi]] = d

        # ---- Per-molecule ζ computation -------------------------------------
        for local_i, central_idx in enumerate(central_indices):
            hb_partners = frame_hb.get(central_idx, [])  # [(partner_idx, dist), ...]
            n_hb = len(hb_partners)
            self.results.n_hb[row_idx, local_i] = min(n_hb, np.iinfo(np.int16).max)

            if n_hb == 0:
                # No HB partners → ζ undefined
                continue

            # Farthest HB partner distance
            hb_dists   = [d for (_, d) in hb_partners]
            d_hb_far   = max(hb_dists)
            hb_set     = {idx for (idx, _) in hb_partners}

            # Nearest non-HB neighbor distance from the shell
            nbr_dict   = neighbor_dist[local_i]
            non_hb_dists = [
                d for nbr_idx, d in nbr_dict.items()
                if nbr_idx not in hb_set
            ]

            if len(non_hb_dists) == 0:
                # No non-HB neighbor within shell_cutoff → ζ undefined
                continue

            d_nonhb_near = min(non_hb_dists)
            zeta_val     = d_nonhb_near - d_hb_far

            self.results.zeta        [row_idx, local_i] = zeta_val
            self.results.d_hb_far    [row_idx, local_i] = d_hb_far
            self.results.d_nonhb_near[row_idx, local_i] = d_nonhb_near

        self._frame_counter += 1

    def _conclude(self):
        # self.frames is fully populated with real frame numbers only after the
        # loop finishes, so we build the frame→row lookup here (not in _prepare).
        self._frame_to_row = {int(f): i for i, f in enumerate(self.frames)}
        # Store atom indices and times for downstream convenience
        self.results.central_indices = self._central_ag.indices.copy()
        self.results.times           = self.frames * self.u.trajectory.dt

    # ------------------------------------------------------------------
    # Analysis helpers – temporal
    # ------------------------------------------------------------------

    def mean_zeta(self, ignore_nan: bool = True) -> float:
        """Grand mean of ζ over all molecules and all frames.

        Parameters
        ----------
        ignore_nan : bool
            If True (default), NaN entries (undefined ζ) are excluded.

        Returns
        -------
        float
        """
        self._require_results()
        fn = np.nanmean if ignore_nan else np.mean
        return float(fn(self.results.zeta))

    def get_zeta_timeseries(
        self,
        atom_indices=None,
        average_over_molecules: bool = True,
    ) -> np.ndarray:
        """Return ζ as a function of time.

        Parameters
        ----------
        atom_indices : array-like of int or None
            Atom indices (0-based, matching universe numbering) of the
            central atoms to include.  If None, all central atoms are used.
        average_over_molecules : bool
            If True (default), return the per-frame mean over selected molecules.
            If False, return the full (n_frames, n_selected) matrix.

        Returns
        -------
        np.ndarray
            Shape (n_frames,) if ``average_over_molecules=True``,
            else (n_frames, n_selected).
        """
        self._require_results()
        z = self._select_columns(self.results.zeta, atom_indices)
        if average_over_molecules:
            return np.nanmean(z, axis=1)
        return z

    def get_zeta_autocorrelation(
        self,
        atom_index: int,
        tau_max: Optional[int] = None,
    ) -> tuple:
        """Time autocorrelation of ζ(t) for a single central molecule.

        Computes  C(τ) = <δζ(0) δζ(τ)> / <δζ²>
        where δζ(t) = ζ(t) − <ζ>.  NaN frames are linearly interpolated
        before computing the ACF so that the denominator is well-defined.

        Parameters
        ----------
        atom_index : int
            Atom index of the chosen central molecule.
        tau_max : int or None
            Maximum lag in frames.  Defaults to n_frames // 2.

        Returns
        -------
        tau_ps : np.ndarray
        acf    : np.ndarray
        """
        self._require_results()
        col_idx = self._atom_index_to_col(atom_index)
        series  = self.results.zeta[:, col_idx].astype(float)

        # Interpolate NaN gaps
        series = _interpolate_nans(series)

        if tau_max is None:
            tau_max = len(series) // 2

        series  -= np.mean(series)                   # detrend
        var      = np.var(series)
        if var == 0:
            return (
                np.arange(tau_max + 1) * self.u.trajectory.dt,
                np.zeros(tau_max + 1),
            )

        acf = np.array([
            np.mean(series[: len(series) - tau] * series[tau:]) / var
            for tau in range(tau_max + 1)
        ])
        tau_ps = np.arange(tau_max + 1) * self.u.trajectory.dt
        return tau_ps, acf

    def mean_zeta_per_frame(self) -> np.ndarray:
        """Mean ζ across all molecules at each frame.

        Returns
        -------
        np.ndarray, shape (n_frames,)
        """
        self._require_results()
        return np.nanmean(self.results.zeta, axis=1)

    def std_zeta_per_frame(self) -> np.ndarray:
        """Standard deviation of ζ across all molecules at each frame.

        Returns
        -------
        np.ndarray, shape (n_frames,)
        """
        self._require_results()
        return np.nanstd(self.results.zeta, axis=1)

    def zeta_percentile_timeseries(
        self, percentiles=(10, 25, 50, 75, 90)
    ) -> dict:
        """Percentiles of the ζ distribution across molecules at each frame.

        Returns
        -------
        dict : {percentile_value: np.ndarray of shape (n_frames,)}
        """
        self._require_results()
        return {
            p: np.nanpercentile(self.results.zeta, p, axis=1)
            for p in percentiles
        }

    # ------------------------------------------------------------------
    # Analysis helpers – distributional / spatial
    # ------------------------------------------------------------------

    def get_zeta_distribution(
        self,
        bins: int = 100,
        range_: tuple = (-2.0, 2.0),
    ) -> tuple:
        """Probability distribution P(ζ) over all molecules and frames.

        Parameters
        ----------
        bins  : int
        range_ : (float, float)
            Histogram edges in Å.

        Returns
        -------
        bin_centers : np.ndarray, shape (bins,)
        prob_density: np.ndarray, shape (bins,)   (normalised to unit area)
        """
        self._require_results()
        flat   = self.results.zeta.ravel()
        flat   = flat[~np.isnan(flat)]
        counts, edges = np.histogram(flat, bins=bins, range=range_, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts

    def classify_structure(
        self, threshold: float = 0.0
    ) -> dict:
        """Classify each molecule as tetrahedral or distorted.

        A molecule is "tetrahedral" at a given frame if ζ > threshold.
        Frames where ζ is NaN are excluded from the fraction.

        Parameters
        ----------
        threshold : float
            Default 0.0 Å (sign of ζ).

        Returns
        -------
        dict with keys:
          ``"tetrahedral_fraction"``  : float  – system-wide mean
          ``"per_molecule_fraction"`` : np.ndarray, shape (n_central,)
          ``"per_frame_fraction"``    : np.ndarray, shape (n_frames,)
        """
        self._require_results()
        z = self.results.zeta

        above = (z > threshold).astype(float)
        valid = (~np.isnan(z)).astype(float)

        # Per-molecule: fraction of frames where ζ > threshold
        per_mol = np.where(
            valid.sum(axis=0) > 0,
            above.sum(axis=0) / valid.sum(axis=0),
            np.nan,
        )
        # Per-frame: fraction of molecules where ζ > threshold
        per_frame = np.where(
            valid.sum(axis=1) > 0,
            above.sum(axis=1) / valid.sum(axis=1),
            np.nan,
        )
        global_frac = float(np.nanmean(per_mol))

        return {
            "tetrahedral_fraction":   global_frac,
            "per_molecule_fraction":  per_mol,
            "per_frame_fraction":     per_frame,
        }

    def mean_zeta_per_molecule(self) -> np.ndarray:
        """Time-averaged ζ for each central molecule.

        Returns
        -------
        np.ndarray, shape (n_central,)
        """
        self._require_results()
        return np.nanmean(self.results.zeta, axis=0)

    def get_joint_distribution(
        self,
        bins: int = 60,
        d_hb_range: tuple = (2.0, 4.0),
        d_nonhb_range: tuple = (2.5, 5.0),
    ) -> tuple:
        """2-D joint distribution of (d_HB_far, d_nonHB_near).

        This reveals the correlation structure: where in the
        (d_HB_far, d_nonHB_near) plane does the system spend most of its time?
        The diagonal d_nonHB_near = d_HB_far is the ζ = 0 contour.

        Returns
        -------
        H            : np.ndarray, shape (bins, bins)  – 2-D histogram (normalised)
        d_hb_edges   : np.ndarray, shape (bins+1,)
        d_nonhb_edges: np.ndarray, shape (bins+1,)
        """
        self._require_results()
        x = self.results.d_hb_far.ravel()
        y = self.results.d_nonhb_near.ravel()
        valid = ~(np.isnan(x) | np.isnan(y))
        H, xe, ye = np.histogram2d(
            x[valid], y[valid],
            bins=bins,
            range=[d_hb_range, d_nonhb_range],
            density=True,
        )
        return H, xe, ye

    def spatial_zeta_map(self, frame: int) -> dict:
        """Return ζ value and 3-D position for each central molecule at one frame.

        Useful for visualisation (e.g., colour-coded VMD/OVITO snapshot).

        Parameters
        ----------
        frame : int

        Returns
        -------
        dict with keys:
          ``"positions"``  : np.ndarray, shape (n_central, 3)
          ``"zeta"``       : np.ndarray, shape (n_central,)  (NaN where undefined)
          ``"n_hb"``       : np.ndarray, shape (n_central,)
        """
        self._require_results()
        if frame not in self._frame_to_row:
            raise ValueError(
                f"Frame {frame} was not analysed.  "
                f"Available frames: {self.frames[0]} … {self.frames[-1]}."
            )
        row = self._frame_to_row[frame]
        self.u.trajectory[frame]
        positions = self._central_ag.positions.copy()
        return {
            "positions": positions,
            "zeta":      self.results.zeta[row].copy(),
            "n_hb":      self.results.n_hb[row].copy(),
        }

    def get_conditional_zeta(self) -> dict:
        """Mean and std of ζ conditioned on the number of H-bond partners.

        Returns a dict keyed by n_hb (0, 1, 2, 3, 4, …) with sub-keys
        ``"mean"``, ``"std"``, ``"count"``.

        This reveals how local order changes with H-bond coordination number.
        """
        self._require_results()
        n_hb_flat  = self.results.n_hb.ravel()
        zeta_flat  = self.results.zeta.ravel()

        result = {}
        for n in np.unique(n_hb_flat):
            mask = (n_hb_flat == n) & (~np.isnan(zeta_flat))
            subset = zeta_flat[mask]
            if len(subset) == 0:
                continue
            result[int(n)] = {
                "mean":  float(np.mean(subset)),
                "std":   float(np.std(subset)),
                "count": int(len(subset)),
            }
        return result

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, prefix: str = "zeta"):
        """Save all result arrays to .npy files.

        Files written: ``{prefix}_zeta.npy``, ``{prefix}_d_hb_far.npy``,
        ``{prefix}_d_nonhb_near.npy``, ``{prefix}_n_hb.npy``,
        ``{prefix}_central_indices.npy``, ``{prefix}_frames.npy``.
        """
        self._require_results()
        np.save(f"{prefix}_zeta.npy",           self.results.zeta)
        np.save(f"{prefix}_d_hb_far.npy",       self.results.d_hb_far)
        np.save(f"{prefix}_d_nonhb_near.npy",   self.results.d_nonhb_near)
        np.save(f"{prefix}_n_hb.npy",           self.results.n_hb)
        np.save(f"{prefix}_central_indices.npy",self.results.central_indices)
        np.save(f"{prefix}_frames.npy",         self.frames)
        logger.info("Saved zeta results with prefix '%s'.", prefix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_results(self):
        if not hasattr(self.results, "zeta") or self.results.zeta is None:
            raise NoDataError(
                "No results yet.  Call ZetaOrderParameter.run() first."
            )

    def _select_columns(self, arr, atom_indices):
        """Select columns of arr corresponding to the given atom indices."""
        if atom_indices is None:
            return arr
        atom_indices = np.asarray(atom_indices)
        central_idx  = self.results.central_indices
        cols = np.array([
            np.where(central_idx == ai)[0][0]
            for ai in atom_indices
            if ai in central_idx
        ])
        if len(cols) == 0:
            raise ValueError("None of the requested atom_indices are in central_sel.")
        return arr[:, cols]

    def _atom_index_to_col(self, atom_index: int) -> int:
        """Convert a universe atom index to a column index in results arrays."""
        central_idx = self.results.central_indices
        matches = np.where(central_idx == atom_index)[0]
        if len(matches) == 0:
            raise ValueError(
                f"Atom index {atom_index} is not in the central selection."
            )
        return int(matches[0])


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D array.

    Leading or trailing NaN blocks are filled with the nearest valid value.
    """
    arr   = arr.copy()
    nans  = np.isnan(arr)
    if not np.any(nans):
        return arr
    idx   = np.arange(len(arr))
    valid = ~nans
    if not np.any(valid):
        return np.zeros_like(arr)
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr