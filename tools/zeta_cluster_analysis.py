# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Zeta Cluster Analysis
======================

Identifies spatially connected structural domains (clusters) at each
trajectory frame based on the zeta order parameter, tracks them across
frames using the Jaccard-overlap Hungarian algorithm, and computes
cluster-level statistical characteristics.

Pipeline overview
-----------------
  ZetaClusterAnalysis.run()
      ↳ _single_frame():  DBSCAN on tetrahedral / distorted sub-populations
                          → per-frame cluster objects with atom membership
      ↳ _conclude():      Hungarian tracking → persistent global cluster IDs
                          → per-cluster time-series of size, ζ̄, Rg, CoM

Cluster labelling convention
-----------------------------
  label = +1  (class "tetrahedral")  ζ >  zeta_threshold
  label = -1  (class "distorted")    ζ ≤  zeta_threshold
  label =  0  (noise / undefined)    NaN ζ, or DBSCAN noise

Cluster ID convention
----------------------
  Per-frame IDs are temporary (0-based integers within each frame/class).
  Global tracked IDs are assigned in _conclude() and stored in
  ``results.track_*`` attributes.  ID = -1 means untracked noise.

Key results attributes
-----------------------
  results.frame_clusters  : list[dict] of length n_frames
      Each dict: {(class_label, local_id): ClusterSnapshot}

  results.tracks          : dict  {global_id: ClusterTrack}
      Each ClusterTrack contains:
          .frames         : np.ndarray  – trajectory frame numbers
          .label          : int         – +1 (tet) or -1 (dis)
          .size           : np.ndarray  – n_molecules per frame
          .mean_zeta      : np.ndarray
          .std_zeta       : np.ndarray
          .rg             : np.ndarray  – radius of gyration (Å)
          .com            : np.ndarray, shape (T, 3) – centre of mass (Å)
          .atom_sets      : list[frozenset] – atom indices per frame
          .lifetime       : int          – number of frames the track exists
          .birth_frame    : int
          .death_frame    : int
          .total_com_displacement : float – |CoM(last) − CoM(first)| (Å)

  results.frame_labels    : np.ndarray, shape (n_frames, n_central)
      Per-molecule structural label (+1 / -1 / 0) at every frame.

  results.frame_cluster_ids : np.ndarray, shape (n_frames, n_central)
      Global tracked cluster ID for every molecule at every frame.
      Molecules not belonging to any tracked cluster have ID = -1.

Dependencies
------------
  MDAnalysis, NumPy, SciPy (DBSCAN via sklearn OR scipy.spatial,
  Hungarian via scipy.optimize.linear_sum_assignment)
  scikit-learn  (optional – used for DBSCAN; falls back to
                 scipy-based implementation if absent)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.exceptions import NoDataError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import sklearn DBSCAN; fall back to a minimal pure-NumPy version
# ---------------------------------------------------------------------------
try:
    from sklearn.cluster import DBSCAN as _DBSCAN
    def _dbscan(positions, eps, min_samples):
        """Return integer cluster labels (−1 = noise)."""
        if len(positions) == 0:
            return np.array([], dtype=int)
        return _DBSCAN(eps=eps, min_samples=min_samples,
                       metric="euclidean", n_jobs=1).fit_predict(positions)
    _SKLEARN = True
except ImportError:
    warnings.warn(
        "scikit-learn not found; falling back to a minimal DBSCAN "
        "implementation.  Install scikit-learn for better performance.",
        ImportWarning,
        stacklevel=2,
    )
    _SKLEARN = False

    def _dbscan(positions, eps, min_samples):
        """Minimal DBSCAN without scikit-learn.  O(N²) – adequate for N<5000."""
        n = len(positions)
        if n == 0:
            return np.array([], dtype=int)
        # Pairwise distance matrix
        diff  = positions[:, None, :] - positions[None, :, :]   # (N,N,3)
        dist2 = (diff ** 2).sum(axis=-1)                         # (N,N)
        eps2  = eps * eps
        neighbors = [np.where(dist2[i] <= eps2)[0] for i in range(n)]
        labels = -np.ones(n, dtype=int)
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            if len(neighbors[i]) < min_samples:
                continue
            labels[i] = cluster_id
            stack = list(neighbors[i])
            while stack:
                j = stack.pop()
                if labels[j] == -1:
                    labels[j] = cluster_id
                    if len(neighbors[j]) >= min_samples:
                        stack.extend(
                            k for k in neighbors[j] if labels[k] == -1
                        )
                elif labels[j] == -1:   # unvisited border
                    labels[j] = cluster_id
            cluster_id += 1
        return labels


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ClusterSnapshot:
    """All information about one cluster at one frame."""
    frame:       int
    label:       int              # +1 tetrahedral, -1 distorted
    local_id:    int              # per-frame, per-class DBSCAN id
    atom_set:    frozenset        # atom indices (0-based universe indices)
    positions:   np.ndarray       # shape (n, 3)
    zeta_values: np.ndarray       # shape (n,)

    @property
    def size(self):
        return len(self.atom_set)

    @property
    def mean_zeta(self):
        valid = self.zeta_values[~np.isnan(self.zeta_values)]
        return float(np.mean(valid)) if len(valid) else np.nan

    @property
    def std_zeta(self):
        valid = self.zeta_values[~np.isnan(self.zeta_values)]
        return float(np.std(valid)) if len(valid) > 1 else np.nan

    @property
    def com(self):
        """Centre of mass (uniform weights)."""
        return self.positions.mean(axis=0)

    @property
    def rg(self):
        """Radius of gyration (Å)."""
        if len(self.positions) < 2:
            return 0.0
        delta = self.positions - self.com
        return float(np.sqrt((delta ** 2).sum(axis=1).mean()))


@dataclass
class ClusterTrack:
    """Full time-resolved record of a single tracked cluster."""
    global_id:   int
    label:       int                    # +1 or -1

    # Filled in _conclude
    frames:      np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    size:        np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    mean_zeta:   np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    std_zeta:    np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rg:          np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    com:         np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    atom_sets:   list        = field(default_factory=list)   # [frozenset, ...]

    @property
    def lifetime(self):
        return len(self.frames)

    @property
    def birth_frame(self):
        return int(self.frames[0]) if len(self.frames) else -1

    @property
    def death_frame(self):
        return int(self.frames[-1]) if len(self.frames) else -1

    @property
    def total_com_displacement(self):
        if len(self.com) < 2:
            return 0.0
        return float(np.linalg.norm(self.com[-1] - self.com[0]))

    @property
    def mean_size(self):
        return float(self.size.mean()) if len(self.size) else np.nan

    @property
    def mean_rg(self):
        return float(self.rg.mean()) if len(self.rg) else np.nan


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ZetaClusterAnalysis(AnalysisBase):
    """Identify, track and characterise structural clusters via ζ + DBSCAN.

    Parameters
    ----------
    zop : ZetaOrderParameter
        A **completed** ZetaOrderParameter object.
    eps : float
        DBSCAN neighbourhood radius (Å).  Typically slightly larger than
        the first O-O coordination shell maximum (~3.1–3.5 Å for water).
        Default 3.5 Å.
    min_samples : int
        DBSCAN minimum cluster size.  Default 4 (a complete tetrahedral
        shell including the central molecule).
    zeta_threshold : float
        ζ value separating tetrahedral (ζ > threshold) from distorted
        (ζ ≤ threshold) molecules.  Default 0.0 Å.
    min_track_overlap : float
        Minimum Jaccard overlap [0, 1) for two clusters in consecutive
        frames to be considered the same tracked entity.  Default 0.3.
    min_track_lifetime : int
        Clusters that exist for fewer than this many frames are discarded
        from ``results.tracks`` (but still appear in per-frame results).
        Default 2.
    cluster_classes : tuple
        Which structural classes to cluster.  Any subset of (1, -1).
        Default (1, -1) – cluster both tetrahedral and distorted molecules.
    """

    def __init__(
        self,
        zop,
        eps:                  float = 3.5,
        min_samples:          int   = 4,
        zeta_threshold:       float = 0.0,
        min_track_overlap:    float = 0.3,
        min_track_lifetime:   int   = 2,
        cluster_classes:      tuple = (1, -1),
    ):
        # Validate zop
        if not hasattr(zop.results, "zeta") or zop.results.zeta is None:
            raise NoDataError("ZetaOrderParameter.results.zeta is empty. "
                              "Call zop.run() first.")

        super().__init__(zop.u.trajectory)
        self.zop               = zop
        self.u                 = zop.u
        self.eps               = eps
        self.min_samples       = min_samples
        self.zeta_threshold    = zeta_threshold
        self.min_track_overlap = min_track_overlap
        self.min_track_lifetime = min_track_lifetime
        self.cluster_classes   = cluster_classes

        # Convenience references into zop results
        self._central_ag      = zop._central_ag
        self._central_indices = zop.results.central_indices   # (N,)
        self._zeta_arr        = zop.results.zeta              # (n_frames, N)
        self._zop_frames      = zop.frames                    # trajectory frames analysed by zop
        self._zop_frame_to_row = zop._frame_to_row           # frame → row index in zeta_arr

        self.results = Results()

    # ------------------------------------------------------------------
    # AnalysisBase protocol
    # ------------------------------------------------------------------

    def _prepare(self):
        n_frames  = len(self.frames)
        n_central = len(self._central_indices)

        # Per-frame cluster snapshots: list of dicts
        # index i corresponds to self.frames[i]
        self._frame_snapshots: List[Dict] = [{} for _ in range(n_frames)]

        # Structural label array (+1/-1/0) – filled during loop
        self.results.frame_labels = np.zeros((n_frames, n_central), dtype=np.int8)

        # Global cluster-ID array (filled in _conclude)
        self.results.frame_cluster_ids = np.full(
            (n_frames, n_central), -1, dtype=np.int32
        )

        self._frame_counter = 0

    def _single_frame(self):
        frame   = self._ts.frame
        fi      = self._frame_counter        # row index into result arrays
        box     = self._ts.dimensions

        # ---- Retrieve ζ values for this frame ----------------------------
        if frame not in self._zop_frame_to_row:
            # Frame was not analysed by zop; skip silently
            self._frame_counter += 1
            return
        zop_row  = self._zop_frame_to_row[frame]
        zeta_row = self._zeta_arr[zop_row]   # shape (n_central,)

        # ---- Assign structural labels ------------------------------------
        labels = np.zeros(len(self._central_indices), dtype=np.int8)
        valid  = ~np.isnan(zeta_row)
        labels[valid & (zeta_row >  self.zeta_threshold)] = +1
        labels[valid & (zeta_row <= self.zeta_threshold)] = -1
        self.results.frame_labels[fi] = labels

        # ---- Read positions (current frame) ------------------------------
        positions_all = self._central_ag.positions.copy()  # (N, 3)

        # ---- DBSCAN on each structural class -----------------------------
        snapshots = {}
        for cls in self.cluster_classes:
            cls_mask = labels == cls
            if cls_mask.sum() < self.min_samples:
                continue

            cls_local_indices = np.where(cls_mask)[0]           # indices into central_ag
            cls_positions     = positions_all[cls_local_indices] # (k, 3)
            cls_atom_indices  = self._central_indices[cls_local_indices]
            cls_zeta          = zeta_row[cls_local_indices]

            db_labels = _dbscan(cls_positions, self.eps, self.min_samples)

            for cid in np.unique(db_labels):
                if cid == -1:   # DBSCAN noise
                    continue
                members = db_labels == cid
                snap = ClusterSnapshot(
                    frame       = frame,
                    label       = cls,
                    local_id    = int(cid),
                    atom_set    = frozenset(cls_atom_indices[members].tolist()),
                    positions   = cls_positions[members].copy(),
                    zeta_values = cls_zeta[members].copy(),
                )
                snapshots[(cls, int(cid))] = snap

        self._frame_snapshots[fi] = snapshots
        self._frame_counter += 1

    def _conclude(self):
        """Build persistent tracks via per-class Hungarian matching."""
        # frame_to_fi maps trajectory frame number → index into self.frames
        frame_to_fi = {int(f): i for i, f in enumerate(self.frames)}
        self.results.frame_cluster_ids = np.full(
            (len(self.frames), len(self._central_indices)), -1, dtype=np.int32
        )

        # ---- Per-class tracking ------------------------------------------
        all_tracks: Dict[int, ClusterTrack] = {}
        next_global_id = 0

        for cls in self.cluster_classes:
            # active_map: local_key → global_id  (for the previous frame)
            active_map: Dict[Tuple[int, int], int] = {}

            for fi, frame in enumerate(self.frames):
                snaps = self._frame_snapshots[fi]
                # Current frame's clusters for this class
                curr_keys = [(cls, cid) for (lbl, cid) in snaps if lbl == cls]

                if not curr_keys:
                    active_map = {}
                    continue

                if not active_map:
                    # First frame for this class: assign fresh IDs
                    new_map = {}
                    for key in curr_keys:
                        gid = next_global_id
                        next_global_id += 1
                        snap = snaps[key]
                        all_tracks[gid] = ClusterTrack(
                            global_id=gid, label=cls
                        )
                        _track_append(all_tracks[gid], frame, snap)
                        new_map[key] = gid
                    active_map = new_map
                    continue

                # Hungarian matching on Jaccard overlap
                prev_keys = list(active_map.keys())
                J = np.zeros((len(prev_keys), len(curr_keys)), dtype=float)
                for pi, pk in enumerate(prev_keys):
                    A = snaps.get(pk, None)
                    prev_snap = self._frame_snapshots[fi - 1].get(pk)
                    if prev_snap is None:
                        continue
                    for ci, ck in enumerate(curr_keys):
                        B = snaps[ck]
                        inter = len(prev_snap.atom_set & B.atom_set)
                        union = len(prev_snap.atom_set | B.atom_set)
                        J[pi, ci] = inter / union if union > 0 else 0.0

                # Maximise overlap = minimise −J
                row_ind, col_ind = linear_sum_assignment(-J)

                matched_curr = set()
                new_map = {}
                for pi, ci in zip(row_ind, col_ind):
                    if J[pi, ci] >= self.min_track_overlap:
                        pk  = prev_keys[pi]
                        ck  = curr_keys[ci]
                        gid = active_map[pk]
                        _track_append(all_tracks[gid], frame, snaps[ck])
                        new_map[ck] = gid
                        matched_curr.add(ci)

                # Unmatched current clusters → new tracks
                for ci, ck in enumerate(curr_keys):
                    if ci not in matched_curr:
                        gid = next_global_id
                        next_global_id += 1
                        all_tracks[gid] = ClusterTrack(global_id=gid, label=cls)
                        _track_append(all_tracks[gid], frame, snaps[ck])
                        new_map[ck] = gid

                active_map = new_map

        # ---- Finalise tracks (convert lists → arrays) --------------------
        for track in all_tracks.values():
            _finalise_track(track)

        # ---- Filter short-lived tracks -----------------------------------
        self.results.tracks = {
            gid: t for gid, t in all_tracks.items()
            if t.lifetime >= self.min_track_lifetime
        }

        # ---- Fill frame_cluster_ids array --------------------------------
        # Map atom index → column position in central_indices
        atom_to_col = {int(idx): ci
                       for ci, idx in enumerate(self._central_indices)}

        for gid, track in self.results.tracks.items():
            for fi_t, (frame, atom_set) in enumerate(
                zip(track.frames, track.atom_sets)
            ):
                fi = frame_to_fi.get(int(frame))
                if fi is None:
                    continue
                for atom_idx in atom_set:
                    col = atom_to_col.get(int(atom_idx))
                    if col is not None:
                        self.results.frame_cluster_ids[fi, col] = gid

        # Store convenience arrays
        self.results.central_indices = self._central_indices.copy()
        self.results.times           = self.frames * self.u.trajectory.dt

    # ------------------------------------------------------------------
    # Statistical summary methods
    # ------------------------------------------------------------------

    def cluster_summary(self) -> dict:
        """Aggregate statistics across all tracked clusters.

        Returns
        -------
        dict with keys (separately for each class label):
          ``"n_tracks"``             – total tracked clusters
          ``"mean_lifetime_frames"`` – mean lifetime in frames
          ``"mean_size"``            – mean number of molecules
          ``"mean_rg"``              – mean radius of gyration (Å)
          ``"mean_displacement"``    – mean CoM displacement (Å)
          ``"size_distribution"``    – flat array of all per-frame sizes
          ``"lifetime_distribution"``– array of lifetimes (frames)
        """
        self._require_results()
        out = {}
        for cls, name in [(+1, "tetrahedral"), (-1, "distorted")]:
            tracks = [t for t in self.results.tracks.values() if t.label == cls]
            if not tracks:
                out[name] = None
                continue
            sizes  = np.concatenate([t.size for t in tracks])
            lives  = np.array([t.lifetime for t in tracks])
            rgs    = np.concatenate([t.rg   for t in tracks])
            disps  = np.array([t.total_com_displacement for t in tracks])
            out[name] = {
                "n_tracks":              len(tracks),
                "mean_lifetime_frames":  float(lives.mean()),
                "std_lifetime_frames":   float(lives.std()),
                "mean_size":             float(sizes.mean()),
                "std_size":              float(sizes.std()),
                "mean_rg":               float(rgs.mean()),
                "std_rg":                float(rgs.std()),
                "mean_displacement":     float(disps.mean()),
                "std_displacement":      float(disps.std()),
                "size_distribution":     sizes,
                "lifetime_distribution": lives,
            }
        return out

    def get_cluster_size_timeseries(self, label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Total number of molecules in all clusters of one class, per frame.

        Parameters
        ----------
        label : +1 or -1

        Returns
        -------
        times : np.ndarray, shape (n_frames,)
        counts : np.ndarray, shape (n_frames,)
        """
        self._require_results()
        counts = np.zeros(len(self.frames), dtype=int)
        for gid, track in self.results.tracks.items():
            if track.label != label:
                continue
            frame_to_fi = {int(f): i for i, f in enumerate(self.frames)}
            for frame, sz in zip(track.frames, track.size):
                fi = frame_to_fi.get(int(frame))
                if fi is not None:
                    counts[fi] += sz
        return self.results.times, counts

    def get_largest_cluster_track(self, label: int = 1) -> Optional[ClusterTrack]:
        """Return the ClusterTrack with the greatest mean size for a given class."""
        self._require_results()
        candidates = [t for t in self.results.tracks.values() if t.label == label]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.mean_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_results(self):
        if not hasattr(self.results, "tracks"):
            raise NoDataError("Call ZetaClusterAnalysis.run() first.")


# ---------------------------------------------------------------------------
# Module-level helpers for track building
# ---------------------------------------------------------------------------

def _track_append(track: ClusterTrack, frame: int, snap: ClusterSnapshot):
    """Append one ClusterSnapshot onto a ClusterTrack (using lists during build)."""
    if not hasattr(track, "_frames_list"):
        track._frames_list     = []
        track._size_list       = []
        track._mean_zeta_list  = []
        track._std_zeta_list   = []
        track._rg_list         = []
        track._com_list        = []
        track._atom_sets_list  = []

    track._frames_list.append(frame)
    track._size_list.append(snap.size)
    track._mean_zeta_list.append(snap.mean_zeta)
    track._std_zeta_list.append(snap.std_zeta)
    track._rg_list.append(snap.rg)
    track._com_list.append(snap.com)
    track._atom_sets_list.append(snap.atom_set)


def _finalise_track(track: ClusterTrack):
    """Convert the temporary lists into NumPy arrays."""
    if not hasattr(track, "_frames_list"):
        return
    track.frames    = np.array(track._frames_list,    dtype=int)
    track.size      = np.array(track._size_list,      dtype=int)
    track.mean_zeta = np.array(track._mean_zeta_list, dtype=float)
    track.std_zeta  = np.array(track._std_zeta_list,  dtype=float)
    track.rg        = np.array(track._rg_list,        dtype=float)
    track.com       = np.array(track._com_list,       dtype=float)
    track.atom_sets = track._atom_sets_list

    # Clean up temporary lists
    for attr in ("_frames_list", "_size_list", "_mean_zeta_list",
                 "_std_zeta_list", "_rg_list", "_com_list", "_atom_sets_list"):
        delattr(track, attr)
