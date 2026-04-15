# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Custom Hydrogen Bond Analysis
==============================

Standalone hydrogen bond analysis module adapted from
MDAnalysis.analysis.hydrogenbonds.hbond_analysis (v2.x), with the following
geometric criteria substituted for the MDAnalysis defaults:

  Criterion                     MDAnalysis default    This module default
  ──────────────────────────────────────────────────────────────────────
  Distance (Donor–Acceptor)     D-A ≤ 3.0 Å           D-A ≤ 3.5 Å
  Angle (measured at …)         D-H-A at H  > 150°     H-D-A at D  < 30°
  ──────────────────────────────────────────────────────────────────────

The H-D-A convention (angle at the *donor*) is standard in water
reorientation dynamics studies (Laage & Hynes 2006, 2008) and hydrogen-bond
network rearrangement analyses.  A small H-D-A angle means the hydrogen
lies nearly along the donor→acceptor axis, which is the expected geometry
for a genuine hydrogen bond.

Output columns in ``results.hbonds``
--------------------------------------
  col 0 – frame index
  col 1 – donor atom index   (0-based, matches Universe atom order)
  col 2 – hydrogen atom index
  col 3 – acceptor atom index
  col 4 – D-A distance (Å)
  col 5 – H-D-A angle  (degrees, at the donor)

Usage
-----
::

    import MDAnalysis as mda
    from custom_hbond_analysis import HydrogenBondAnalysis as HBA

    u = mda.Universe("topology.tpr", "trajectory.xtc")

    # Pure-water example (TIP3P / SPC-E naming)
    hba = HBA(
        universe=u,
        donors_sel="resname SOL and name OW",
        hydrogens_sel="resname SOL and name HW1 HW2",
        acceptors_sel="resname SOL and name OW",
        d_h_cutoff=1.2,
        d_a_cutoff=3.5,
        h_d_a_angle_cutoff=30.0,
    )
    hba.run()

    # Downstream: network rearrangement
    events = hba.count_hbond_events()

    # Downstream: reorientation dynamics
    switches = hba.find_hbond_switches()

Dependencies
------------
  MDAnalysis >= 2.0.0
  NumPy
  SciPy (optional, only for networkx-free adjacency export)
"""

import logging
import warnings
from collections.abc import Iterable

import numpy as np

from MDAnalysis.analysis.base import AnalysisBase, Results, ResultsGroup
from MDAnalysis.lib.distances import capped_distance, calc_angles
from MDAnalysis.lib.correlations import autocorrelation, correct_intermittency
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.hydrogenbonds.hbond_autocorrel import find_hydrogen_donors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: PBC-aware vector from atom positions
# ---------------------------------------------------------------------------

def _vector_with_pbc(pos_from, pos_to, box):
    """Return the minimum-image displacement vector(s) from pos_from to pos_to.

    Parameters
    ----------
    pos_from : np.ndarray, shape (N, 3) or (3,)
    pos_to   : np.ndarray, same shape
    box      : array-like, shape (6,) – MDAnalysis box [lx, ly, lz, α, β, γ]
               Pass None to skip PBC wrapping.

    Returns
    -------
    np.ndarray, same shape as input
    """
    diff = pos_to - pos_from
    if box is not None:
        # Only handle orthorhombic boxes (α=β=γ=90).
        # For triclinic boxes use MDAnalysis.lib.distances.minimize_vectors.
        lengths = box[:3]
        diff -= lengths * np.round(diff / lengths)
    return diff


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HydrogenBondAnalysis(AnalysisBase):
    """Hydrogen bond analysis using the H-D-A angle criterion.

    Geometric criteria (defaults)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **D-A distance** ≤ ``d_a_cutoff`` (default 3.5 Å) – distance between the
      donor heavy atom and the acceptor heavy atom.
    * **H-D-A angle** ≤ ``h_d_a_angle_cutoff`` (default 30°) – angle measured
      *at the donor*, between the D←H and D→A directions.  A small angle means
      the hydrogen points roughly toward the acceptor.

    These two criteria together are equivalent to placing the hydrogen inside
    a cone of half-angle 30° aligned along the D→A axis, and are the standard
    criteria used in reorientation-dynamics and network-rearrangement studies.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    donors_sel : str or None
        Selection string for donor heavy atoms.  Leave as ``None`` when the
        topology carries bond information so that D-H pairs are resolved
        unambiguously from the bond graph.
    hydrogens_sel : str or None
        Selection string for hydrogen atoms.  Leave as ``None`` to auto-detect
        via ``guess_hydrogens()``.
    acceptors_sel : str or None
        Selection string for acceptor heavy atoms.  Leave as ``None`` to
        auto-detect via ``guess_acceptors()``.
    between : list or None
        Restrict hydrogen bonds to pairs of atom groups.  Same semantics as
        the MDAnalysis original: pass ``["groupA", "groupB"]`` to find only
        A–B bonds, or ``[["groupA","groupB"], ["groupA","groupA"]]`` for
        multiple pairs.
    d_h_cutoff : float
        D-H distance cutoff used *only* when ``donors_sel`` is provided and no
        bond information is available (to build donor-hydrogen pairs).
        Default 1.2 Å.
    d_a_cutoff : float
        Maximum D-A (donor-to-acceptor heavy-atom) distance for a hydrogen
        bond.  Default **3.5 Å**.
    h_d_a_angle_cutoff : float
        Maximum H-D-A angle (at the donor) for a hydrogen bond, in degrees.
        Default **30°**.
    update_selections : bool
        If ``True`` (default), atom selections are re-evaluated every frame.
        Set to ``False`` for a rigid system to speed up the run.

    Attributes
    ----------
    results.hbonds : np.ndarray, shape (N, 6)
        Columns: [frame, donor_idx, hydrogen_idx, acceptor_idx,
                  d_a_distance, h_d_a_angle]
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        universe,
        donors_sel=None,
        hydrogens_sel=None,
        acceptors_sel=None,
        between=None,
        d_h_cutoff=1.2,
        d_a_cutoff=3.5,
        h_d_a_angle_cutoff=30.0,
        update_selections=True,
    ):
        self.u = universe
        self._trajectory = self.u.trajectory

        # Strip whitespace from selection strings
        def _clean(s):
            return s.strip() if isinstance(s, str) else s

        self._donors_sel = _clean(donors_sel)
        self._hydrogens_sel = _clean(hydrogens_sel)
        self._acceptors_sel = _clean(acceptors_sel)

        # Warn on empty selection strings
        _msg = (
            "{} is an empty selection string – no hydrogen bonds will be "
            "found.  This may be intended, but please check your selection."
        )
        for _name in ("donors_sel", "hydrogens_sel", "acceptors_sel"):
            _val = getattr(self, _name)
            if isinstance(_val, str) and not _val:
                warnings.warn(_msg.format(_name))

        # ---- 'between' groups ----
        if between is not None:
            if not isinstance(between, Iterable) or len(between) == 0:
                raise ValueError("'between' must be a non-empty list/iterable")
            if isinstance(between[0], str):
                between = [between]
            self.between_ags = [
                [
                    self.u.select_atoms(g1, updating=False),
                    self.u.select_atoms(g2, updating=False),
                ]
                for g1, g2 in between
            ]
        else:
            self.between_ags = None

        # ---- geometric parameters ----
        self.d_h_cutoff = d_h_cutoff
        self.d_a_cutoff = d_a_cutoff
        self.h_d_a_angle = h_d_a_angle_cutoff   # angle AT THE DONOR, < threshold
        self.update_selections = update_selections

        # Results container
        self.results = Results()
        self.results.hbonds = None

        # ---- resolve atom selections ----
        if self._acceptors_sel is None:
            self._acceptors_sel = self.guess_acceptors()
        if self._hydrogens_sel is None:
            self._hydrogens_sel = self.guess_hydrogens()

        self._acceptors = self.u.select_atoms(
            self.acceptors_sel, updating=self.update_selections
        )
        self._donors, self._hydrogens = self._get_dh_pairs()

    # ------------------------------------------------------------------
    # Atom-selection helpers  (unchanged from MDAnalysis original)
    # ------------------------------------------------------------------

    def guess_hydrogens(self, select="all", max_mass=1.1, min_charge=0.3, min_mass=0.9):
        """Guess hydrogen atoms by mass and charge.

        Returns a selection string of atoms with mass in (min_mass, max_mass)
        and charge > min_charge.
        """
        if min_mass >= max_mass:
            raise ValueError("min_mass must be less than max_mass")
        ag = self.u.select_atoms(select)
        mask = np.logical_and.reduce(
            (ag.masses < max_mass, ag.charges > min_charge, ag.masses > min_mass)
        )
        return self._group_categories(ag[mask])

    def guess_donors(self, select="all", max_charge=-0.5):
        """Guess donor heavy atoms by charge.

        Only use when the topology lacks bond information; otherwise use the
        default ``donors_sel=None`` to resolve pairs from the bond graph.
        """
        if self.hydrogens_sel is None:
            hydrogens_sel = self.guess_hydrogens()
        else:
            hydrogens_sel = self.hydrogens_sel
        hydrogens_ag = self.u.select_atoms(hydrogens_sel)

        if (
            hasattr(self.u._topology, "bonds")
            and len(self.u._topology.bonds.values) != 0
        ):
            donors_ag = find_hydrogen_donors(hydrogens_ag)
            donors_ag = donors_ag.intersection(self.u.select_atoms(select))
        else:
            donors_ag = hydrogens_ag.residues.atoms.select_atoms(
                "({donors_sel}) and around {d_h_cutoff} {hydrogens_sel}".format(
                    donors_sel=select,
                    d_h_cutoff=self.d_h_cutoff,
                    hydrogens_sel=hydrogens_sel,
                )
            )

        donors_ag = donors_ag[donors_ag.charges < max_charge]
        return self._group_categories(donors_ag)

    def guess_acceptors(self, select="all", max_charge=-0.5):
        """Guess acceptor heavy atoms by partial charge (charge < max_charge)."""
        ag = self.u.select_atoms(select)
        return self._group_categories(ag[ag.charges < max_charge])

    @staticmethod
    def _group_categories(group):
        """Build an MDAnalysis selection string from an AtomGroup.

        Uses '(resname X and name Y)' pairs when available, else 'type Z'.
        """
        if hasattr(group, "resnames") and hasattr(group, "names"):
            entries = np.unique(
                [
                    "(resname {} and name {})".format(r, n)
                    for r, n in zip(group.resnames, group.names)
                ]
            )
        else:
            entries = np.unique(["type {}".format(t) for t in group.types])
        return " or ".join(entries)

    # ------------------------------------------------------------------
    # Donor-hydrogen pair identification
    # ------------------------------------------------------------------

    def _get_dh_pairs(self):
        """Return aligned (donors, hydrogens) AtomGroups.

        If ``donors_sel`` is None, pairs are resolved from the bond graph.
        Otherwise a distance cutoff ``d_h_cutoff`` is used.
        """
        if self.donors_sel is None:
            # Topology path – fast via _topology.bonds
            if not (
                hasattr(self.u._topology, "bonds")
                and len(self.u._topology.bonds.values) != 0
            ):
                raise NoDataError(
                    "No bond information in topology.  Either load a topology "
                    "with bonds (PSF/TPR/PRMTOP), call guess_bonds(), or set "
                    "donors_sel so that a distance cutoff can be used instead."
                )
            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = (
                sum(h.bonded_atoms[0] for h in hydrogens)
                if hydrogens
                else AtomGroup([], self.u)
            )
        else:
            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = self.u.select_atoms(self.donors_sel)
            d_idx, h_idx = capped_distance(
                donors.positions,
                hydrogens.positions,
                max_cutoff=self.d_h_cutoff,
                box=self.u.dimensions,
                return_distances=False,
            ).T
            donors = donors[d_idx]
            hydrogens = hydrogens[h_idx]

        return donors, hydrogens

    # ------------------------------------------------------------------
    # 'between' filter
    # ------------------------------------------------------------------

    def _filter_atoms(self, donors, acceptors):
        """Return a boolean mask selecting D-A pairs that belong to the
        requested ``between`` group combinations.

        Parameters
        ----------
        donors : AtomGroup
        acceptors : AtomGroup  (same length, element-wise paired with donors)

        Returns
        -------
        mask : np.ndarray, dtype bool
        """
        mask = np.zeros(donors.n_atoms, dtype=bool)
        for g1, g2 in self.between_ags:
            # donors in G1 & acceptors in G2
            mask |= np.logical_and(
                np.isin(donors.indices, g1.indices),
                np.isin(acceptors.indices, g2.indices),
            )
            # donors in G2 & acceptors in G1
            mask |= np.logical_and(
                np.isin(donors.indices, g2.indices),
                np.isin(acceptors.indices, g1.indices),
            )
        return mask

    # ------------------------------------------------------------------
    # AnalysisBase protocol
    # ------------------------------------------------------------------

    def _prepare(self):
        # Six lists: frame, donor, hydrogen, acceptor, distance, angle
        self.results.hbonds = [[], [], [], [], [], []]

    def _single_frame(self):
        box = self._ts.dimensions

        if self.update_selections:
            self._donors, self._hydrogens = self._get_dh_pairs()

        # ---- Step 1: candidate D-A pairs within distance cutoff ----
        d_a_indices, d_a_distances = capped_distance(
            self._donors.positions,
            self._acceptors.positions,
            max_cutoff=self.d_a_cutoff,
            min_cutoff=1.0,          # atoms cannot bond with themselves
            box=box,
            return_distances=True,
        )

        if np.size(d_a_indices) == 0:
            warnings.warn(
                f"No donor-acceptor pairs within {self.d_a_cutoff} Å at "
                f"frame {self._ts.frame}.  Check your selections and cutoff."
            )
            return

        # Gather aligned temporary AtomGroups
        tmp_donors    = self._donors[d_a_indices[:, 0]]
        tmp_hydrogens = self._hydrogens[d_a_indices[:, 0]]
        tmp_acceptors = self._acceptors[d_a_indices[:, 1]]

        # ---- Step 2: apply 'between' group filter ----
        if self.between_ags is not None:
            between_mask  = self._filter_atoms(tmp_donors, tmp_acceptors)
            tmp_donors    = tmp_donors[between_mask]
            tmp_hydrogens = tmp_hydrogens[between_mask]
            tmp_acceptors = tmp_acceptors[between_mask]
            d_a_distances = d_a_distances[between_mask]

        if tmp_donors.n_atoms == 0:
            return

        # ---- Step 3: compute H-D-A angle (at the donor) ----
        #
        #  calc_angles(A, B, C) computes the A-B-C angle at vertex B.
        #  We want the angle at D between the incoming H and the outgoing A,
        #  i.e. the H-D-A angle, so B = donor.
        #
        h_d_a_angles = np.rad2deg(
            calc_angles(
                tmp_hydrogens.positions,   # A
                tmp_donors.positions,      # B (vertex = donor)
                tmp_acceptors.positions,   # C
                box=box,
            )
        )

        # Accept bonds where the angle is LESS THAN the cutoff
        hbond_mask = h_d_a_angles < self.h_d_a_angle
        hbond_indices = np.where(hbond_mask)[0]

        if hbond_indices.size == 0:
            return

        # ---- Step 4: store results ----
        hbond_donors    = tmp_donors[hbond_indices]
        hbond_hydrogens = tmp_hydrogens[hbond_indices]
        hbond_acceptors = tmp_acceptors[hbond_indices]
        hbond_distances = d_a_distances[hbond_indices]
        hbond_angles    = h_d_a_angles[hbond_indices]

        self.results.hbonds[0].extend(
            np.full(len(hbond_donors), self._ts.frame, dtype=int)
        )
        self.results.hbonds[1].extend(hbond_donors.indices)
        self.results.hbonds[2].extend(hbond_hydrogens.indices)
        self.results.hbonds[3].extend(hbond_acceptors.indices)
        self.results.hbonds[4].extend(hbond_distances)
        self.results.hbonds[5].extend(hbond_angles)

    def _conclude(self):
        self.results.hbonds = np.asarray(self.results.hbonds, dtype=float).T

    def _get_aggregator(self):
        """Support parallel execution via multiprocessing/dask backends."""
        return ResultsGroup(lookup={"hbonds": ResultsGroup.ndarray_hstack})

    # ------------------------------------------------------------------
    # Backward-compatibility alias
    # ------------------------------------------------------------------

    @property
    def hbonds(self):
        """Deprecated alias for ``results.hbonds``."""
        warnings.warn(
            "The `hbonds` attribute is deprecated. Use `results.hbonds`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.results.hbonds

    # ------------------------------------------------------------------
    # Selection properties (kept for API parity with MDAnalysis original)
    # ------------------------------------------------------------------

    @property
    def donors_sel(self):
        return self._donors_sel

    @donors_sel.setter
    def donors_sel(self, value):
        self._donors_sel = value
        self._donors, self._hydrogens = self._get_dh_pairs()

    @property
    def hydrogens_sel(self):
        return self._hydrogens_sel

    @hydrogens_sel.setter
    def hydrogens_sel(self, value):
        self._hydrogens_sel = value
        if self._hydrogens_sel is None:
            self._hydrogens_sel = self.guess_hydrogens()
        self._donors, self._hydrogens = self._get_dh_pairs()

    @property
    def acceptors_sel(self):
        return self._acceptors_sel

    @acceptors_sel.setter
    def acceptors_sel(self, value):
        self._acceptors_sel = value
        if self._acceptors_sel is None:
            self._acceptors_sel = self.guess_acceptors()
        self._acceptors = self.u.select_atoms(
            self._acceptors_sel, updating=self.update_selections
        )

    # ------------------------------------------------------------------
    # Original analysis methods (unchanged logic, compatible with new results)
    # ------------------------------------------------------------------

    def lifetime(self, tau_max=20, window_step=1, intermittency=0):
        """Compute the hydrogen-bond lifetime autocorrelation.

        Unique bonds are identified by the (hydrogen, acceptor) pair.
        See ``MDAnalysis.lib.correlations`` for the underlying functions.

        Parameters
        ----------
        tau_max : int
            Autocorrelation is computed for lag times 1 … tau_max frames.
        window_step : int
            Stride for selecting t₀ origins.
        intermittency : int
            Number of consecutive missing frames still counted as "present".
            0 = strict continuous autocorrelation.

        Returns
        -------
        np.ndarray, shape (2, tau_max)
            Row 0: tau values.  Row 1: C(tau) values.
        """
        self._check_run_called()

        if self.step != 1:
            warnings.warn(
                "Hydrogen bonds were computed with step > 1.  Lifetime "
                "calculation may be unreliable; consider re-running with "
                "step=1 and using the intermittency parameter."
            )

        found_hydrogen_bonds = [set() for _ in self.frames]
        for frame_index, frame in enumerate(self.frames):
            frame_hbonds = self.results.hbonds[self.results.hbonds[:, 0] == frame]
            for hbond in frame_hbonds:
                # Identify bond by (hydrogen_idx, acceptor_idx)
                found_hydrogen_bonds[frame_index].add(frozenset(hbond[2:4]))

        intermittent_hbonds = correct_intermittency(
            found_hydrogen_bonds, intermittency=intermittency
        )
        tau_ts, ts, _ = autocorrelation(
            intermittent_hbonds, tau_max, window_step=window_step
        )
        return np.vstack([tau_ts, ts])

    def count_by_time(self):
        """Total number of hydrogen bonds found at each trajectory frame.

        Returns
        -------
        np.ndarray, shape (n_frames,)
            Counts ordered by ``self.frames``.
        """
        self._check_run_called()
        hbond_frames = self.results.hbonds[:, 0].astype(int)
        frame_unique, frame_counts = np.unique(hbond_frames, return_counts=True)
        frame_min, frame_max = self.frames.min(), self.frames.max()
        counts = np.zeros(frame_max - frame_min + 1, dtype=int)
        counts[frame_unique - frame_min] = frame_counts
        return counts[self.frames - frame_min]

    def count_by_type(self):
        """Count hydrogen bonds grouped by donor/acceptor residue and atom type.

        Returns
        -------
        np.ndarray
            Columns: donor_resname:type, acceptor_resname:type, count.
        """
        self._check_run_called()
        d = self.u.atoms[self.results.hbonds[:, 1].astype(np.intp)]
        a = self.u.atoms[self.results.hbonds[:, 3].astype(np.intp)]

        d_res = d.resnames if hasattr(d, "resnames") else ["None"] * len(d.types)
        a_res = a.resnames if hasattr(a, "resnames") else ["None"] * len(a.types)

        tmp = np.array([d_res, d.types, a_res, a.types], dtype=str).T
        hbond_type, type_counts = np.unique(tmp, axis=0, return_counts=True)
        return np.array(
            [
                [":".join(ht[:2]), ":".join(ht[2:4]), cnt]
                for ht, cnt in zip(hbond_type, type_counts)
            ]
        )

    def count_by_ids(self):
        """Count hydrogen bonds for each unique (donor, hydrogen, acceptor) triple.

        Returns
        -------
        np.ndarray
            Columns: donor_id, hydrogen_id, acceptor_id, count.
            Sorted by frequency (most frequent first).
        """
        self._check_run_called()
        d = self.u.atoms[self.results.hbonds[:, 1].astype(np.intp)]
        h = self.u.atoms[self.results.hbonds[:, 2].astype(np.intp)]
        a = self.u.atoms[self.results.hbonds[:, 3].astype(np.intp)]

        tmp = np.array([d.ids, h.ids, a.ids]).T
        hbond_ids, ids_counts = np.unique(tmp, axis=0, return_counts=True)
        unique_hbonds = np.concatenate([hbond_ids, ids_counts[:, None]], axis=1)
        return unique_hbonds[unique_hbonds[:, 3].argsort()[::-1]]

    # ------------------------------------------------------------------
    # Hydrogen-bond network analysis
    # (new methods for network rearrangement studies)
    # ------------------------------------------------------------------

    def get_network_at_frame(self, frame):
        """Return the hydrogen-bond network at a single trajectory frame.

        Parameters
        ----------
        frame : int
            Frame index (as stored in ``results.hbonds[:, 0]``).

        Returns
        -------
        dict
            ``{donor_atom_index: set(acceptor_atom_index, ...)}``
            Each donor maps to the set of all acceptors it hydrogen-bonds to
            at this frame.
        """
        self._check_run_called()
        mask = self.results.hbonds[:, 0] == frame
        frame_data = self.results.hbonds[mask]
        network = {}
        for row in frame_data:
            d_idx = int(row[1])
            a_idx = int(row[3])
            network.setdefault(d_idx, set()).add(a_idx)
        return network

    def get_connectivity_changes(self):
        """Identify hydrogen bonds formed and broken between consecutive frames.

        Returns
        -------
        dict
            Keys are frame indices (starting from the *second* frame in
            ``self.frames``).  Each value is a dict with:

            * ``"formed"`` – list of ``(donor_idx, hydrogen_idx, acceptor_idx)``
              tuples that appeared at this frame but were absent at the previous.
            * ``"broken"`` – list of the same format for bonds that disappeared.
        """
        self._check_run_called()
        changes = {}
        hb = self.results.hbonds

        # Build a set of (donor, hydrogen, acceptor) triples per frame
        def _triples(frame):
            mask = hb[:, 0] == frame
            return {
                (int(r[1]), int(r[2]), int(r[3]))
                for r in hb[mask]
            }

        for i in range(1, len(self.frames)):
            prev_frame = self.frames[i - 1]
            curr_frame = self.frames[i]
            prev_set = _triples(prev_frame)
            curr_set = _triples(curr_frame)
            changes[curr_frame] = {
                "formed": list(curr_set - prev_set),
                "broken": list(prev_set - curr_set),
            }
        return changes

    def count_hbond_events(self):
        """Count formation and breaking events frame by frame.

        Returns
        -------
        np.ndarray, shape (n_frames - 1, 3)
            Columns: [frame, n_formed, n_broken]
            Starting from the second frame in ``self.frames``.
        """
        changes = self.get_connectivity_changes()
        rows = [
            [frame, len(v["formed"]), len(v["broken"])]
            for frame, v in sorted(changes.items())
        ]
        return np.array(rows, dtype=int)

    def get_hbond_matrix(self, frame, atom_indices=None):
        """Return a symmetric adjacency matrix for a given frame.

        Bonds are counted as undirected edges (D-A and A-D both set to True).

        Parameters
        ----------
        frame : int
        atom_indices : array-like or None
            If provided, only rows/columns for these atom indices are included
            and the matrix is indexed 0…len(atom_indices)-1.  If None, the
            matrix covers all atoms in the universe.

        Returns
        -------
        np.ndarray, shape (N, N), dtype bool
        """
        self._check_run_called()
        mask = self.results.hbonds[:, 0] == frame
        frame_data = self.results.hbonds[mask]

        if atom_indices is None:
            n = self.u.atoms.n_atoms
            idx_map = None
        else:
            atom_indices = np.asarray(atom_indices)
            n = len(atom_indices)
            idx_map = {orig: new for new, orig in enumerate(atom_indices)}

        mat = np.zeros((n, n), dtype=bool)
        for row in frame_data:
            d_idx = int(row[1])
            a_idx = int(row[3])
            if idx_map is not None:
                if d_idx not in idx_map or a_idx not in idx_map:
                    continue
                d_idx = idx_map[d_idx]
                a_idx = idx_map[a_idx]
            mat[d_idx, a_idx] = True
            mat[a_idx, d_idx] = True  # symmetric

        return mat

    # ------------------------------------------------------------------
    # Reorientation dynamics helpers
    # (new methods for O-H reorientation / extended jump model analyses)
    # ------------------------------------------------------------------

    def find_hbond_switches(self):
        """Locate frames at which a donor-hydrogen pair changes its acceptor.

        This is the elementary 'jump' event in the Laage-Hynes extended jump
        model of water reorientation dynamics.

        Returns
        -------
        list of dict
            One entry per detected switch event.  Each dict contains:

            ``"frame"``
                Frame at which the new bond was first detected.
            ``"prev_frame"``
                Preceding frame.
            ``"donor_idx"``
                Atom index of the donor (0-based).
            ``"hydrogen_idx"``
                Atom index of the hydrogen (0-based).
            ``"old_acceptor_idx"``
                Atom index of the acceptor in the previous frame.
            ``"new_acceptor_idx"``
                Atom index of the new acceptor.
            ``"jump_angle_deg"``
                Angle (in degrees) between the old D→A and new D→A unit
                vectors, computed at the frame of the switch from the stored
                trajectory positions.

        Notes
        -----
        * Only switches within a *continuous* H-bond (same D-H pair present
          in both frames) are reported.
        * Jump angle calculation re-reads trajectory positions; it is
          therefore most accurate when ``update_selections=True`` was used.
        * PBC-minimum-image convention is applied for orthorhombic boxes.
        """
        self._check_run_called()
        hb = self.results.hbonds

        # Build per-frame mapping: (donor_idx, hydrogen_idx) -> acceptor_idx
        def _dh_to_a(frame):
            mask = hb[:, 0] == frame
            return {
                (int(r[1]), int(r[2])): int(r[3])
                for r in hb[mask]
            }

        events = []
        for i in range(1, len(self.frames)):
            prev_f = self.frames[i - 1]
            curr_f = self.frames[i]
            prev_map = _dh_to_a(prev_f)
            curr_map = _dh_to_a(curr_f)

            # D-H pairs present in both frames
            common_pairs = set(prev_map) & set(curr_map)
            for dh_pair in common_pairs:
                old_acc = prev_map[dh_pair]
                new_acc = curr_map[dh_pair]
                if old_acc == new_acc:
                    continue  # no switch

                d_idx, h_idx = dh_pair

                # Compute jump angle from positions at curr_f
                self.u.trajectory[curr_f]
                box = self.u.trajectory.ts.dimensions
                d_pos  = self.u.atoms[d_idx].position
                oa_pos = self.u.atoms[old_acc].position
                na_pos = self.u.atoms[new_acc].position

                vec_old = _vector_with_pbc(d_pos, oa_pos, box)
                vec_new = _vector_with_pbc(d_pos, na_pos, box)
                norm_old = np.linalg.norm(vec_old)
                norm_new = np.linalg.norm(vec_new)

                if norm_old > 0 and norm_new > 0:
                    cos_theta = np.dot(vec_old, vec_new) / (norm_old * norm_new)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    jump_angle = np.degrees(np.arccos(cos_theta))
                else:
                    jump_angle = float("nan")

                events.append(
                    {
                        "frame":            curr_f,
                        "prev_frame":       prev_f,
                        "donor_idx":        d_idx,
                        "hydrogen_idx":     h_idx,
                        "old_acceptor_idx": old_acc,
                        "new_acceptor_idx": new_acc,
                        "jump_angle_deg":   jump_angle,
                    }
                )
        return events

    def get_donor_hydrogen_vectors(self, frame=None):
        """Return D→H unit vectors for every hydrogen bond at one or all frames.

        Parameters
        ----------
        frame : int or None
            If an integer, return vectors only for that trajectory frame.
            If None, return vectors for all frames (may be large).

        Returns
        -------
        np.ndarray, shape (N, 3)
            Each row is the D→H unit vector for one hydrogen bond record.
            Corresponding rows in ``results.hbonds`` give the atom indices
            and frame.
        """
        self._check_run_called()
        hb = self.results.hbonds

        if frame is not None:
            hb = hb[hb[:, 0] == frame]

        vectors = np.empty((len(hb), 3), dtype=float)
        current_frame = None
        for i, row in enumerate(hb):
            f = int(row[0])
            if f != current_frame:
                self.u.trajectory[f]
                box = self.u.trajectory.ts.dimensions
                current_frame = f
            d_pos = self.u.atoms[int(row[1])].position
            h_pos = self.u.atoms[int(row[2])].position
            v = _vector_with_pbc(d_pos, h_pos, box)
            norm = np.linalg.norm(v)
            vectors[i] = v / norm if norm > 0 else v
        return vectors

    def get_donor_acceptor_vectors(self, frame=None):
        """Return D→A unit vectors for every hydrogen bond at one or all frames.

        Parameters and return value are identical in shape to
        ``get_donor_hydrogen_vectors``.
        """
        self._check_run_called()
        hb = self.results.hbonds

        if frame is not None:
            hb = hb[hb[:, 0] == frame]

        vectors = np.empty((len(hb), 3), dtype=float)
        current_frame = None
        for i, row in enumerate(hb):
            f = int(row[0])
            if f != current_frame:
                self.u.trajectory[f]
                box = self.u.trajectory.ts.dimensions
                current_frame = f
            d_pos = self.u.atoms[int(row[1])].position
            a_pos = self.u.atoms[int(row[3])].position
            v = _vector_with_pbc(d_pos, a_pos, box)
            norm = np.linalg.norm(v)
            vectors[i] = v / norm if norm > 0 else v
        return vectors

    def compute_oh_acf(self, donor_idx, hydrogen_idx, l=2):
        """Compute the Legendre-polynomial autocorrelation function of the
        O-H bond vector for a specified donor-hydrogen pair.

        This is the rotational time-correlation function::

            C_l(t) = <P_l( û(0) · û(t) )>

        where û is the unit D→H vector and P_l is the l-th Legendre polynomial.
        l=2 corresponds to what is measured by NMR or IR experiments.

        Parameters
        ----------
        donor_idx : int
            Atom index (0-based) of the donor.
        hydrogen_idx : int
            Atom index (0-based) of the hydrogen.
        l : int
            Legendre polynomial order.  Default 2.

        Returns
        -------
        tau_array : np.ndarray
            Lag times in frames (0 … n_frames-1).
        acf : np.ndarray
            C_l(tau) for each lag time.

        Notes
        -----
        This function re-reads trajectory positions and is independent of
        whether the D-H pair was ever detected as a hydrogen bond.
        """
        if l not in (1, 2):
            raise ValueError("Only l=1 and l=2 Legendre polynomials are supported.")

        vectors = []
        for ts in self.u.trajectory[self.start:self.stop:self.step]:
            box = ts.dimensions
            d_pos = self.u.atoms[donor_idx].position
            h_pos = self.u.atoms[hydrogen_idx].position
            v = _vector_with_pbc(d_pos, h_pos, box)
            norm = np.linalg.norm(v)
            vectors.append(v / norm if norm > 0 else v)
        vectors = np.array(vectors)  # (n_frames, 3)

        n = len(vectors)
        acf = np.zeros(n)
        counts = np.zeros(n, dtype=int)

        for tau in range(n):
            dots = np.einsum("ij,ij->i", vectors[: n - tau], vectors[tau:])
            dots = np.clip(dots, -1.0, 1.0)
            if l == 1:
                acf[tau] = np.mean(dots)
            else:  # l == 2
                acf[tau] = np.mean(0.5 * (3.0 * dots**2 - 1.0))
            counts[tau] = n - tau

        return np.arange(n, dtype=float), acf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_run_called(self):
        """Raise ``NoDataError`` if ``run()`` has not been called yet."""
        if self.results.hbonds is None or len(self.results.hbonds) == 0:
            raise NoDataError(
                "No hydrogen bond data found.  Call run() before using "
                "analysis or post-processing methods."
            )
