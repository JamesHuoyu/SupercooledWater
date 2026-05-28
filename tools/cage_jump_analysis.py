# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Cage-Jump Analysis for Supercooled Water
==========================================

Implements the cage-jump model of Kikutsuji, Kim & Matubayasi,
*J. Chem. Phys.* **150**, 204502 (2019), using completed
``HydrogenBondAnalysis`` (and optionally ``ZetaOrderParameter``) objects
as input.

State-machine algorithm (Section II of the paper)
--------------------------------------------------
Each water molecule's trajectory is classified into alternating
**Caged (C)** and **Jumping (J)** states:

  C state
    Entered when the molecule is simultaneously H-bonded to exactly
    *n_hb_cage* (default 4) partner oxygens.  The identity of those
    partners is frozen at entry.
    Ends ("C→J") when **all** *n_hb_cage* original C-state partners
    are simultaneously absent from the current H-bond list.

  J state
    Entered at the C→J transition.  The O-atom position at entry is
    recorded as the jump origin.
    Continues even if old partners re-bond — the criterion for exit
    is purely about NEW partners.
    Ends ("J→C") when the molecule is simultaneously bonded to exactly
    *n_hb_cage* partners **none of which** appeared in the previous C
    state.  The displacement |Δr_J| between entry and exit positions
    is recorded as the jump length.

Observables computed
--------------------
  Per-molecule per-episode:
    τ_C, τ_J   – duration of each C / J episode (ps)
    r_J        – O-atom displacement during J episode (Å)

  Distributions:
    P(τ_C), P(τ_J), P(r_J)

  Averages:
    ⟨τ_C⟩, ⟨τ_J⟩, ⟨r²_J⟩

  ρ_J          – fraction of time in J state (averaged per-molecule
                 first, then over molecules – see paper)

  JMSD(Θ_J)   – jumping mean-square displacement vs. jump count
  D_J          – self-diffusion from JMSD slope
  D_estimate   – ρ_J × ⟨r²_J⟩ / ⟨τ_J⟩  (cage-jump model prediction)

  H-bond ACF   – c(t) = ⟨h(0)h(t)⟩ / ⟨h(0)⟩  (history-independent)
  τ_HB         – lifetime from stretched-exponential fit to c(t)

  MSD          – mean-square displacement from O positions
  D_MSD        – self-diffusion from long-time MSD slope

  Optional: per-episode mean ζ from ZetaOrderParameter, allowing
  correlation of cage-jump events with local structural order.

Usage
-----
::

    from water.tools.custom_hbond_analysis import HydrogenBondAnalysis as HBA
    from water.tools.zeta_order_parameter   import ZetaOrderParameter  as ZOP
    from water.tools.cage_jump_analysis     import CageJumpAnalysis    as CJA

    hba = HBA(universe=u, donors_sel="type 1", hydrogens_sel="type 2",
              acceptors_sel="type 1", d_a_cutoff=3.5, h_d_a_angle_cutoff=30.0)
    hba.run()

    zop = ZOP(hba=hba, central_sel="type 1", shell_cutoff=6.0)
    zop.run()

    cja = CJA(hba=hba, zop=zop,      # zop is optional
              oxygen_sel="type 1",
              n_hb_cage=4,
              temperature=225.0)      # K – for D comparison
    cja.run()                         # pure post-processing, no traj re-read

    print(cja.results.D_estimate)     # Pa·s … m²/s
    cja.plot_all("cagejump_plots/")   # generate all paper figures
"""

import os
import warnings
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

from MDAnalysis.exceptions import NoDataError

logger = logging.getLogger(__name__)

# Physical constants
kB_SI      = 1.380649e-23   # J/K
ANG2_TO_M2 = 1e-20          # Å² → m²
PS_TO_S    = 1e-12          # ps → s


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------

@dataclass
class MoleculeTrajectory:
    """All cage-jump episodes extracted from one molecule's history."""
    atom_idx:    int
    tau_C:       List[float]  = field(default_factory=list)   # ps
    tau_J:       List[float]  = field(default_factory=list)   # ps
    r_J:         List[float]  = field(default_factory=list)   # Å
    rho_J:       float        = np.nan                        # fraction of J time
    # Optional: mean ζ at start/end of each J episode
    zeta_C_mean: List[float]  = field(default_factory=list)
    zeta_J_mean: List[float]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# CageJumpAnalysis
# ---------------------------------------------------------------------------

class CageJumpAnalysis:
    """Cage-jump model analysis following Kikutsuji et al. (2019).

    Parameters
    ----------
    hba : HydrogenBondAnalysis
        Completed HBA object (``hba.run()`` already called).
    zop : ZetaOrderParameter or None
        Completed ZOP object for optional structural correlation.
    oxygen_sel : str
        MDAnalysis selection for oxygen atoms (one per molecule).
    n_hb_cage : int
        Number of simultaneous H-bonds that defines the caged state.
        Default 4 (tetrahedral water).
    temperature : float
        System temperature in K (used for labels / comparison only).
    """

    def __init__(
        self,
        hba,
        zop=None,
        oxygen_sel:  str   = "type 1",
        n_hb_cage:   int   = 4,
        temperature: float = 300.0,
    ):
        # ---- Validate inputs -------------------------------------------------
        if hba.results.hbonds is None or len(hba.results.hbonds) == 0:
            raise NoDataError("hba.results.hbonds is empty – call hba.run() first.")
        if zop is not None and (not hasattr(zop.results, "zeta")
                                or zop.results.zeta is None):
            raise NoDataError("zop.results.zeta is empty – call zop.run() first.")

        self.hba         = hba
        self.zop         = zop
        self.u           = hba.u
        self.oxygen_sel  = oxygen_sel
        self.n_hb_cage   = n_hb_cage
        self.T           = temperature
        self.dt          = self.u.trajectory.dt   # ps

        self._oxygen_ag   = self.u.select_atoms(oxygen_sel)
        self._ow_indices  = self._oxygen_ag.indices          # (N_OW,)
        self._ow_idx_to_col = {int(idx): i
                               for i, idx in enumerate(self._ow_indices)}
        self.frames        = hba.frames   # same frame range as HBA

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Run all analyses in sequence."""
        print("Step 1/5  Building per-frame H-bond partner lookup …")
        self._build_partner_lookup()

        print("Step 2/5  Reading O-atom positions from trajectory …")
        self._build_position_lookup()

        print("Step 3/5  Running per-molecule state machine …")
        self._run_state_machine()

        print("Step 4/5  Computing aggregate statistics …")
        self._compute_statistics()

        print("Step 5/5  Computing MSD and H-bond ACF …")
        self._compute_msd()
        self._compute_hbond_acf()

        if self.zop is not None:
            print("         Correlating with ζ order parameter …")
            self._correlate_with_zeta()

        self._print_summary()
        return self

    # ------------------------------------------------------------------
    # Step 1: build partner lookup
    # ------------------------------------------------------------------

    def _build_partner_lookup(self):
        """
        Build: self._partners[frame][ow_atom_idx] = frozenset(partner_ow_indices)

        From hba.results.hbonds we know donor/acceptor atom indices which are
        OW atoms. Both directions of each bond are registered so every OW
        knows all its HB partners regardless of whether it donated or accepted.
        """
        hb = self.hba.results.hbonds
        ow_set = set(self._ow_indices.tolist())

        partners: Dict[int, Dict[int, set]] = {}
        for row in hb:
            frame   = int(row[0])
            d_idx   = int(row[1])   # donor OW
            a_idx   = int(row[3])   # acceptor OW
            if frame not in partners:
                partners[frame] = {}
            partners[frame].setdefault(d_idx, set()).add(a_idx)
            partners[frame].setdefault(a_idx, set()).add(d_idx)

        # Convert inner sets to frozensets for fast intersection
        self._partners: Dict[int, Dict[int, frozenset]] = {
            frame: {idx: frozenset(s) for idx, s in mol_dict.items()}
            for frame, mol_dict in partners.items()
        }
        self._empty_fs = frozenset()

    # ------------------------------------------------------------------
    # Step 2: build position lookup
    # ------------------------------------------------------------------

    def _build_position_lookup(self):
        """
        Build: self._positions[frame_idx] = np.ndarray shape (N_OW, 3)
        stored as a dict {frame: array} to match the HBA frame set.
        """
        self._positions: Dict[int, np.ndarray] = {}
        for ts in self.u.trajectory[
            self.frames[0]: self.frames[-1] + 1:
            (self.frames[1] - self.frames[0]) if len(self.frames) > 1 else 1
        ]:
            if ts.frame in set(self.frames.tolist()):
                self._positions[ts.frame] = self._oxygen_ag.positions.copy()

    # ------------------------------------------------------------------
    # Step 3: per-molecule state machine
    # ------------------------------------------------------------------

    def _run_state_machine(self):
        """Classify each OW trajectory into alternating C / J episodes."""
        self.mol_trajectories: List[MoleculeTrajectory] = []

        for col_i, ow_idx in enumerate(self._ow_indices):
            mt = self._classify_molecule(ow_idx, col_i)
            self.mol_trajectories.append(mt)

    def _classify_molecule(self, ow_idx: int, col_i: int) -> MoleculeTrajectory:
        """
        Run the paper's state machine for a single oxygen atom.

        State machine (verbatim from paper §II):
        -----------------------------------------
        C state  – molecule is H-bonded to n_hb_cage specific partners.
                   Records those partners at entry.
                   Ends when ALL recorded partners simultaneously absent.

        J state  – entered at C→J transition.
                   Records O position at entry (= jump origin).
                   Ends when n_hb_cage bonds formed with partners NONE
                   of which were in the previous C state.
        """
        mt = MoleculeTrajectory(atom_idx=ow_idx)
        n  = self.n_hb_cage

        # State variables
        state          = "INIT"    # INIT → C or J
        c_partners     = None      # frozenset of partners at C entry
        c_start_frame  = None
        j_start_frame  = None
        j_start_pos    = None
        prev_c_partners = None     # to detect C→J→C with full partner change

        frames_list = self.frames.tolist()

        for frame in frames_list:
            current = self._partners.get(frame, {}).get(ow_idx, self._empty_fs)
            pos     = self._positions[frame][col_i]

            # ---- INIT: wait for first n-coordinated moment ----------------
            if state == "INIT":
                if len(current) == n:
                    state         = "C"
                    c_partners    = current
                    c_start_frame = frame
                continue

            # ---- C state --------------------------------------------------
            elif state == "C":
                # C→J condition: all original C-state partners simultaneously gone
                still_bonded = c_partners & current   # intersection
                if len(still_bonded) == 0:
                    # Record C episode
                    tau_c = (frame - c_start_frame) * self.dt
                    mt.tau_C.append(tau_c)

                    # Enter J state
                    state            = "J"
                    j_start_frame    = frame
                    j_start_pos      = pos.copy()
                    prev_c_partners  = c_partners     # remember to check J→C
                    c_partners       = None
                # else: remain in C; no action needed

            # ---- J state --------------------------------------------------
            elif state == "J":
                # J→C condition: exactly n bonds AND none with prev C partners
                if len(current) == n and len(current & prev_c_partners) == 0:
                    # Record J episode
                    tau_j  = (frame - j_start_frame) * self.dt
                    r_j    = float(np.linalg.norm(pos - j_start_pos))
                    mt.tau_J.append(tau_j)
                    mt.r_J.append(r_j)

                    # Enter new C state
                    state         = "C"
                    c_partners    = current
                    c_start_frame = frame
                # else: remain in J; old partners may reform, that is allowed

        # ---- ρ_J: fraction of J time for this molecule -------------------
        total_j_time = sum(mt.tau_J)
        total_c_time = sum(mt.tau_C)
        total_time   = total_j_time + total_c_time
        mt.rho_J     = total_j_time / total_time if total_time > 0 else np.nan

        return mt

    # ------------------------------------------------------------------
    # Step 4: aggregate statistics
    # ------------------------------------------------------------------

    def _compute_statistics(self):
        """Aggregate per-molecule results into system-wide distributions."""
        from types import SimpleNamespace
        r = SimpleNamespace()

        # ---- Collect all episodes ----------------------------------------
        all_tau_C = np.array([tc for mt in self.mol_trajectories
                              for tc in mt.tau_C], dtype=float)
        all_tau_J = np.array([tj for mt in self.mol_trajectories
                              for tj in mt.tau_J], dtype=float)
        all_r_J   = np.array([rj for mt in self.mol_trajectories
                              for rj in mt.r_J], dtype=float)
        all_rho_J = np.array([mt.rho_J for mt in self.mol_trajectories
                              if not np.isnan(mt.rho_J)], dtype=float)

        r.tau_C_all = all_tau_C    # (M_C,) ps – all individual C durations
        r.tau_J_all = all_tau_J    # (M_J,) ps
        r.r_J_all   = all_r_J     # (M_J,) Å

        # ---- First moments -----------------------------------------------
        r.mean_tau_C   = float(all_tau_C.mean()) if len(all_tau_C) else np.nan
        r.mean_tau_J   = float(all_tau_J.mean()) if len(all_tau_J) else np.nan
        r.mean_r2_J    = float((all_r_J ** 2).mean()) if len(all_r_J) else np.nan
        r.mean_rJ      = float(all_r_J.mean())  if len(all_r_J) else np.nan

        # ---- ρ_J (average over per-molecule fractions) -------------------
        r.rho_J        = float(all_rho_J.mean()) if len(all_rho_J) else np.nan
        # Cross-check: ρ_J vs ⟨τ_J⟩/(⟨τ_C⟩+⟨τ_J⟩) (inset Fig. 5)
        r.rho_J_mean_field = (r.mean_tau_J / (r.mean_tau_C + r.mean_tau_J)
                              if (r.mean_tau_C + r.mean_tau_J) > 0 else np.nan)

        # ---- D_J from ⟨r²_J⟩/⟨τ_J⟩ (Fig. 4b) --------------------------
        # D_J = ⟨r²_J⟩ / (6 × ⟨τ_J⟩) for 3-D random walk in m²/s
        r.D_J_ang2ps = (r.mean_r2_J / (6.0 * r.mean_tau_J)
                        if r.mean_tau_J > 0 else np.nan)
        r.D_J = r.D_J_ang2ps * ANG2_TO_M2 / PS_TO_S  # m²/s

        # ---- Cage-jump model estimate of D (Fig. 5) ----------------------
        # D ≈ ρ_J × D_J = ρ_J × ⟨r²_J⟩ / (6 × ⟨τ_J⟩)
        r.D_estimate_ang2ps = (r.rho_J * r.mean_r2_J / (6.0 * r.mean_tau_J)
                               if r.mean_tau_J > 0 else np.nan)
        r.D_estimate = r.D_estimate_ang2ps * ANG2_TO_M2 / PS_TO_S

        # ---- JMSD(Θ_J) (Fig. 4a) ----------------------------------------
        r.jmsd, r.jmsd_Theta = self._compute_jmsd()

        # ---- D_J from JMSD slope -----------------------------------------
        if len(r.jmsd) >= 3:
            slope, _ = np.polyfit(r.jmsd_Theta.astype(float),
                                  r.jmsd, 1)
            # JMSD = 2 D_J ⟨τ_J⟩ Θ_J  →  D_J = slope / (2 ⟨τ_J⟩)  [not 6, since JMSD not time]
            # Actually: ⟨δr²_J(Θ_J)⟩ = Θ_J ⟨r²_J⟩ → D_J = slope / (6⟨τ_J⟩)
            r.D_J_from_JMSD = (slope / (6.0 * r.mean_tau_J)
                               * ANG2_TO_M2 / PS_TO_S
                               if r.mean_tau_J > 0 else np.nan)
        else:
            r.D_J_from_JMSD = np.nan

        self.results = r

    def _compute_jmsd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute JMSD(Θ_J) = (1/N_J) Σᵢ Σ_{θ=1}^{Θ_J} |Δr^J_i(θ)|²

        Returns the cumulative JMSD as a function of jump count Θ_J,
        averaged over all molecules that have at least Θ_J jumps.
        """
        # Collect per-molecule squared jump lengths
        per_mol_r2 = [np.array(mt.r_J, dtype=float) ** 2
                      for mt in self.mol_trajectories
                      if len(mt.r_J) > 0]
        if not per_mol_r2:
            return np.array([]), np.array([])

        max_jumps = max(len(v) for v in per_mol_r2)
        jmsd   = np.zeros(max_jumps)
        counts = np.zeros(max_jumps, dtype=int)

        for r2_arr in per_mol_r2:
            cumsum = np.cumsum(r2_arr)
            jmsd[:len(cumsum)]   += cumsum
            counts[:len(cumsum)] += 1

        valid = counts > 0
        jmsd[valid] /= counts[valid]
        jmsd[~valid] = np.nan

        Theta = np.arange(1, max_jumps + 1)
        return jmsd, Theta

    # ------------------------------------------------------------------
    # Step 5a: MSD
    # ------------------------------------------------------------------

    def _compute_msd(self):
        """Compute O-atom MSD and extract D from long-time slope."""
        # Stack positions: shape (n_frames, N_OW, 3)
        pos_arr = np.array([self._positions[f] for f in self.frames])
        n_frames, n_atoms, _ = pos_arr.shape

        tau_max = min(n_frames // 2, 2000)
        msd = np.zeros(tau_max + 1)
        for tau in range(1, tau_max + 1):
            disp = pos_arr[tau:] - pos_arr[: n_frames - tau]
            msd[tau] = float((disp ** 2).sum(axis=2).mean())

        self.results.msd          = msd          # Å²
        self.results.msd_tau_ps   = np.arange(tau_max + 1) * self.dt

        # D from linear fit over 20–80 % of MSD range
        lo = tau_max // 5
        hi = 4 * tau_max // 5
        if hi > lo + 2:
            tau_s = self.results.msd_tau_ps[lo:hi] * PS_TO_S
            slope, _ = np.polyfit(tau_s, msd[lo:hi], 1)
            self.results.D_MSD = slope / 6.0 * ANG2_TO_M2 / PS_TO_S  # m²/s already scaled
        else:
            self.results.D_MSD = np.nan

    # ------------------------------------------------------------------
    # Step 5b: H-bond ACF
    # ------------------------------------------------------------------

    def _compute_hbond_acf(self):
        """
        Compute the history-independent H-bond ACF:
          c(t) = ⟨h(0) h(t)⟩ / ⟨h(0)⟩

        h(t) = 1 if the bond exists at time t (regardless of history).
        Uses only the OW-OW bonds already stored in hba.results.hbonds.
        """
        hb = self.hba.results.hbonds
        n_frames = len(self.frames)
        frame_to_fi = {int(f): i for i, f in enumerate(self.frames)}

        # Build a unique bond set per frame: frozenset of (donor, acceptor)
        # using sorted pairs so each bond appears once
        bond_sets: List[set] = [set() for _ in range(n_frames)]
        for row in hb:
            fi = frame_to_fi.get(int(row[0]))
            if fi is None:
                continue
            d, a = int(row[1]), int(row[3])
            bond_sets[fi].add((min(d, a), max(d, a)))

        tau_max = min(n_frames // 2, 2000)
        acf_num = np.zeros(tau_max + 1)
        acf_den = 0.0

        # Reference h(0): all bonds present at frame 0
        for fi0 in range(n_frames - 1):
            bonds0 = bond_sets[fi0]
            if not bonds0:
                continue
            h0_count = len(bonds0)
            acf_den += h0_count

            for tau in range(min(tau_max + 1, n_frames - fi0)):
                bonds_t = bond_sets[fi0 + tau]
                acf_num[tau] += len(bonds0 & bonds_t)

        if acf_den > 0:
            acf = acf_num / acf_den
            # Normalise so c(0) = 1
            if acf[0] > 0:
                acf /= acf[0]
        else:
            acf = np.zeros(tau_max + 1)

        tau_ps_acf = np.arange(tau_max + 1) * self.dt
        self.results.hbond_acf       = acf
        self.results.hbond_acf_tau_ps = tau_ps_acf

        # Fit c(t) to stretched exponential: exp(−(t/τ_HB)^β)
        try:
            def stretched_exp(t, tau, beta):
                return np.exp(-(t / tau) ** beta)

            # Fit where ACF > 0.05 to avoid noise in the tail
            mask = acf > 0.05
            if mask.sum() > 5:
                popt, _ = curve_fit(
                    stretched_exp, tau_ps_acf[mask], acf[mask],
                    p0=[self.results.mean_tau_C or 10.0, 0.8],
                    bounds=([0.1, 0.2], [1e5, 1.5]),
                    maxfev=20000,
                )
                self.results.tau_HB  = float(popt[0])   # ps
                self.results.beta_HB = float(popt[1])
            else:
                self.results.tau_HB  = np.nan
                self.results.beta_HB = np.nan
        except Exception as exc:
            warnings.warn(f"H-bond ACF fit failed: {exc}")
            self.results.tau_HB  = np.nan
            self.results.beta_HB = np.nan

    # ------------------------------------------------------------------
    # Optional: ζ correlation
    # ------------------------------------------------------------------

    def _correlate_with_zeta(self):
        """Attach mean ζ per C/J episode using ZOP results."""
        if self.zop is None:
            return

        zeta_arr = self.zop.results.zeta          # (n_zop_frames, N_OW)
        zop_fi_to_row = self.zop._frame_to_row    # {frame: row_idx}
        ow_to_zop_col = {
            int(idx): ci
            for ci, idx in enumerate(self.zop.results.central_indices)
        }

        def mean_zeta_range(ow_idx, f_start, f_end):
            col = ow_to_zop_col.get(ow_idx)
            if col is None:
                return np.nan
            vals = []
            for f in self.frames:
                if f < f_start or f >= f_end:
                    continue
                row = zop_fi_to_row.get(int(f))
                if row is not None:
                    v = zeta_arr[row, col]
                    if not np.isnan(v):
                        vals.append(v)
            return float(np.mean(vals)) if vals else np.nan

        # Reconstruct frame ranges from durations – requires re-running the
        # state machine with frame bookkeeping. We store episode frame ranges
        # as a separate lightweight re-run.
        self._attach_zeta_to_episodes(ow_to_zop_col, zeta_arr, zop_fi_to_row)

    def _attach_zeta_to_episodes(self, ow_to_col, zeta_arr, fi_map):
        """Lightweight re-scan to attach ζ means to C/J episodes."""
        n = self.n_hb_cage
        for mi, (ow_idx, mt) in enumerate(
            zip(self._ow_indices, self.mol_trajectories)
        ):
            col = ow_to_col.get(int(ow_idx))
            if col is None:
                continue

            state = "INIT"
            c_partners = None
            prev_c_partners = None
            ep_frames = []   # frames in current episode
            c_ep_idx = 0
            j_ep_idx = 0

            for frame in self.frames:
                current = self._partners.get(frame, {}).get(ow_idx, self._empty_fs)
                row = fi_map.get(int(frame))
                z = zeta_arr[row, col] if row is not None else np.nan

                if state == "INIT":
                    if len(current) == n:
                        state = "C"; c_partners = current; ep_frames = [z]
                elif state == "C":
                    ep_frames.append(z)
                    if len(c_partners & current) == 0:
                        # C episode ends
                        valid = [v for v in ep_frames if not np.isnan(v)]
                        mean_z = float(np.mean(valid)) if valid else np.nan
                        if c_ep_idx < len(mt.zeta_C_mean):
                            mt.zeta_C_mean[c_ep_idx] = mean_z
                        else:
                            mt.zeta_C_mean.append(mean_z)
                        c_ep_idx += 1
                        state = "J"; prev_c_partners = c_partners; ep_frames = [z]
                elif state == "J":
                    ep_frames.append(z)
                    if len(current) == n and len(current & prev_c_partners) == 0:
                        valid = [v for v in ep_frames if not np.isnan(v)]
                        mean_z = float(np.mean(valid)) if valid else np.nan
                        if j_ep_idx < len(mt.zeta_J_mean):
                            mt.zeta_J_mean[j_ep_idx] = mean_z
                        else:
                            mt.zeta_J_mean.append(mean_z)
                        j_ep_idx += 1
                        state = "C"; c_partners = current; ep_frames = [z]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        r = self.results
        sep = "=" * 60
        print(f"\n{sep}")
        print("CAGE-JUMP MODEL RESULTS  –  Kikutsuji et al. (2019)")
        print(sep)
        print(f"  Temperature          : {self.T:.1f} K")
        print(f"  Frames analysed      : {len(self.frames)}")
        print(f"  n_hb_cage            : {self.n_hb_cage}")
        print()
        print(f"  ⟨τ_C⟩               : {r.mean_tau_C:.3f} ps")
        print(f"  ⟨τ_J⟩               : {r.mean_tau_J:.3f} ps")
        print(f"  ⟨r²_J⟩              : {r.mean_r2_J:.4f} Å²")
        print(f"  ρ_J                  : {r.rho_J:.4f}")
        print(f"  ρ_J (mean-field)     : {r.rho_J_mean_field:.4f}")
        print()
        print(f"  D_J (⟨r²_J⟩/6⟨τ_J⟩): {r.D_J:.4e} m²/s")
        print(f"  D_estimate (ρ_J D_J) : {r.D_estimate:.4e} m²/s")
        print(f"  D_MSD                : {r.D_MSD:.4e} m²/s")
        print()
        print(f"  τ_HB                 : {r.tau_HB:.3f} ps  "
              f"(β = {r.beta_HB:.3f})")
        print(sep + "\n")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(self, output_dir: str = "cagejump_plots"):
        """Generate all figures corresponding to the paper's Figs. 2–5.

        Parameters
        ----------
        output_dir : str
            Directory where PNG files are saved.
        """
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)

        r   = self.results
        dt  = self.dt

        # -----------------------------------------------------------------
        # Fig. 2a: P(τ_C)
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.logspace(np.log10(max(r.tau_C_all.min(), dt)),
                           np.log10(r.tau_C_all.max()), 50)
        counts, edges = np.histogram(r.tau_C_all, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        ax.semilogy(centers, counts, "o-", ms=4, lw=1.2)
        ax.set(xlabel=r"$\tau_C$ (ps)", ylabel=r"$P(\tau_C)$",
               title=r"Distribution of C-state duration  $P(\tau_C)$")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig2a_P_tauC.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # Fig. 2b: ⟨τ_C⟩, τ_HB, D⁻¹  vs 1000/T  (single-T: bar chart)
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(5, 4))
        quantities = {
            r"$\langle\tau_C\rangle$": r.mean_tau_C,
            r"$\tau_{HB}$":            r.tau_HB,
        }
        colors = ["steelblue", "tomato"]
        for (label, val), color in zip(quantities.items(), colors):
            ax.bar(label, val, color=color, alpha=0.8, edgecolor="black")
        ax.set(ylabel="Time (ps)",
               title=r"$\langle\tau_C\rangle$ vs $\tau_{HB}$  "
                     f"(T = {self.T:.0f} K)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig2b_tauC_tauHB.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # Fig. 3a: P(τ_J)
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(r.tau_J_all, bins=40, density=True,
                color="mediumseagreen", edgecolor="white")
        ax.axvline(r.mean_tau_J, color="black", lw=1.5, ls="--",
                   label=f"⟨τ_J⟩ = {r.mean_tau_J:.2f} ps")
        ax.set(xlabel=r"$\tau_J$ (ps)", ylabel=r"$P(\tau_J)$",
               title=r"Distribution of J-state duration  $P(\tau_J)$")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig3a_P_tauJ.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # Fig. 3b: P(r_J)
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        rJ_bins = np.linspace(0, r.r_J_all.max() * 1.05, 50)
        ax.hist(r.r_J_all, bins=rJ_bins, density=True,
                color="darkorchid", edgecolor="white")
        ax.axvline(r.mean_rJ, color="black", lw=1.5, ls="--",
                   label=f"⟨r_J⟩ = {r.mean_rJ:.3f} Å")
        ax.set(xlabel=r"$r_J$ (Å)", ylabel=r"$P(r_J)$",
               title=r"Distribution of jump length  $P(r_J)$")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig3b_P_rJ.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # Fig. 4a: JMSD(Θ_J)
        # -----------------------------------------------------------------
        if len(r.jmsd) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.loglog(r.jmsd_Theta, r.jmsd, "o", ms=3,
                      color="navy", label="JMSD")
            # Reference line: slope 1 (diffusive)
            x0, x1 = r.jmsd_Theta[0], r.jmsd_Theta[-1]
            y0 = r.jmsd[0] if not np.isnan(r.jmsd[0]) else r.mean_r2_J
            ax.loglog([x0, x1], [y0, y0 * (x1 / x0)],
                      "k--", lw=1.0, label="Slope 1 (Fickian)")
            ax.set(xlabel=r"$\Theta_J$ (jump count)",
                   ylabel=r"$\langle\delta r^2_J(\Theta_J)\rangle$ (Å²)",
                   title="Jumping MSD vs jump count")
            ax.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig4a_JMSD.png"), dpi=150)
            plt.close()

        # -----------------------------------------------------------------
        # Fig. 4b inset: MSD with long-time asymptote
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        valid = r.msd_tau_ps > 0
        ax.loglog(r.msd_tau_ps[valid], r.msd[valid],
                  lw=1.5, color="steelblue", label="MSD")
        # Mark ⟨r²_J⟩
        ax.axhline(r.mean_r2_J, color="tomato", lw=1.0, ls="--",
                   label=r"$\langle r^2_J\rangle$"
                         f" = {r.mean_r2_J:.3f} Å²")
        # Long-time asymptote 6Dt
        if not np.isnan(r.D_MSD):
            D_ang2_ps = r.D_MSD / (ANG2_TO_M2 / PS_TO_S)
            t_arr = r.msd_tau_ps
            ax.loglog(t_arr[t_arr > 0],
                      6.0 * D_ang2_ps * t_arr[t_arr > 0],
                      "k:", lw=1.0, label="6Dt asymptote")
        ax.set(xlabel=r"$t$ (ps)", ylabel=r"MSD (Å²)",
               title="Mean-square displacement")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig4b_MSD.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # Fig. 5: D vs ρ_J D_J   (single-T: single point + parity line)
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(5, 5))
        x_est = r.D_estimate
        y_msd = r.D_MSD
        if not (np.isnan(x_est) or np.isnan(y_msd)):
            ax.loglog([x_est], [y_msd], "o", ms=12, color="navy",
                      label=f"T = {self.T:.0f} K")
            lo = min(x_est, y_msd) * 0.5
            hi = max(x_est, y_msd) * 2.0
            ax.loglog([lo, hi], [lo, hi], "k-", lw=1.0, label="D = estimate")
            ax.set(xlabel=r"$\rho_J \langle r^2_J\rangle / \langle\tau_J\rangle$"
                          r"  (m²/s)",
                   ylabel=r"$D_{MSD}$ (m²/s)",
                   title="Cage-jump model vs MSD diffusion")
            ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig5_D_vs_estimate.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # H-bond ACF with stretched-exp fit
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        tau_hb = r.hbond_acf_tau_ps
        ax.semilogy(tau_hb, r.hbond_acf, lw=1.5, color="darkorange",
                    label="c(t)")
        if not np.isnan(r.tau_HB):
            fit = np.exp(-(tau_hb / r.tau_HB) ** r.beta_HB)
            ax.semilogy(tau_hb, fit, "k--", lw=1.2,
                        label=fr"Fit  $\tau_{{HB}}={r.tau_HB:.1f}$ ps,"
                              fr"  $\beta={r.beta_HB:.2f}$")
        ax.set(xlabel=r"$t$ (ps)", ylabel=r"$c(t)$",
               title="H-bond autocorrelation function",
               ylim=(1e-3, 1.2))
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hbond_acf.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # ρ_J distribution across molecules
        # -----------------------------------------------------------------
        rho_J_per_mol = np.array([mt.rho_J for mt in self.mol_trajectories
                                  if not np.isnan(mt.rho_J)])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(rho_J_per_mol, bins=40, color="mediumslateblue",
                edgecolor="white", density=True)
        ax.axvline(float(rho_J_per_mol.mean()), color="black", lw=1.5, ls="--",
                   label=f"⟨ρ_J⟩ = {rho_J_per_mol.mean():.3f}")
        ax.axvline(r.rho_J_mean_field, color="tomato", lw=1.5, ls="-.",
                   label=f"⟨τ_J⟩/(⟨τ_C⟩+⟨τ_J⟩) = {r.rho_J_mean_field:.3f}")
        ax.set(xlabel=r"$\rho_J$", ylabel="Probability density",
               title="Per-molecule J-state fraction")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rhoJ_distribution.png"), dpi=150)
        plt.close()

        # -----------------------------------------------------------------
        # ζ–τ_C correlation (only if ZOP provided)
        # -----------------------------------------------------------------
        if self.zop is not None:
            all_zC = [z for mt in self.mol_trajectories for z in mt.zeta_C_mean
                      if not np.isnan(z)]
            all_tC = [tc for mt in self.mol_trajectories
                      for tc, z in zip(mt.tau_C, mt.zeta_C_mean)
                      if not np.isnan(z)]
            if all_zC:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(all_tC, all_zC, s=3, alpha=0.3, color="steelblue")
                ax.axhline(0, color="black", lw=0.8, ls="--")
                ax.set(xlabel=r"$\tau_C$ (ps)", ylabel=r"$\langle\zeta\rangle_C$ (Å)",
                       title=r"Local order ζ vs cage duration $\tau_C$")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "zeta_vs_tauC.png"),
                            dpi=150)
                plt.close()

        print(f"\nAll figures saved to '{output_dir}/':")
        for fname in os.listdir(output_dir):
            print(f"  {fname}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_results(self, prefix: str = "cagejump"):
        """Save numerical results to .npy / .csv files."""
        r = self.results
        np.save(f"{prefix}_tau_C.npy",  r.tau_C_all)
        np.save(f"{prefix}_tau_J.npy",  r.tau_J_all)
        np.save(f"{prefix}_r_J.npy",    r.r_J_all)
        np.save(f"{prefix}_msd.npy",
                np.column_stack([r.msd_tau_ps, r.msd]))
        np.save(f"{prefix}_jmsd.npy",
                np.column_stack([r.jmsd_Theta, r.jmsd]))
        np.save(f"{prefix}_hbond_acf.npy",
                np.column_stack([r.hbond_acf_tau_ps, r.hbond_acf]))

        import csv
        summary = [
            ("quantity",               "value",           "unit"),
            ("temperature",            self.T,            "K"),
            ("n_hb_cage",              self.n_hb_cage,    ""),
            ("mean_tau_C",             r.mean_tau_C,      "ps"),
            ("mean_tau_J",             r.mean_tau_J,      "ps"),
            ("mean_r2_J",              r.mean_r2_J,       "Å²"),
            ("mean_rJ",                r.mean_rJ,         "Å"),
            ("rho_J",                  r.rho_J,           ""),
            ("rho_J_mean_field",       r.rho_J_mean_field,""),
            ("D_J",                    r.D_J,             "m²/s"),
            ("D_estimate",             r.D_estimate,      "m²/s"),
            ("D_MSD",                  r.D_MSD,           "m²/s"),
            ("tau_HB",                 r.tau_HB,          "ps"),
            ("beta_HB",                r.beta_HB,         ""),
        ]
        with open(f"{prefix}_summary.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(summary)

        print(f"Results saved with prefix '{prefix}'.")
