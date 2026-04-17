# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Mesoscopic Stress Analysis for LAMMPS Trajectories
====================================================

This module solves one specific problem: LAMMPS dump files that carry
per-atom stress-tensor components (``c_stress/atom`` or any per-atom
compute) alongside the usual coordinates.  MDAnalysis' ``DumpReader``
already has an ``additional_columns`` mechanism that stores arbitrary
dump columns in ``ts.data``; this module builds on top of it to provide:

1.  ``StressDumpReader``
    A thin ``DumpReader`` subclass that
    - auto-discovers stress component columns by prefix pattern
    - renames bracket-notation keys (``c_peratom[4]``) to clean names
      (``stress_xy`` etc.) inside ``ts.data``
    - stacks all components into a single ``ts.data["stress"]`` array
      of shape ``(n_atoms, n_components)``
    - applies the unit conversion from LAMMPS pressure×volume to Pa

2.  ``PerAtomStressAttr``
    A custom ``TopologyAttr`` that stores the *time-averaged* per-atom
    stress tensor so the mean field is accessible as ``u.atoms.mean_stress``.

3.  ``MesoscopicAnalysis``
    ``AnalysisBase`` subclass that computes, from a trajectory carrying
    per-atom stress:
    - System-level stress tensor components P_αβ(t) = Σᵢ sᵢ_αβ / V
    - Stress autocorrelation function (ACF)  C_αβ(τ)
    - Green-Kubo shear viscosity  η = V/(kT) ∫ C_xy(τ) dτ
    - Running integral viscosity η(t) for convergence diagnosis
    - Mean-squared displacement D from OW positions (Einstein relation)
    - Velocity autocorrelation function (VACF) → D_VACF (Green-Kubo)

LAMMPS stress convention reminder
-----------------------------------
``compute stress/atom`` outputs six values per atom in order:
    [Sxx, Syy, Szz, Sxy, Sxz, Syz]
Each value has units of **pressure × volume** (e.g. bar·Å³ in metal
units, Pa·Å³ in SI-ish units).  The sign convention in LAMMPS is:
    S_αβ = − (kinetic + virial) contribution  (compressive = negative)
so the actual pressure tensor element is  P_αβ = −Σᵢ Sᵢ_αβ / V.

Usage
-----
::

    import MDAnalysis as mda
    from stress_lammps import StressDumpReader, MesoscopicAnalysis

    # ---- Option A: use the adapted reader directly ----------------------
    u = mda.Universe(
        "tip4p-ice-225K.data",
        "dump_stress.lammpstrj",
        format=StressDumpReader,
        dt=0.2,
        stress_prefix="c_peratom",   # column pattern
        lammps_units="metal",
    )
    # Access per-frame per-atom stress:
    u.trajectory[0]
    pxy_frame0 = u.trajectory.ts.data["stress_xy"]  # shape (n_atoms,)
    stress_all  = u.trajectory.ts.data["stress"]    # shape (n_atoms, 6)

    # ---- Option B: convenience factory ---------------------------------
    from stress_lammps import stress_universe
    u = stress_universe("topology.data", "dump_stress.lammpstrj",
                        stress_prefix="c_peratom", dt=0.2)

    # ---- Run mesoscopic analysis ----------------------------------------
    ma = MesoscopicAnalysis(
        universe=u,
        temperature=225.0,          # K
        oxygen_sel="type 1",
        stress_component="xy",
        tau_max=2000,               # frames for ACF
    )
    ma.run()

    eta = ma.results.viscosity_GK   # Pa·s
    D   = ma.results.diffusion_msd  # m²/s
"""

import re
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid

import MDAnalysis as mda
from MDAnalysis.coordinates.LAMMPS import DumpReader
from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.core.topologyattrs import TopologyAttr
from MDAnalysis.exceptions import NoDataError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

kB_SI   = 1.380649e-23   # J/K
ANG3_TO_M3 = 1e-30        # Å³ → m³

# Unit conversion factors: LAMMPS-unit pressure×volume → Pa·m³ (i.e. Joules)
# Then dividing by volume (m³) gives Pa.
_UNIT_CONV: Dict[str, float] = {
    # (pressure_in_Pa) × (volume_in_m³ / Å³_in_m³)
    "real":   1.01325e5 * 1e-30,   # atm·Å³  → J   (1 atm = 101325 Pa)
    "metal":  1.0e5     * 1e-30,   # bar·Å³  → J   (1 bar = 1e5 Pa)
    "si":     1.0       * 1e-30,   # Pa·Å³   → J
    "lj":     1.0,                 # dimensionless – user must supply own T/V
    "nano":   1.0e9     * 1e-30,   # GPa·Å³  → J   (approx; verify for your LAMMPS build)
}

# LAMMPS stress/atom component order and canonical names
_STRESS_COMPONENTS = ["xx", "yy", "zz", "xy", "xz", "yz"]   # indices 0-5


# ---------------------------------------------------------------------------
# StressDumpReader
# ---------------------------------------------------------------------------

class StressDumpReader(DumpReader):
    """DumpReader that auto-parses per-atom stress-tensor columns.

    Inherits from ``DumpReader``; the only difference is that it
    pre-fills ``additional_columns`` with the stress component names
    found in the dump header, renames them to canonical ``stress_xx``
    etc. labels inside ``ts.data``, and stacks them into a single
    ``ts.data["stress"]`` array.

    Parameters
    ----------
    filename : str
    stress_prefix : str
        Column name prefix used in the LAMMPS dump header.
        For ``ITEM: ATOMS id type xu yu zu c_peratom[1] … c_peratom[6]``
        the prefix is ``"c_peratom"``.
        The reader expects **up to** 6 numbered components
        ``{prefix}[1]`` … ``{prefix}[6]`` in the LAMMPS order
        [Sxx, Syy, Szz, Sxy, Sxz, Syz].
    stress_components : list of str or None
        Which components to read, e.g. ``["xy"]`` to read only Sxy.
        ``None`` (default) reads all found components.
    lammps_units : str
        LAMMPS unit system.  One of ``"real"``, ``"metal"``, ``"si"``,
        ``"lj"``, ``"nano"``.  Controls the bar→Pa conversion factor.
        Default ``"metal"``.
    **kwargs
        Forwarded to ``DumpReader.__init__``.

    Attributes available on ``ts.data`` after each ``_read_next_timestep``
    -----------------------------------------------------------------------
    ``"stress"``       np.ndarray shape (n_atoms, n_components), float32
                       Per-atom stress tensor in bar·Å³ (raw LAMMPS value,
                       before unit conversion).
    ``"stress_xx"``    np.ndarray shape (n_atoms,)  … one per component
    ``"stress_yy"``    …
    ``"stress_zz"``    …
    ``"stress_xy"``    …
    ``"stress_xz"``    …
    ``"stress_yz"``    …

    Note: to get stress in Pa you need ``ts.data["stress_xy"] * unit_factor / V``
    where ``V`` is the box volume.  Use ``StressDumpReader.to_pressure()`` or
    ``MesoscopicAnalysis`` which handles this automatically.
    """

    def __init__(
        self,
        filename: str,
        stress_prefix: str = "c_peratom",
        stress_components: Optional[List[str]] = None,
        lammps_units: str = "metal",
        **kwargs,
    ):
        self._stress_prefix     = stress_prefix
        self._stress_components = stress_components or _STRESS_COMPONENTS
        self._lammps_units      = lammps_units
        self._unit_factor       = _UNIT_CONV.get(lammps_units, _UNIT_CONV["metal"])

        # Build the column names expected in the dump header
        # LAMMPS uses 1-based indexing for compute array components
        _COMP_TO_IDX = {c: i + 1 for i, c in enumerate(_STRESS_COMPONENTS)}
        self._stress_col_map: Dict[str, str] = {}  # "c_peratom[4]" → "stress_xy"
        additional = []
        for comp in self._stress_components:
            idx = _COMP_TO_IDX.get(comp)
            if idx is None:
                warnings.warn(f"Unknown stress component '{comp}', skipping.")
                continue
            raw_name = f"{stress_prefix}[{idx}]"
            canonical = f"stress_{comp}"
            self._stress_col_map[raw_name] = canonical
            additional.append(raw_name)

        if not additional:
            raise ValueError(
                f"No valid stress components found for prefix='{stress_prefix}' "
                f"and components={self._stress_components}."
            )

        # Pass the raw names to the parent's additional_columns mechanism.
        # If the user already passed additional_columns, merge them.
        user_extra = kwargs.pop("additional_columns", None)
        if user_extra and user_extra is not True:
            additional = list(additional) + [c for c in user_extra
                                              if c not in additional]
        elif user_extra is True:
            additional = True   # grab everything

        super().__init__(filename, additional_columns=additional, **kwargs)

    def _read_next_timestep(self):
        """Read one frame, then rename and stack stress columns."""
        ts = super()._read_next_timestep()
        self._postprocess_stress(ts)
        return ts

    def _postprocess_stress(self, ts):
        """Rename raw LAMMPS keys → canonical names; stack into 'stress' array."""
        found_components: List[Tuple[str, np.ndarray]] = []
        for raw_name, canonical in self._stress_col_map.items():
            if raw_name in ts.data:
                arr = ts.data.pop(raw_name).astype(np.float32)
                ts.data[canonical] = arr
                found_components.append((canonical, arr))

        if found_components:
            ts.data["stress"] = np.column_stack(
                [arr for _, arr in found_components]
            ).astype(np.float32)

    def to_pressure(self, component: str = "xy") -> np.ndarray:
        """Return current-frame per-atom ``component`` stress in Pa.

        The LAMMPS value (bar·Å³ or equivalent) is divided by the box
        volume to give Pa.

        Parameters
        ----------
        component : str  one of xx/yy/zz/xy/xz/yz

        Returns
        -------
        np.ndarray, shape (n_atoms,), float64
        """
        ts  = self.ts
        key = f"stress_{component}"
        if key not in ts.data:
            raise NoDataError(
                f"'{key}' not in ts.data.  "
                f"Available stress keys: "
                f"{[k for k in ts.data if k.startswith('stress_')]}"
            )
        V_ang3 = float(np.prod(ts.dimensions[:3]))   # Å³ (orthorhombic approx.)
        V_m3   = V_ang3 * ANG3_TO_M3
        # raw value is Σᵢ sᵢ; we want Σᵢ sᵢ / V in Pa.
        # raw units: bar·Å³  →  ×1e5 → Pa·Å³  →  /V[m³] → Pa
        # combined: × unit_factor / V_m3
        return ts.data[key].astype(np.float64) * self._unit_factor / V_m3


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def stress_universe(
    topology_file: str,
    trajectory_file: str,
    stress_prefix: str = "c_peratom",
    stress_components: Optional[List[str]] = None,
    lammps_units: str = "metal",
    dt: float = 1.0,
    **kwargs,
) -> mda.Universe:
    """Create an MDAnalysis Universe that reads per-atom stress columns.

    This is the recommended entry point.  It constructs a Universe using
    ``StressDumpReader`` as the trajectory reader, so every
    ``u.trajectory.ts.data`` dict contains clean ``stress_xy`` etc. arrays.

    Parameters
    ----------
    topology_file : str
        LAMMPS data file path.
    trajectory_file : str
        LAMMPS dump file path.
    stress_prefix : str
        Column prefix in the dump header (see ``StressDumpReader``).
    stress_components : list of str or None
        Which tensor components to read.  ``None`` → all six.
    lammps_units : str
        LAMMPS unit system for conversion.
    dt : float
        Timestep in ps.
    **kwargs
        Additional kwargs forwarded to ``mda.Universe``.

    Returns
    -------
    mda.Universe
        Ready-to-use universe.  Access per-atom stress via
        ``u.trajectory.ts.data["stress_xy"]`` etc.
    """
    return mda.Universe(
        topology_file,
        trajectory_file,
        format=StressDumpReader,
        dt=dt,
        stress_prefix=stress_prefix,
        stress_components=stress_components,
        lammps_units=lammps_units,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Custom TopologyAttr: time-averaged per-atom stress
# ---------------------------------------------------------------------------

class PerAtomStressAttr(TopologyAttr):
    """Time-averaged per-atom stress tensor as a topology attribute.

    Adds ``u.atoms.mean_stress`` (shape ``(n_atoms, 6)``) and
    per-component properties ``u.atoms.mean_stress_xy`` etc. to the
    Universe after a trajectory scan.

    Do not construct directly; use ``PerAtomStressAttr.from_trajectory()``.
    """

    attrname  = "mean_stress"
    singular  = "mean_stress"
    dtype     = np.float32

    @classmethod
    def from_trajectory(
        cls,
        universe: mda.Universe,
        start: Optional[int] = None,
        stop:  Optional[int] = None,
        step:  Optional[int] = None,
    ) -> "PerAtomStressAttr":
        """Compute the time-average of ``ts.data["stress"]`` over the trajectory.

        Parameters
        ----------
        universe : mda.Universe
            Must use ``StressDumpReader`` (or any reader that populates
            ``ts.data["stress"]``).
        start, stop, step : int or None
            Frame range.

        Returns
        -------
        PerAtomStressAttr
            Add to universe with ``universe.add_TopologyAttr(attr)``.
        """
        n_atoms = universe.atoms.n_atoms
        accumulator = None
        count = 0

        for ts in universe.trajectory[start:stop:step]:
            if "stress" not in ts.data:
                raise NoDataError(
                    "ts.data['stress'] not found.  Use StressDumpReader."
                )
            arr = ts.data["stress"]   # (n_atoms, n_comp)
            if accumulator is None:
                accumulator = np.zeros((n_atoms, arr.shape[1]), dtype=np.float64)
            accumulator += arr
            count += 1

        if count == 0:
            raise NoDataError("No frames were iterated.")

        mean_stress = (accumulator / count).astype(np.float32)
        return cls(mean_stress)

    def __getitem__(self, atoms):
        return self.values[atoms.indices]


def add_mean_stress_attr(
    universe: mda.Universe,
    start: Optional[int] = None,
    stop:  Optional[int] = None,
    step:  Optional[int] = None,
) -> None:
    """Convenience wrapper: compute and attach ``PerAtomStressAttr`` to ``u``.

    After calling this, ``u.atoms.mean_stress`` returns shape (n_atoms, 6)
    and ``u.select_atoms("type 1").mean_stress`` returns the sub-array.
    """
    attr = PerAtomStressAttr.from_trajectory(universe, start, stop, step)
    universe.add_TopologyAttr(attr)
    logger.info("mean_stress topology attribute added to universe.")


# ---------------------------------------------------------------------------
# MesoscopicAnalysis
# ---------------------------------------------------------------------------

class MesoscopicAnalysis(AnalysisBase):
    """Compute viscosity and diffusion contributions from per-atom stress.

    Theory
    ------
    **Viscosity (Green-Kubo)**::

        η = V/(kB T) * ∫₀^∞ ⟨δP_xy(0) δP_xy(τ)⟩ dτ

    where  P_xy(t) = (1/V) Σᵢ sᵢ_xy(t)  is the total off-diagonal pressure
    element.  The per-atom LAMMPS values sᵢ_xy are stored (with sign from
    LAMMPS convention) in ``ts.data["stress_xy"]``.

    **Diffusion (Einstein MSD)**::

        D = lim_{τ→∞} MSD(τ) / (6τ)

    **Diffusion (Green-Kubo VACF)**::

        D = (1/3) ∫₀^∞ ⟨v(0)·v(τ)⟩ dτ

    Parameters
    ----------
    universe : mda.Universe
        Must use ``StressDumpReader`` so that ``ts.data["stress_xy"]``
        is populated.
    temperature : float
        System temperature in K.
    oxygen_sel : str
        MDAnalysis selection string for the atoms whose MSD / VACF gives D.
    stress_component : str
        Which off-diagonal component to integrate.  ``"xy"`` (default),
        ``"xz"``, or ``"yz"``.
    tau_max : int
        Maximum lag time (frames) for ACF calculation.  Should be large
        enough that the ACF decays to zero.  Default 1000.
    lammps_units : str
        Must match the reader's unit system so that pressure conversion
        factors are consistent.
    """

    def __init__(
        self,
        universe: mda.Universe,
        temperature: float,
        oxygen_sel:       str   = "type 1",
        stress_component: str   = "xy",
        tau_max:          int   = 1000,
        lammps_units:     str   = "metal",
    ):
        super().__init__(universe.trajectory)
        self.u                 = universe
        self.T                 = temperature
        self.oxygen_sel        = oxygen_sel
        self.stress_component  = stress_component
        self.tau_max           = tau_max
        self._unit_factor      = _UNIT_CONV.get(lammps_units, _UNIT_CONV["metal"])
        self._stress_key       = f"stress_{stress_component}"

        self._oxygen_ag = universe.select_atoms(oxygen_sel)
        if len(self._oxygen_ag) == 0:
            raise ValueError(f"Selection '{oxygen_sel}' matched no atoms.")

    def _prepare(self):
        n = len(self.frames)
        # Collective stress (sum over atoms) per frame – raw LAMMPS units
        self._P_raw   = np.zeros(n, dtype=np.float64)
        # Box volume per frame (Å³)
        self._vol_ang3 = np.zeros(n, dtype=np.float64)
        # Oxygen positions per frame for MSD
        n_oxy = len(self._oxygen_ag)
        self._positions = np.zeros((n, n_oxy, 3), dtype=np.float32)
        # Velocities if available
        self._has_vels = False   # updated in _single_frame
        self._velocities = None

        self._frame_counter = 0

    def _single_frame(self):
        ts  = self._ts
        fi  = self._frame_counter

        # ---- Collective stress -------------------------------------------
        if self._stress_key not in ts.data:
            raise NoDataError(
                f"'{self._stress_key}' not in ts.data.  "
                "Ensure the trajectory was loaded with StressDumpReader "
                f"and stress_components includes '{self.stress_component}'."
            )
        self._P_raw[fi]    = ts.data[self._stress_key].sum()
        self._vol_ang3[fi] = float(np.prod(ts.dimensions[:3]))

        # ---- Oxygen positions -------------------------------------------
        self._positions[fi] = self._oxygen_ag.positions.copy()

        # ---- Velocities (if present) ------------------------------------
        if ts.has_velocities:
            self._has_vels = True
            if self._velocities is None:
                self._velocities = np.zeros(
                    (len(self.frames), len(self._oxygen_ag), 3), dtype=np.float32
                )
            self._velocities[fi] = self._oxygen_ag.velocities.copy()

        self._frame_counter += 1

    def _conclude(self):
        dt_s = self.u.trajectory.dt * 1e-12   # ps → s

        # Convert collective stress to Pa (system pressure tensor element)
        # P_xy [Pa] = Σᵢ sᵢ_xy [bar·Å³] × unit_factor / V [m³]
        V_m3_arr = self._vol_ang3 * ANG3_TO_M3
        P_Pa     = (self._P_raw * self._unit_factor) / V_m3_arr  # (n_frames,)

        # Remove mean (fluctuation)
        dP = P_Pa - P_Pa.mean()

        # Store raw timeseries
        self.results.P_xy_Pa    = P_Pa
        self.results.times_ps   = self.frames * self.u.trajectory.dt
        self.results.times_s    = self.results.times_ps * 1e-12

        # ---- Stress ACF (Green-Kubo) ------------------------------------
        tau_max = min(self.tau_max, len(self.frames) - 1)
        acf = _compute_acf(dP, tau_max)  # normalised C(τ)
        acf_unnorm = _compute_acf_unnorm(dP, tau_max)  # ⟨δP(0)δP(τ)⟩ Pa²

        tau_s = np.arange(tau_max + 1) * dt_s

        self.results.acf_tau_ps = tau_s * 1e12
        self.results.acf_norm   = acf
        self.results.acf_Pa2    = acf_unnorm

        # ---- Green-Kubo viscosity ---------------------------------------
        # η = V/(kBT) ∫ C(τ) dτ  where V and T are averages
        V_avg_m3 = float(V_m3_arr.mean())
        integrand = acf_unnorm                   # Pa²
        running_integral = cumulative_trapezoid(integrand, tau_s, initial=0)
        eta_running      = (V_avg_m3 / (kB_SI * self.T)) * running_integral

        self.results.eta_running_mPas  = eta_running * 1e3  # Pa·s → mPa·s
        self.results.viscosity_GK      = float(eta_running[-1])  # Pa·s

        # ---- MSD → diffusion (Einstein) ---------------------------------
        msd_arr, tau_arr_s = _compute_msd(self._positions, dt_s)
        # Linear fit over 20-80 % of the MSD range for D extraction
        fit_slice = slice(
            len(tau_arr_s) // 5,
            4 * len(tau_arr_s) // 5,
        )
        if tau_arr_s[fit_slice].size >= 3:
            slope, _  = np.polyfit(tau_arr_s[fit_slice], msd_arr[fit_slice], 1)
            D_msd     = slope / 6.0   # 3D: MSD = 6Dt
        else:
            D_msd = np.nan

        self.results.msd_ang2       = msd_arr          # Å²
        self.results.msd_tau_ps     = tau_arr_s * 1e12 # ps
        self.results.diffusion_msd  = D_msd * ANG3_TO_M3 / 1e-30  # Å²/s → m²/s
        # Note: ANG3_TO_M3 is Å³→m³; for Å²→m² use 1e-20
        self.results.diffusion_msd  = D_msd * 1e-20    # Å²/s → m²/s

        # ---- VACF → diffusion (Green-Kubo) ------------------------------
        if self._has_vels and self._velocities is not None:
            vacf, tau_vacf_s = _compute_vacf(self._velocities, dt_s,
                                              tau_max=min(tau_max, 500))
            # D = (1/3) ∫ VACF dt  [Å²/ps units in LAMMPS metal → ×1e-8 → m²/s]
            # LAMMPS metal velocities in Å/ps
            # VACF in Å²/ps²; ×dt_s = Å²/ps; Σ → Å²/ps × ps = Å²
            # D [m²/s] = (1/3) × ∫VACF dt × (Å²/ps → m²/s)
            # 1 Å²/ps = 1e-20 m² / 1e-12 s = 1e-8 m²/s
            D_vacf_ang2ps = (1.0 / 3.0) * np.trapz(vacf, tau_vacf_s * 1e12)
            self.results.vacf              = vacf
            self.results.vacf_tau_ps       = tau_vacf_s * 1e12
            self.results.diffusion_vacf    = D_vacf_ang2ps * 1e-8  # m²/s
        else:
            self.results.vacf           = None
            self.results.diffusion_vacf = np.nan
            if not self._has_vels:
                warnings.warn(
                    "Velocities not found in trajectory.  VACF-based "
                    "diffusion will be NaN.  Include vx/vy/vz columns in "
                    "the LAMMPS dump to enable this."
                )

        # ---- Print summary ----------------------------------------------
        _print_summary(self)


# ---------------------------------------------------------------------------
# Module-level computation helpers
# ---------------------------------------------------------------------------

def _compute_acf(x: np.ndarray, tau_max: int) -> np.ndarray:
    """Normalised autocorrelation C(τ) = ⟨x(0)x(τ)⟩ / ⟨x²⟩ via FFT."""
    n    = len(x)
    fft  = np.fft.rfft(x, n=2 * n)
    acf  = np.fft.irfft(fft * np.conj(fft))[:n]
    acf /= (np.arange(n, 0, -1))      # correct for varying sample count
    norm = acf[0]
    if norm == 0:
        return np.zeros(tau_max + 1)
    return (acf / norm)[: tau_max + 1]


def _compute_acf_unnorm(x: np.ndarray, tau_max: int) -> np.ndarray:
    """Un-normalised ACF  ⟨x(0)x(τ)⟩  (same units as x²)."""
    n    = len(x)
    fft  = np.fft.rfft(x, n=2 * n)
    acf  = np.fft.irfft(fft * np.conj(fft))[:n]
    acf /= np.arange(n, 0, -1)
    return acf[: tau_max + 1]


def _compute_msd(
    positions: np.ndarray,
    dt_s: float,
    max_tau_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Window-averaged MSD over multiple time origins.

    Parameters
    ----------
    positions : np.ndarray, shape (n_frames, n_atoms, 3)  in Å
    dt_s      : float  – timestep in seconds
    max_tau_frac : float  – use at most this fraction of total frames as τ

    Returns
    -------
    msd  : np.ndarray, shape (tau_max+1,)  in Å²
    taus : np.ndarray, shape (tau_max+1,)  in seconds
    """
    n_frames, n_atoms, _ = positions.shape
    tau_max = int(n_frames * max_tau_frac)
    msd  = np.zeros(tau_max + 1)

    for tau in range(tau_max + 1):
        if tau == 0:
            msd[0] = 0.0
            continue
        disp = positions[tau:] - positions[: n_frames - tau]   # (T-τ, N, 3)
        msd[tau] = float((disp ** 2).sum(axis=2).mean())

    taus = np.arange(tau_max + 1) * dt_s
    return msd, taus


def _compute_vacf(
    velocities: np.ndarray,
    dt_s: float,
    tau_max: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Window-averaged VACF  ⟨v(0)·v(τ)⟩ per atom, averaged over atoms.

    Parameters
    ----------
    velocities : np.ndarray, shape (n_frames, n_atoms, 3)  in Å/ps
    dt_s       : float  – timestep in seconds
    tau_max    : int

    Returns
    -------
    vacf : np.ndarray, shape (tau_max+1,)
    taus : np.ndarray, shape (tau_max+1,)  in seconds
    """
    n_frames, n_atoms, _ = velocities.shape
    tau_max = min(tau_max, n_frames - 1)
    vacf = np.zeros(tau_max + 1)
    counts = np.zeros(tau_max + 1, dtype=int)

    # Use FFT per atom for speed
    for a in range(n_atoms):
        v = velocities[:, a, :]   # (n_frames, 3)
        for dim in range(3):
            x = v[:, dim].astype(np.float64)
            n = len(x)
            fft = np.fft.rfft(x, n=2 * n)
            c   = np.fft.irfft(fft * np.conj(fft))[:n]
            c  /= np.arange(n, 0, -1)
            vacf[:tau_max + 1] += c[:tau_max + 1]

    vacf /= (3.0 * n_atoms)   # normalise by 3 dims and n_atoms
    taus  = np.arange(tau_max + 1) * dt_s
    return vacf, taus


def _print_summary(ma: MesoscopicAnalysis):
    """Print a formatted result summary."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("MESOSCOPIC ANALYSIS RESULTS")
    print(sep)
    print(f"  Temperature            : {ma.T:.1f} K")
    print(f"  Stress component       : P_{ma.stress_component}")
    print(f"  Frames analysed        : {len(ma.frames)}")
    print(f"  τ_max for ACF          : {ma.tau_max} frames "
          f"({ma.tau_max * ma.u.trajectory.dt:.1f} ps)")
    print()
    print(f"  Green-Kubo viscosity   : {ma.results.viscosity_GK * 1e3:.4f} mPa·s")
    print(f"  Diffusion (MSD)        : {ma.results.diffusion_msd:.4e} m²/s")
    if ma.results.diffusion_vacf is not np.nan and not np.isnan(ma.results.diffusion_vacf):
        print(f"  Diffusion (VACF)       : {ma.results.diffusion_vacf:.4e} m²/s")
    print(f"  ⟨P_xy⟩ (mean stress)   : {ma.results.P_xy_Pa.mean():.4f} Pa")
    print(f"  σ(P_xy) (fluctuation)  : {ma.results.P_xy_Pa.std():.4f} Pa")
    print(sep + "\n")
