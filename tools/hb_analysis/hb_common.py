from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, minimize_vectors


@dataclass
class MappingReport:
    strategy: str
    max_oh_distance_sampled: float
    bad_oh_pairs_sampled: int
    sampled_frames: List[int]
    notes: List[str]


class WaterHBondToolkit:
    """Utilities to compute water hydrogen-bond networks using the HOO criterion.

    HB criterion for a water pair (i, j):
        O_i O_j distance <= oo_cutoff
        and min(H-O...O) over the 4 hydrogens attached to i or j <= hoo_cutoff
    """

    def __init__(
        self,
        dump_file: str,
        oxygen_selection: str = "type 1",
        hydrogen_selection: str = "type 2",
        oo_cutoff: float = 3.5,
        hoo_cutoff: float = 30.0,
        oh_cutoff: float = 1.25,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        validation_frames: int = 5,
    ) -> None:
        self.universe = mda.Universe(dump_file, format="LAMMPSDUMP")
        self.oxygen_selection = oxygen_selection
        self.hydrogen_selection = hydrogen_selection
        self.O_atoms = self.universe.select_atoms(oxygen_selection)
        self.H_atoms = self.universe.select_atoms(hydrogen_selection)
        if len(self.H_atoms) != 2 * len(self.O_atoms):
            raise ValueError("Expected 2 hydrogens per oxygen in selected water set")
        self.O_indices = np.asarray(self.O_atoms.indices, dtype=np.int64)
        self.H_indices = np.asarray(self.H_atoms.indices, dtype=np.int64)
        self.O_ids = self._safe_ids(self.O_atoms)
        self.n_oxygen = len(self.O_indices)
        self.oo_cutoff = float(oo_cutoff)
        self.hoo_cutoff = float(hoo_cutoff)
        self.oh_cutoff = float(oh_cutoff)
        self.start_frame = 0 if start_frame is None else int(start_frame)
        self.end_frame = len(self.universe.trajectory) if end_frame is None else int(end_frame)
        self.validation_frames = max(1, int(validation_frames))
        self._global_to_local = {int(g): i for i, g in enumerate(self.O_indices)}
        self.o_to_h_global, self.mapping_report = self._build_water_mapping()

    @staticmethod
    def _safe_ids(atomgroup) -> np.ndarray:
        try:
            ids = np.asarray(atomgroup.ids, dtype=np.int64)
            if len(ids) == len(atomgroup):
                return ids
        except Exception:
            pass
        return np.asarray(atomgroup.indices, dtype=np.int64) + 1

    def _reference_frames(self) -> List[int]:
        frames = np.linspace(self.start_frame, self.end_frame - 1, min(self.validation_frames, self.end_frame - self.start_frame), dtype=int)
        out: List[int] = []
        seen = set()
        for fr in frames.tolist():
            if fr not in seen:
                out.append(int(fr))
                seen.add(int(fr))
        return out

    def _mapping_from_residues(self) -> Tuple[Optional[np.ndarray], List[str]]:
        notes: List[str] = []
        try:
            residues = self.universe.residues
        except Exception as exc:
            return None, [f"Residue mapping unavailable: {exc!r}"]
        if len(residues) != self.n_oxygen:
            return None, ["Residue count does not match oxygen count"]
        oset = set(int(x) for x in self.O_indices.tolist())
        hset = set(int(x) for x in self.H_indices.tolist())
        mapping = np.full((self.n_oxygen, 2), -1, dtype=np.int64)
        for res in residues:
            res_o = [int(a.index) for a in res.atoms if int(a.index) in oset]
            res_h = [int(a.index) for a in res.atoms if int(a.index) in hset]
            if len(res_o) != 1 or len(res_h) != 2:
                return None, ["Residue mapping failed because a residue is not OHH"]
            mapping[self._global_to_local[res_o[0]]] = np.sort(np.asarray(res_h, dtype=np.int64))
        notes.append("Mapping built from residues")
        return mapping, notes

    def _mapping_from_sequential(self) -> Tuple[Optional[np.ndarray], List[str]]:
        hset = set(int(x) for x in self.H_indices.tolist())
        mapping = np.full((self.n_oxygen, 2), -1, dtype=np.int64)
        n_atoms = len(self.universe.atoms)
        for i, go in enumerate(self.O_indices):
            h1 = int(go) + 1
            h2 = int(go) + 2
            if h2 >= n_atoms:
                return None, ["Sequential mapping ran past atom list"]
            if h1 not in hset or h2 not in hset:
                return None, ["Sequential mapping expected O-H-H atom order"]
            mapping[i] = np.asarray([h1, h2], dtype=np.int64)
        return mapping, ["Mapping built from sequential O-H-H order"]

    def _mapping_from_geometry(self) -> Tuple[Optional[np.ndarray], List[str]]:
        self.universe.trajectory[self.start_frame]
        O_pos = np.asarray(self.O_atoms.positions, dtype=np.float32)
        H_pos = np.asarray(self.H_atoms.positions, dtype=np.float32)
        box = np.asarray(self.universe.dimensions, dtype=np.float32)
        pairs, dists = capped_distance(O_pos, H_pos, max_cutoff=self.oh_cutoff, box=box, return_distances=True)
        if len(pairs) == 0:
            return None, ["No O-H candidates found within oh_cutoff"]
        nearest_o_for_h = np.full(len(self.H_indices), -1, dtype=np.int64)
        nearest_dist_for_h = np.full(len(self.H_indices), np.inf, dtype=float)
        for (ol, hl), dist in zip(pairs, dists):
            if dist < nearest_dist_for_h[hl]:
                nearest_dist_for_h[hl] = float(dist)
                nearest_o_for_h[hl] = int(ol)
        mapping_lists: List[List[int]] = [[] for _ in range(self.n_oxygen)]
        for hlocal, olocal in enumerate(nearest_o_for_h):
            if olocal >= 0:
                mapping_lists[int(olocal)].append(int(self.H_indices[hlocal]))
        if any(len(v) != 2 for v in mapping_lists):
            return None, ["Geometry mapping failed because some oxygens do not have exactly 2 hydrogens"]
        mapping = np.empty((self.n_oxygen, 2), dtype=np.int64)
        for i, hs in enumerate(mapping_lists):
            mapping[i] = np.sort(np.asarray(hs, dtype=np.int64))
        return mapping, ["Mapping built from nearest-oxygen geometry"]

    def _validate_mapping(self, mapping: np.ndarray, strategy: str, notes: Sequence[str]) -> Tuple[bool, MappingReport]:
        bad = 0
        max_oh = 0.0
        sampled = self._reference_frames()
        for fr in sampled:
            self.universe.trajectory[fr]
            pos = np.asarray(self.universe.atoms.positions, dtype=np.float64)
            box = np.asarray(self.universe.dimensions, dtype=np.float64)
            O_pos = pos[self.O_indices]
            H_pos = pos[mapping.reshape(-1)]
            O_rep = np.repeat(O_pos, 2, axis=0)
            vec = minimize_vectors(H_pos - O_rep, box)
            dist = np.linalg.norm(vec, axis=1)
            if dist.size:
                max_oh = max(max_oh, float(np.max(dist)))
            bad += int(np.count_nonzero(dist > self.oh_cutoff))
        report = MappingReport(strategy=strategy, max_oh_distance_sampled=max_oh, bad_oh_pairs_sampled=bad, sampled_frames=sampled, notes=list(notes))
        return bad == 0, report

    def _build_water_mapping(self) -> Tuple[np.ndarray, MappingReport]:
        last_report: Optional[MappingReport] = None
        for strategy, builder in (("residue", self._mapping_from_residues), ("sequential", self._mapping_from_sequential), ("geometry", self._mapping_from_geometry)):
            mapping, notes = builder()
            if mapping is None:
                last_report = MappingReport(strategy=strategy, max_oh_distance_sampled=float("nan"), bad_oh_pairs_sampled=-1, sampled_frames=[], notes=list(notes))
                continue
            ok, report = self._validate_mapping(mapping, strategy, notes)
            last_report = report
            if ok:
                return mapping, report
        raise ValueError("Failed to build O-H mapping: " + "; ".join(last_report.notes if last_report else ["unknown error"]))

    def min_hoo_angle_deg(self, o_local_i: int, o_local_j: int, positions: np.ndarray, box: np.ndarray) -> float:
        gi = int(self.O_indices[o_local_i])
        gj = int(self.O_indices[o_local_j])
        pos_Oi = positions[gi]
        pos_Oj = positions[gj]
        oo_vec = minimize_vectors((pos_Oj - pos_Oi).reshape(1, 3), box).reshape(3)
        oo_norm = float(np.linalg.norm(oo_vec))
        if oo_norm == 0.0:
            return 0.0
        min_angle = 180.0
        for h_global in self.o_to_h_global[o_local_i]:
            pos_H = positions[int(h_global)]
            oh_vec = minimize_vectors((pos_H - pos_Oi).reshape(1, 3), box).reshape(3)
            oh_norm = float(np.linalg.norm(oh_vec))
            if oh_norm > 0.0:
                c = np.dot(oh_vec, oo_vec) / (oh_norm * oo_norm)
                min_angle = min(min_angle, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        minus_oo_vec = -oo_vec
        for h_global in self.o_to_h_global[o_local_j]:
            pos_H = positions[int(h_global)]
            oh_vec = minimize_vectors((pos_H - pos_Oj).reshape(1, 3), box).reshape(3)
            oh_norm = float(np.linalg.norm(oh_vec))
            if oh_norm > 0.0:
                c = np.dot(oh_vec, minus_oo_vec) / (oh_norm * oo_norm)
                min_angle = min(min_angle, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        return min_angle

    def compute_frame_network(self, frame: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.universe.trajectory[frame]
        positions = np.asarray(self.universe.atoms.positions, dtype=np.float64)
        box = np.asarray(self.universe.dimensions, dtype=np.float64)
        O_pos = positions[self.O_indices]
        pairs, dists = capped_distance(O_pos, O_pos, max_cutoff=self.oo_cutoff, box=box, return_distances=True)
        edges_i: List[int] = []
        edges_j: List[int] = []
        edge_dist: List[float] = []
        angles: List[float] = []
        for (i, j), dist in zip(pairs, dists):
            i = int(i)
            j = int(j)
            if i >= j:
                continue
            ang = self.min_hoo_angle_deg(i, j, positions, box)
            if ang <= self.hoo_cutoff:
                edges_i.append(i)
                edges_j.append(j)
                edge_dist.append(float(dist))
                angles.append(float(ang))
        if not edges_i:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=float), np.empty((0,), dtype=float)
        return np.column_stack([np.asarray(edges_i, dtype=np.int64), np.asarray(edges_j, dtype=np.int64)]), np.asarray(edge_dist, dtype=float), np.asarray(angles, dtype=float)

    def count_hbonds(self, edges: np.ndarray) -> np.ndarray:
        counts = np.zeros(self.n_oxygen, dtype=np.int64)
        if edges.size:
            np.add.at(counts, edges[:, 0], 1)
            np.add.at(counts, edges[:, 1], 1)
        return counts

    def partner_sets(self, edges: np.ndarray) -> List[set[int]]:
        partners = [set() for _ in range(self.n_oxygen)]
        for i, j in edges:
            partners[int(i)].add(int(j))
            partners[int(j)].add(int(i))
        return partners


def autocorrelation_mean_centered(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("autocorrelation expects a 1D array")
    n = x.size
    if n == 0:
        return np.empty(0, dtype=float)
    if max_lag is None or max_lag >= n:
        max_lag = n - 1
    x0 = x - np.mean(x)
    var = np.dot(x0, x0) / n
    if var <= 0.0:
        out = np.zeros(max_lag + 1, dtype=float)
        out[0] = 1.0
        return out
    corr = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        corr[lag] = np.dot(x0[: n - lag], x0[lag:]) / (n - lag) / var
    return corr


def survival_from_lengths(lengths: Sequence[int], max_lag: Optional[int] = None) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return np.empty(0, dtype=float)
    if max_lag is None:
        max_lag = int(np.max(lengths))
    out = np.zeros(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        out[lag] = np.mean(lengths >= lag)
    return out
