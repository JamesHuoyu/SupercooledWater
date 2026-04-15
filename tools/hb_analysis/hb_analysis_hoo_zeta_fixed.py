
#!/usr/bin/env python3
"""
Compute the translational order parameter zeta for water using the HOO hydrogen-bond criterion.

Definition
----------
zeta = d_min(non-HB water) - d_max(HB water)

where for each central water molecule:
- d_max(HB water) is the farthest O···O distance among water molecules that satisfy
  the hydrogen-bond criterion:
      O···O <= oo_cutoff  and  min(H-O···O) <= hoo_cutoff
- d_min(non-HB water) is the nearest O···O distance to a water molecule that does
  NOT satisfy the above hydrogen-bond criterion.

Key fixes compared with the previous script
-------------------------------------------
1) Uses an explicit HOO-based geometric criterion instead of MDAnalysis HBA
   (HBA uses a D-H-A angle cutoff, not an H-O···O cutoff).
2) Avoids silently missing nearest non-HB partners:
   - first searches within neighbor_cutoff via FastNS for speed
   - then falls back to a full minimum-image O···O search ONLY for atoms still missing
     a nearest non-HB partner
3) Writes a self-check report so every missing/undefined zeta value has an explicit reason.

Outputs
-------
- hb_counts_per_idx.csv
- max_distance_per_idx.csv
- nhb_min_distances.csv
- zeta.csv           (all oxygens, includes status column)
- zeta_valid.csv     (only rows with status == "ok")
- self_check_frame_summary.csv
- self_check_summary.json

Usage
-----
python hb_analysis_hoo_zeta.py --dump_file traj.dump --out_dir output
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, distance_array, minimize_vectors
from MDAnalysis.lib.nsgrid import FastNS


@dataclass
class MappingReport:
    strategy: str
    reference_frame: int
    n_oxygen: int
    n_hydrogen: int
    max_oh_distance_sampled: float
    bad_oh_pairs_sampled: int
    sampled_frames: List[int]
    notes: List[str]


class ZetaHOOAnalyzer:
    def __init__(
        self,
        dump_file: str,
        out_dir: str = "output",
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        oxygen_selection: str = "type 1",
        hydrogen_selection: str = "type 2",
        oo_cutoff: float = 3.5,
        hoo_cutoff: float = 30.0,
        oh_cutoff: float = 1.25,
        neighbor_cutoff: float = 6.0,
        validation_frames: int = 5,
        fallback_chunk: int = 256,
    ) -> None:
        self.dump_file = dump_file
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.oo_cutoff = float(oo_cutoff)
        self.hoo_cutoff = float(hoo_cutoff)
        self.oh_cutoff = float(oh_cutoff)
        self.neighbor_cutoff = float(max(neighbor_cutoff, oo_cutoff))
        self.validation_frames = int(max(validation_frames, 1))
        self.fallback_chunk = int(max(fallback_chunk, 1))

        self.u = mda.Universe(dump_file, format="LAMMPSDUMP")
        self.n_frames_total = len(self.u.trajectory)

        self.start_frame = 0 if start_frame is None else int(start_frame)
        self.end_frame = self.n_frames_total if end_frame is None else int(end_frame)
        if not (0 <= self.start_frame < self.end_frame <= self.n_frames_total):
            raise ValueError(
                f"Invalid frame range: start={self.start_frame}, end={self.end_frame}, "
                f"total={self.n_frames_total}"
            )

        self.oxygen_selection = oxygen_selection
        self.hydrogen_selection = hydrogen_selection
        self.O_atoms = self.u.select_atoms(oxygen_selection)
        self.H_atoms = self.u.select_atoms(hydrogen_selection)

        if len(self.O_atoms) < 2:
            raise ValueError("Need at least two oxygen atoms to compute zeta.")
        if len(self.H_atoms) != 2 * len(self.O_atoms):
            raise ValueError(
                f"Expected 2 hydrogens per oxygen, but found {len(self.H_atoms)} H for "
                f"{len(self.O_atoms)} O."
            )

        self.O_indices = np.asarray(self.O_atoms.indices, dtype=np.int64)
        self.H_indices = np.asarray(self.H_atoms.indices, dtype=np.int64)
        self.n_oxygen = len(self.O_indices)
        self.n_hydrogen = len(self.H_indices)
        self.O_ids = self._safe_ids(self.O_atoms)
        self.H_ids = self._safe_ids(self.H_atoms)
        self._global_to_olocal = {int(idx): i for i, idx in enumerate(self.O_indices)}

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

    def _reference_frames_for_validation(self) -> List[int]:
        frames = np.linspace(
            self.start_frame,
            self.end_frame - 1,
            num=min(self.validation_frames, self.end_frame - self.start_frame),
            dtype=int,
        )
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
            residues = self.u.residues
        except Exception as exc:
            return None, [f"Residue-based mapping unavailable: {exc!r}"]

        if len(residues) != self.n_oxygen:
            return None, [
                f"Residue-based mapping skipped: number of residues ({len(residues)}) "
                f"!= number of oxygens ({self.n_oxygen})."
            ]

        O_set = set(int(x) for x in self.O_indices.tolist())
        H_set = set(int(x) for x in self.H_indices.tolist())
        mapping = np.full((self.n_oxygen, 2), -1, dtype=np.int64)

        for res in residues:
            res_O = [int(a.index) for a in res.atoms if int(a.index) in O_set]
            res_H = [int(a.index) for a in res.atoms if int(a.index) in H_set]
            if len(res_O) != 1 or len(res_H) != 2:
                return None, [
                    "Residue-based mapping failed: not every residue contains exactly "
                    "1 oxygen and 2 hydrogens."
                ]
            oi = self._global_to_olocal[res_O[0]]
            mapping[oi] = np.sort(np.asarray(res_H, dtype=np.int64))

        notes.append("Water mapping built from residue membership.")
        return mapping, notes

    def _mapping_from_sequential_order(self) -> Tuple[Optional[np.ndarray], List[str]]:
        H_set = set(int(x) for x in self.H_indices.tolist())
        mapping = np.full((self.n_oxygen, 2), -1, dtype=np.int64)
        notes: List[str] = []

        n_atoms = len(self.u.atoms)
        for local_i, global_o in enumerate(self.O_indices):
            h1 = int(global_o) + 1
            h2 = int(global_o) + 2
            if h2 >= n_atoms:
                return None, ["Sequential O-H-H mapping failed: ran past atom list."]
            if h1 not in H_set or h2 not in H_set:
                return None, [
                    "Sequential O-H-H mapping failed: the two atoms after an oxygen "
                    "are not both hydrogens."
                ]
            mapping[local_i] = np.asarray([h1, h2], dtype=np.int64)

        notes.append("Water mapping built from sequential O-H-H topology order.")
        return mapping, notes

    def _mapping_from_geometry(self) -> Tuple[Optional[np.ndarray], List[str]]:
        notes: List[str] = []
        self.u.trajectory[self.start_frame]
        O_pos = np.asarray(self.O_atoms.positions, dtype=np.float32)
        H_pos = np.asarray(self.H_atoms.positions, dtype=np.float32)
        box = np.asarray(self.u.dimensions, dtype=np.float32)

        pairs, dists = capped_distance(
            O_pos,
            H_pos,
            max_cutoff=self.oh_cutoff,
            box=box,
            return_distances=True,
        )

        if len(pairs) == 0:
            return None, [
                f"Geometry-based mapping failed: no O-H pairs found within "
                f"oh_cutoff={self.oh_cutoff:.3f} Å."
            ]

        nearest_olocal_for_h = np.full(self.n_hydrogen, -1, dtype=np.int64)
        nearest_dist_for_h = np.full(self.n_hydrogen, np.inf, dtype=np.float64)

        for (olocal, hlocal), dist in zip(pairs, dists):
            if dist < nearest_dist_for_h[hlocal]:
                nearest_dist_for_h[hlocal] = float(dist)
                nearest_olocal_for_h[hlocal] = int(olocal)

        mapping_lists: List[List[int]] = [[] for _ in range(self.n_oxygen)]
        for hlocal, olocal in enumerate(nearest_olocal_for_h):
            if olocal >= 0:
                mapping_lists[int(olocal)].append(int(self.H_indices[hlocal]))

        bad = [i for i, hs in enumerate(mapping_lists) if len(hs) != 2]
        if bad:
            return None, [
                "Geometry-based mapping failed: after assigning each hydrogen to its "
                "nearest oxygen, not every oxygen had exactly 2 hydrogens."
            ]

        mapping = np.empty((self.n_oxygen, 2), dtype=np.int64)
        for i, hs in enumerate(mapping_lists):
            mapping[i] = np.sort(np.asarray(hs, dtype=np.int64))

        notes.append(
            "Water mapping built geometrically from the reference frame by assigning "
            "each hydrogen to its nearest oxygen within oh_cutoff."
        )
        return mapping, notes

    def _validate_mapping(
        self,
        mapping: np.ndarray,
        strategy: str,
        inherited_notes: Sequence[str],
    ) -> Tuple[bool, MappingReport]:
        frames = self._reference_frames_for_validation()
        max_oh = 0.0
        bad_pairs = 0
        notes = list(inherited_notes)

        O_idx = self.O_indices
        H_idx = mapping

        for fr in frames:
            self.u.trajectory[fr]
            positions = np.asarray(self.u.atoms.positions, dtype=np.float64)
            box = np.asarray(self.u.dimensions, dtype=np.float64)

            O_pos = positions[O_idx]                        # (nO, 3)
            H_pos = positions[H_idx.reshape(-1)]           # (2*nO, 3)
            O_rep = np.repeat(O_pos, 2, axis=0)
            vec = minimize_vectors(H_pos - O_rep, box)
            dist = np.linalg.norm(vec, axis=1)

            frame_max = float(np.max(dist)) if dist.size else 0.0
            max_oh = max(max_oh, frame_max)
            bad_pairs += int(np.count_nonzero(dist > self.oh_cutoff))

        if bad_pairs > 0:
            notes.append(
                f"Mapping validation failed: {bad_pairs} sampled O-H pairs exceeded "
                f"oh_cutoff={self.oh_cutoff:.3f} Å."
            )
            ok = False
        else:
            notes.append("Mapping validation passed on sampled frames.")
            ok = True

        report = MappingReport(
            strategy=strategy,
            reference_frame=self.start_frame,
            n_oxygen=self.n_oxygen,
            n_hydrogen=self.n_hydrogen,
            max_oh_distance_sampled=max_oh,
            bad_oh_pairs_sampled=bad_pairs,
            sampled_frames=frames,
            notes=notes,
        )
        return ok, report

    def _build_water_mapping(self) -> Tuple[np.ndarray, MappingReport]:
        candidates = [
            ("residue", self._mapping_from_residues),
            ("sequential_OHH", self._mapping_from_sequential_order),
            ("geometry", self._mapping_from_geometry),
        ]

        last_report: Optional[MappingReport] = None
        for strategy, builder in candidates:
            mapping, notes = builder()
            if mapping is None:
                last_report = MappingReport(
                    strategy=strategy,
                    reference_frame=self.start_frame,
                    n_oxygen=self.n_oxygen,
                    n_hydrogen=self.n_hydrogen,
                    max_oh_distance_sampled=float("nan"),
                    bad_oh_pairs_sampled=-1,
                    sampled_frames=[],
                    notes=list(notes),
                )
                continue

            ok, report = self._validate_mapping(mapping, strategy, notes)
            last_report = report
            if ok:
                return mapping, report

        detail = "\n".join(
            f"- {note}"
            for note in (last_report.notes if last_report is not None else ["Unknown failure"])
        )
        raise ValueError(
            "Failed to build a reliable O->H mapping for water molecules.\n" + detail
        )

    def _min_hoo_angle_deg(self, o_local_i: int, o_local_j: int, positions: np.ndarray, box: np.ndarray) -> float:
        gi = int(self.O_indices[o_local_i])
        gj = int(self.O_indices[o_local_j])

        pos_Oi = positions[gi]
        pos_Oj = positions[gj]
        oo_vec = minimize_vectors((pos_Oj - pos_Oi).reshape(1, 3), box).reshape(3)
        oo_norm = float(np.linalg.norm(oo_vec))
        if oo_norm == 0.0:
            return 0.0

        min_angle = 180.0

        # Hydrogens attached to i: angle H_i - O_i ... O_j
        for h_global in self.o_to_h_global[o_local_i]:
            pos_H = positions[int(h_global)]
            oh_vec = minimize_vectors((pos_H - pos_Oi).reshape(1, 3), box).reshape(3)
            oh_norm = float(np.linalg.norm(oh_vec))
            if oh_norm == 0.0:
                continue
            cos_theta = np.dot(oh_vec, oo_vec) / (oh_norm * oo_norm)
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
            if angle_deg < min_angle:
                min_angle = angle_deg

        # Hydrogens attached to j: angle H_j - O_j ... O_i
        minus_oo_vec = -oo_vec
        for h_global in self.o_to_h_global[o_local_j]:
            pos_H = positions[int(h_global)]
            oh_vec = minimize_vectors((pos_H - pos_Oj).reshape(1, 3), box).reshape(3)
            oh_norm = float(np.linalg.norm(oh_vec))
            if oh_norm == 0.0:
                continue
            cos_theta = np.dot(oh_vec, minus_oo_vec) / (oh_norm * oo_norm)
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
            if angle_deg < min_angle:
                min_angle = angle_deg

        return min_angle

    @staticmethod
    def _fmt_float(value: float) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return ""
        return f"{float(value):.8f}"

    def run_complete(self) -> Dict[str, str]:
        hb_count_file = os.path.join(self.out_dir, "hb_counts_per_idx.csv")
        max_dist_file = os.path.join(self.out_dir, "max_distance_per_idx.csv")
        nhb_dist_file = os.path.join(self.out_dir, "nhb_min_distances.csv")
        zeta_file = os.path.join(self.out_dir, "zeta.csv")
        zeta_valid_file = os.path.join(self.out_dir, "zeta_valid.csv")
        frame_check_file = os.path.join(self.out_dir, "self_check_frame_summary.csv")
        summary_json = os.path.join(self.out_dir, "self_check_summary.json")

        total_rows = 0
        total_valid_zeta = 0
        total_no_hbond = 0
        total_missing_nonhb_after_fallback = 0
        total_fallback_atoms = 0
        frame_summaries: List[Dict[str, object]] = []

        with open(hb_count_file, "w", newline="") as fhb, \
             open(max_dist_file, "w", newline="") as fmax, \
             open(nhb_dist_file, "w", newline="") as fnhb, \
             open(zeta_file, "w", newline="") as fzeta, \
             open(zeta_valid_file, "w", newline="") as fzeta_valid, \
             open(frame_check_file, "w", newline="") as fcheck:

            whb = csv.writer(fhb)
            wmax = csv.writer(fmax)
            wnhb = csv.writer(fnhb)
            wzeta = csv.writer(fzeta)
            wzeta_valid = csv.writer(fzeta_valid)
            wcheck = csv.writer(fcheck)

            whb.writerow(["frame", "O_idx", "O_id", "hb_count"])
            wmax.writerow(["frame", "O_idx", "O_id", "max_hb_distance"])
            wnhb.writerow(["frame", "O_idx", "O_id", "min_nonhb_distance", "source"])
            wzeta.writerow(
                [
                    "frame",
                    "O_idx",
                    "O_id",
                    "hb_count",
                    "max_hb_distance",
                    "min_nonhb_distance",
                    "zeta",
                    "status",
                    "nonhb_source",
                ]
            )
            wzeta_valid.writerow(
                ["frame", "O_idx", "O_id", "hb_count", "max_hb_distance", "min_nonhb_distance", "zeta"]
            )
            wcheck.writerow(
                [
                    "frame",
                    "n_oxygen",
                    "n_candidate_pairs_oo_cutoff",
                    "n_hbond_pairs",
                    "n_hbonded_oxygen",
                    "n_no_hbond",
                    "n_fallback_atoms",
                    "n_missing_nonhb_after_fallback",
                    "n_valid_zeta",
                    "mean_hb_count",
                    "max_hb_count",
                ]
            )

            for ts in tqdm(
                self.u.trajectory[self.start_frame:self.end_frame],
                desc="frames",
            ):
                frame = int(ts.frame)
                positions = np.asarray(self.u.atoms.positions, dtype=np.float64)
                O_pos = np.asarray(self.O_atoms.positions, dtype=np.float32)
                box = np.asarray(ts.dimensions, dtype=np.float32)

                hb_count = np.zeros(self.n_oxygen, dtype=np.int32)
                max_hb_dist = np.full(self.n_oxygen, np.nan, dtype=np.float64)
                min_nonhb_dist = np.full(self.n_oxygen, np.nan, dtype=np.float64)
                nonhb_source = np.array(["within_cutoff"] * self.n_oxygen, dtype=object)
                hb_neighbor_lists: List[set] = [set() for _ in range(self.n_oxygen)]
                hb_pair_set = set()

                # Stage A: identify H-bonded water-water pairs under the HOO criterion.
                ns_hb = FastNS(float(self.oo_cutoff), O_pos, box=box)
                hb_res = ns_hb.self_search()
                oo_pairs = hb_res.get_pairs()
                oo_dists = hb_res.get_pair_distances()

                n_hbond_pairs = 0
                for (i, j), dist in zip(oo_pairs, oo_dists):
                    angle_deg = self._min_hoo_angle_deg(int(i), int(j), positions, box)
                    if angle_deg <= self.hoo_cutoff:
                        hb_pair_set.add((int(i), int(j)))
                        hb_pair_set.add((int(j), int(i)))
                        hb_neighbor_lists[int(i)].add(int(j))
                        hb_neighbor_lists[int(j)].add(int(i))
                        hb_count[int(i)] += 1
                        hb_count[int(j)] += 1
                        if np.isnan(max_hb_dist[int(i)]) or dist > max_hb_dist[int(i)]:
                            max_hb_dist[int(i)] = float(dist)
                        if np.isnan(max_hb_dist[int(j)]) or dist > max_hb_dist[int(j)]:
                            max_hb_dist[int(j)] = float(dist)
                        n_hbond_pairs += 1

                # Stage B: nearest non-HB distance.
                ns_all = FastNS(float(self.neighbor_cutoff), O_pos, box=box)
                all_res = ns_all.self_search()
                all_pairs = all_res.get_pairs()
                all_dists = all_res.get_pair_distances()

                for (i, j), dist in zip(all_pairs, all_dists):
                    ii = int(i)
                    jj = int(j)
                    if (ii, jj) in hb_pair_set:
                        continue
                    if np.isnan(min_nonhb_dist[ii]) or dist < min_nonhb_dist[ii]:
                        min_nonhb_dist[ii] = float(dist)
                    if np.isnan(min_nonhb_dist[jj]) or dist < min_nonhb_dist[jj]:
                        min_nonhb_dist[jj] = float(dist)

                missing_nonhb = np.where(np.isnan(min_nonhb_dist))[0]
                fallback_atoms = 0

                if len(missing_nonhb) > 0 and self.n_oxygen > 1:
                    for start in range(0, len(missing_nonhb), self.fallback_chunk):
                        chunk = np.asarray(missing_nonhb[start:start + self.fallback_chunk], dtype=np.int64)
                        dist_mat = distance_array(O_pos[chunk], O_pos, box=box)
                        for row, i in enumerate(chunk):
                            dist_row = dist_mat[row]
                            dist_row[int(i)] = np.inf
                            if hb_neighbor_lists[int(i)]:
                                dist_row[list(hb_neighbor_lists[int(i)])] = np.inf
                            jmin = int(np.argmin(dist_row))
                            dmin = float(dist_row[jmin])
                            if np.isfinite(dmin):
                                min_nonhb_dist[int(i)] = dmin
                                nonhb_source[int(i)] = "fallback_full"
                                fallback_atoms += 1

                missing_after_fallback = int(np.count_nonzero(np.isnan(min_nonhb_dist)))
                n_hbonded_oxygen = int(np.count_nonzero(hb_count > 0))
                n_no_hbond = int(self.n_oxygen - n_hbonded_oxygen)
                n_valid_zeta = 0

                for i in range(self.n_oxygen):
                    O_idx = int(self.O_indices[i])
                    O_id = int(self.O_ids[i])
                    hbct = int(hb_count[i])
                    maxd = max_hb_dist[i]
                    mind = min_nonhb_dist[i]
                    source = str(nonhb_source[i])

                    whb.writerow([frame, O_idx, O_id, hbct])
                    if hbct > 0:
                        wmax.writerow([frame, O_idx, O_id, self._fmt_float(maxd)])
                    if np.isfinite(mind):
                        wnhb.writerow([frame, O_idx, O_id, self._fmt_float(mind), source])

                    if hbct == 0:
                        status = "no_hbond_partner"
                        zeta = np.nan
                    elif not np.isfinite(mind):
                        status = "no_nonhb_partner_found_after_fallback"
                        zeta = np.nan
                    else:
                        status = "ok"
                        zeta = float(mind - maxd)
                        n_valid_zeta += 1
                        wzeta_valid.writerow(
                            [
                                frame,
                                O_idx,
                                O_id,
                                hbct,
                                self._fmt_float(maxd),
                                self._fmt_float(mind),
                                self._fmt_float(zeta),
                            ]
                        )

                    wzeta.writerow(
                        [
                            frame,
                            O_idx,
                            O_id,
                            hbct,
                            self._fmt_float(maxd),
                            self._fmt_float(mind),
                            self._fmt_float(zeta),
                            status,
                            source if np.isfinite(mind) else "",
                        ]
                    )
                    total_rows += 1

                mean_hb_count = float(np.mean(hb_count))
                max_hb_count = int(np.max(hb_count)) if len(hb_count) else 0

                wcheck.writerow(
                    [
                        frame,
                        self.n_oxygen,
                        len(oo_pairs),
                        n_hbond_pairs,
                        n_hbonded_oxygen,
                        n_no_hbond,
                        fallback_atoms,
                        missing_after_fallback,
                        n_valid_zeta,
                        f"{mean_hb_count:.8f}",
                        max_hb_count,
                    ]
                )

                frame_summaries.append(
                    {
                        "frame": frame,
                        "n_oxygen": self.n_oxygen,
                        "n_candidate_pairs_oo_cutoff": int(len(oo_pairs)),
                        "n_hbond_pairs": int(n_hbond_pairs),
                        "n_hbonded_oxygen": n_hbonded_oxygen,
                        "n_no_hbond": n_no_hbond,
                        "n_fallback_atoms": int(fallback_atoms),
                        "n_missing_nonhb_after_fallback": int(missing_after_fallback),
                        "n_valid_zeta": int(n_valid_zeta),
                        "mean_hb_count": mean_hb_count,
                        "max_hb_count": max_hb_count,
                    }
                )

                total_valid_zeta += n_valid_zeta
                total_no_hbond += n_no_hbond
                total_missing_nonhb_after_fallback += missing_after_fallback
                total_fallback_atoms += fallback_atoms

                if frame % 100 == 0:
                    gc.collect()

        summary = {
            "dump_file": self.dump_file,
            "frame_range": [self.start_frame, self.end_frame],
            "n_frames_analyzed": self.end_frame - self.start_frame,
            "n_oxygen": self.n_oxygen,
            "n_hydrogen": self.n_hydrogen,
            "parameters": {
                "oxygen_selection": self.oxygen_selection,
                "hydrogen_selection": self.hydrogen_selection,
                "oo_cutoff_A": self.oo_cutoff,
                "hoo_cutoff_deg": self.hoo_cutoff,
                "oh_cutoff_A": self.oh_cutoff,
                "neighbor_cutoff_A": self.neighbor_cutoff,
                "fallback_chunk": self.fallback_chunk,
            },
            "water_mapping": {
                "strategy": self.mapping_report.strategy,
                "reference_frame": self.mapping_report.reference_frame,
                "sampled_frames": self.mapping_report.sampled_frames,
                "max_oh_distance_sampled_A": self.mapping_report.max_oh_distance_sampled,
                "bad_oh_pairs_sampled": self.mapping_report.bad_oh_pairs_sampled,
                "notes": self.mapping_report.notes,
            },
            "totals": {
                "rows_written_in_zeta_csv": total_rows,
                "valid_zeta_rows": total_valid_zeta,
                "rows_without_hbond_partner": total_no_hbond,
                "rows_still_missing_nonhb_after_fallback": total_missing_nonhb_after_fallback,
                "rows_using_full_fallback_for_nonhb": total_fallback_atoms,
            },
            "files": {
                "hb_counts": hb_count_file,
                "max_distances": max_dist_file,
                "nhb_distances": nhb_dist_file,
                "zeta_all": zeta_file,
                "zeta_valid": zeta_valid_file,
                "frame_check": frame_check_file,
            },
        }

        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "hb_counts": hb_count_file,
            "max_distances": max_dist_file,
            "nhb_distances": nhb_dist_file,
            "zeta": zeta_file,
            "zeta_valid": zeta_valid_file,
            "frame_check": frame_check_file,
            "summary_json": summary_json,
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute zeta for water using the HOO hydrogen-bond criterion."
    )
    parser.add_argument("--dump_file", required=True, help="LAMMPS dump file.")
    parser.add_argument("--out_dir", default="output", help="Output directory.")
    parser.add_argument("--start_frame", type=int, default=None, help="First frame to analyze.")
    parser.add_argument("--end_frame", type=int, default=None, help="Stop frame (exclusive).")
    parser.add_argument("--oxygen_selection", default="type 1", help="MDAnalysis selection for oxygens.")
    parser.add_argument("--hydrogen_selection", default="type 2", help="MDAnalysis selection for hydrogens.")
    parser.add_argument("--oo_cutoff", type=float, default=3.5, help="O···O cutoff for HB water pairs (Å).")
    parser.add_argument("--hoo_cutoff", type=float, default=30.0, help="H-O···O cutoff for HB water pairs (degrees).")
    parser.add_argument("--oh_cutoff", type=float, default=1.25, help="O-H bond cutoff used to build/validate water mapping (Å).")
    parser.add_argument("--neighbor_cutoff", type=float, default=6.0, help="Fast first-pass cutoff for nearest non-HB search (Å).")
    parser.add_argument("--validation_frames", type=int, default=5, help="How many frames to sample when validating the O->H mapping.")
    parser.add_argument("--fallback_chunk", type=int, default=256, help="Chunk size for full-search fallback over missing non-HB partners.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    analyzer = ZetaHOOAnalyzer(
        dump_file=args.dump_file,
        out_dir=args.out_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        oxygen_selection=args.oxygen_selection,
        hydrogen_selection=args.hydrogen_selection,
        oo_cutoff=args.oo_cutoff,
        hoo_cutoff=args.hoo_cutoff,
        oh_cutoff=args.oh_cutoff,
        neighbor_cutoff=args.neighbor_cutoff,
        validation_frames=args.validation_frames,
        fallback_chunk=args.fallback_chunk,
    )

    results = analyzer.run_complete()
    print("Analysis complete.")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
