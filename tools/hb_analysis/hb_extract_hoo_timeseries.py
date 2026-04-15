from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from tqdm import tqdm

from hb_common import WaterHBondToolkit


def main() -> None:
    p = argparse.ArgumentParser(description="Extract per-frame HOO hydrogen-bond network and per-molecule HB counts")
    p.add_argument("--dump_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--oxygen_selection", default="type 1")
    p.add_argument("--hydrogen_selection", default="type 2")
    p.add_argument("--oo_cutoff", type=float, default=3.5)
    p.add_argument("--hoo_cutoff", type=float, default=30.0)
    p.add_argument("--oh_cutoff", type=float, default=1.25)
    p.add_argument("--start_frame", type=int, default=None)
    p.add_argument("--end_frame", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    toolkit = WaterHBondToolkit(
        dump_file=args.dump_file,
        oxygen_selection=args.oxygen_selection,
        hydrogen_selection=args.hydrogen_selection,
        oo_cutoff=args.oo_cutoff,
        hoo_cutoff=args.hoo_cutoff,
        oh_cutoff=args.oh_cutoff,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    edges_csv = os.path.join(args.out_dir, "hb_edges.csv")
    counts_csv = os.path.join(args.out_dir, "hb_counts.csv")
    frame_csv = os.path.join(args.out_dir, "hb_frame_summary.csv")
    meta_json = os.path.join(args.out_dir, "hb_meta.json")

    with open(edges_csv, "w", newline="") as fedges, open(counts_csv, "w", newline="") as fcounts, open(frame_csv, "w", newline="") as fframe:
        we = csv.writer(fedges)
        wc = csv.writer(fcounts)
        wf = csv.writer(fframe)
        we.writerow(["frame", "O_idx_i", "O_idx_j", "atom_id_i", "atom_id_j", "oo_distance", "min_hoo_angle_deg"])
        wc.writerow(["frame", "O_idx", "atom_id", "n_hb"])
        wf.writerow(["frame", "n_edges", "mean_n_hb", "frac_4coord", "frac_3coord", "frac_5coord"])

        for frame in tqdm(range(toolkit.start_frame, toolkit.end_frame), desc="Extract HB network"):
            edges, dists, angles = toolkit.compute_frame_network(frame)
            counts = toolkit.count_hbonds(edges)
            for (i, j), d, a in zip(edges, dists, angles):
                we.writerow([frame, int(i), int(j), int(toolkit.O_ids[int(i)]), int(toolkit.O_ids[int(j)]), f"{float(d):.8f}", f"{float(a):.8f}"])
            for i, n_hb in enumerate(counts):
                wc.writerow([frame, i, int(toolkit.O_ids[i]), int(n_hb)])
            wf.writerow([
                frame,
                int(edges.shape[0]),
                f"{float(np.mean(counts)):.8f}",
                f"{float(np.mean(counts == 4)):.8f}",
                f"{float(np.mean(counts == 3)):.8f}",
                f"{float(np.mean(counts == 5)):.8f}",
            ])

    meta = {
        "dump_file": args.dump_file,
        "oo_cutoff": args.oo_cutoff,
        "hoo_cutoff": args.hoo_cutoff,
        "oh_cutoff": args.oh_cutoff,
        "start_frame": toolkit.start_frame,
        "end_frame": toolkit.end_frame,
        "n_frames": toolkit.end_frame - toolkit.start_frame,
        "n_oxygen": toolkit.n_oxygen,
        "mapping_strategy": toolkit.mapping_report.strategy,
        "mapping_report": {
            "max_oh_distance_sampled": toolkit.mapping_report.max_oh_distance_sampled,
            "bad_oh_pairs_sampled": toolkit.mapping_report.bad_oh_pairs_sampled,
            "sampled_frames": toolkit.mapping_report.sampled_frames,
            "notes": toolkit.mapping_report.notes,
        },
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
