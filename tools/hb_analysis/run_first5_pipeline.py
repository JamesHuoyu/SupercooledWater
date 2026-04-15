from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run automated workflow for analyses 1-5")
    p.add_argument("--dump_file", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument(
        "--zeta_csv", required=True, help="zeta.csv or zeta_valid.csv from the HOO zeta script"
    )
    p.add_argument(
        "--zeta_cg_csv",
        default=None,
        help="Existing zeta_cg.csv. If omitted, it will be generated.",
    )
    p.add_argument("--old_counts_csv", default=None)
    p.add_argument("--oo_cutoff", type=float, default=3.5)
    p.add_argument("--hoo_cutoff", type=float, default=30.0)
    p.add_argument("--oh_cutoff", type=float, default=1.25)
    p.add_argument("--cg_cutoff", type=float, default=3.5)
    p.add_argument("--kernel_length", type=float, default=3.0)
    p.add_argument("--max_lag", type=int, default=None)
    args = p.parse_args()

    work = Path(args.work_dir)
    hb_dir = work / "hb_network"
    hb_post = work / "hb_post"
    zeta_dir = work / "zeta_post"
    hb_dir.mkdir(parents=True, exist_ok=True)
    hb_post.mkdir(parents=True, exist_ok=True)
    zeta_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    # run(
    #     [
    #         py,
    #         "/home/debian/water/tools/hb_analysis/hb_extract_hoo_timeseries.py",
    #         "--dump_file",
    #         args.dump_file,
    #         "--out_dir",
    #         str(hb_dir),
    #         "--oo_cutoff",
    #         str(args.oo_cutoff),
    #         "--hoo_cutoff",
    #         str(args.hoo_cutoff),
    #         "--oh_cutoff",
    #         str(args.oh_cutoff),
    #     ]
    # )

    cmd = [
        py,
        "hb_analysis/hb_dynamics_postprocess_fixed.py",
        "--hb_dir",
        str(hb_dir),
        "--out_dir",
        str(hb_post),
    ]
    if args.old_counts_csv:
        cmd.extend(["--old_counts_csv", args.old_counts_csv])
    if args.max_lag is not None:
        cmd.extend(["--max_lag", str(args.max_lag)])
    run(cmd)

    zeta_cg_csv = args.zeta_cg_csv
    if zeta_cg_csv is None:
        zeta_cg_csv = str(work / "zeta_cg.csv")
        run(
            [
                py,
                "hb_analysis/zeta_cg_hoo_updated.py",
                "--dump_file",
                args.dump_file,
                "--zeta_file",
                args.zeta_csv,
                "--output_csv",
                zeta_cg_csv,
                "--cutoff",
                str(args.cg_cutoff),
                "--kernel_length",
                str(args.kernel_length),
            ]
        )

    cmd = [
        py,
        "hb_analysis/zeta_dynamics_postprocess.py",
        "--zeta_csv",
        args.zeta_csv,
        "--zeta_cg_csv",
        zeta_cg_csv,
        "--out_dir",
        str(zeta_dir),
    ]
    if args.max_lag is not None:
        cmd.extend(["--max_lag", str(args.max_lag)])
    run(cmd)


if __name__ == "__main__":
    main()
