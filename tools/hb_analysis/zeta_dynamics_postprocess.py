from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hb_common import autocorrelation_mean_centered


def load_zeta(csv_file: str | Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col} not found in {csv_file}")
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    required = {"frame", "O_idx", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[np.isfinite(df[value_col].to_numpy())].copy()
    return df.sort_values(["frame", "O_idx"]).reset_index(drop=True)


def distribution_and_frame_stats(df: pd.DataFrame, value_col: str, out_dir: Path, prefix: str) -> dict:
    hist, bins = np.histogram(df[value_col].to_numpy(dtype=float), bins=250, density=True)
    mids = 0.5 * (bins[:-1] + bins[1:])
    pd.DataFrame({value_col: mids, "pdf": hist}).to_csv(out_dir / f"{prefix}_distribution.csv", index=False)
    frame_stats = df.groupby("frame")[value_col].agg(mean="mean", std="std", q10=lambda x: x.quantile(0.1), q50="median", q90=lambda x: x.quantile(0.9)).reset_index()
    frame_stats.to_csv(out_dir / f"{prefix}_frame_stats.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.hist(df[value_col].to_numpy(dtype=float), bins=250, density=True)
    plt.xlabel(value_col)
    plt.ylabel("Probability density")
    plt.title(f"Distribution of {value_col}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_distribution.png", dpi=300)
    plt.close()
    return {
        "mean": float(df[value_col].mean()),
        "std": float(df[value_col].std()),
        "median": float(df[value_col].median()),
    }


def autocorr(df: pd.DataFrame, value_col: str, out_dir: Path, prefix: str, max_lag: Optional[int] = None) -> dict:
    pivot = df.pivot(index="frame", columns="O_idx", values=value_col).sort_index(axis=0).sort_index(axis=1)
    n_frames = pivot.shape[0]
    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames for autocorrelation of {value_col}")
    if max_lag is None or max_lag >= n_frames:
        max_lag = n_frames - 1
    acs = []
    for col in pivot.columns:
        arr = pivot[col].to_numpy(dtype=float)
        valid = np.isfinite(arr)
        if np.count_nonzero(valid) < 2:
            continue
        acs.append(autocorrelation_mean_centered(arr[valid], max_lag=min(max_lag, np.count_nonzero(valid) - 1)))
    min_len = min((len(a) for a in acs), default=0)
    if min_len == 0:
        raise ValueError(f"No valid time series for autocorrelation of {value_col}")
    acs_arr = np.vstack([a[:min_len] for a in acs])
    mean_ac = acs_arr.mean(axis=0)
    std_ac = acs_arr.std(axis=0)
    pd.DataFrame({"lag": np.arange(min_len), "mean_autocorr": mean_ac, "std_autocorr": std_ac}).to_csv(out_dir / f"{prefix}_autocorr.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(min_len), mean_ac)
    plt.xlabel("Lag (frames)")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation of {value_col}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_autocorr.png", dpi=300)
    plt.close()
    return {"lag1_autocorr": float(mean_ac[1]) if len(mean_ac) > 1 else 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze zeta and zeta_cg distributions and autocorrelations")
    p.add_argument("--zeta_csv", required=True)
    p.add_argument("--zeta_cg_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--zeta_col", default="zeta")
    p.add_argument("--zeta_cg_col", default="zeta_cg")
    p.add_argument("--max_lag", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zeta_df = load_zeta(args.zeta_csv, args.zeta_col)
    zeta_cg_df = load_zeta(args.zeta_cg_csv, args.zeta_cg_col)

    summary = {
        "zeta_distribution": distribution_and_frame_stats(zeta_df, args.zeta_col, out_dir, "zeta"),
        "zeta_autocorr": autocorr(zeta_df, args.zeta_col, out_dir, "zeta", max_lag=args.max_lag),
        "zeta_cg_distribution": distribution_and_frame_stats(zeta_cg_df, args.zeta_cg_col, out_dir, "zeta_cg"),
        "zeta_cg_autocorr": autocorr(zeta_cg_df, args.zeta_cg_col, out_dir, "zeta_cg", max_lag=args.max_lag),
    }

    with open(out_dir / "zeta_postprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
