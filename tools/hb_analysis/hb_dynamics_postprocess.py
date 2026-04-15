from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hb_common import autocorrelation_mean_centered, survival_from_lengths


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_counts(counts_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(counts_csv)
    required = {"frame", "O_idx", "n_hb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in hb_counts.csv: {missing}")
    df = df.sort_values(["frame", "O_idx"]).reset_index(drop=True)
    return df


def load_edges(edges_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(edges_csv)
    required = {"frame", "O_idx_i", "O_idx_j"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in hb_edges.csv: {missing}")
    df = df.sort_values(["frame", "O_idx_i", "O_idx_j"]).reset_index(drop=True)
    return df


def analyze_counts_distribution(counts_df: pd.DataFrame, out_dir: Path, old_counts_csv: str | None = None) -> dict:
    out: dict = {}
    frame_summary = counts_df.groupby("frame")["n_hb"].agg(mean_n_hb="mean", frac4=lambda x: np.mean(x == 4), frac3=lambda x: np.mean(x == 3), frac5=lambda x: np.mean(x == 5)).reset_index()
    frame_summary.to_csv(out_dir / "hb_count_frame_summary.csv", index=False)

    hist = counts_df["n_hb"].value_counts(normalize=True).sort_index().rename_axis("n_hb").reset_index(name="probability")
    hist.to_csv(out_dir / "hb_count_distribution.csv", index=False)
    out["new_mean_n_hb"] = float(counts_df["n_hb"].mean())
    out["new_frac4"] = float(np.mean(counts_df["n_hb"] == 4))

    plt.figure(figsize=(7, 5))
    plt.bar(hist["n_hb"], hist["probability"], width=0.8, alpha=0.8, label="new criterion")
    if old_counts_csv:
        old_df = load_counts(old_counts_csv)
        old_hist = old_df["n_hb"].value_counts(normalize=True).sort_index().rename_axis("n_hb").reset_index(name="probability")
        plt.plot(old_hist["n_hb"], old_hist["probability"], marker="o", linewidth=1.8, label="old criterion")
        compare = pd.DataFrame({
            "metric": ["mean_n_hb", "frac4", "frac3", "frac5"],
            "old": [old_df["n_hb"].mean(), np.mean(old_df["n_hb"] == 4), np.mean(old_df["n_hb"] == 3), np.mean(old_df["n_hb"] == 5)],
            "new": [counts_df["n_hb"].mean(), np.mean(counts_df["n_hb"] == 4), np.mean(counts_df["n_hb"] == 3), np.mean(counts_df["n_hb"] == 5)],
        })
        compare["delta"] = compare["new"] - compare["old"]
        compare.to_csv(out_dir / "hb_criterion_comparison_summary.csv", index=False)
        out["old_mean_n_hb"] = float(old_df["n_hb"].mean())
        out["old_frac4"] = float(np.mean(old_df["n_hb"] == 4))
    plt.xlabel("n_HB per molecule")
    plt.ylabel("Probability")
    plt.title("HB count distribution")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hb_count_distribution.png", dpi=300)
    plt.close()
    return out


def build_pair_presence(edges_df: pd.DataFrame) -> Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray]:
    frames = np.sort(edges_df["frame"].unique())
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames)}
    presence: Dict[Tuple[int, int], np.ndarray] = {}
    for row in edges_df.itertuples(index=False):
        pair = (int(row.O_idx_i), int(row.O_idx_j))
        arr = presence.setdefault(pair, np.zeros(len(frames), dtype=np.int8))
        arr[frame_to_idx[int(row.frame)]] = 1
    return presence, frames


def continuous_and_intermittent_lifetimes(edges_df: pd.DataFrame, out_dir: Path, max_lag: Optional[int] = None) -> dict:
    presence, frames = build_pair_presence(edges_df)
    run_lengths: List[int] = []
    intermittent_num: Optional[np.ndarray] = None
    intermittent_den = 0.0
    if max_lag is None:
        max_lag = max(1, len(frames) - 1)

    for arr in presence.values():
        starts = np.where((arr == 1) & (np.r_[0, arr[:-1]] == 0))[0]
        ends = np.where((arr == 1) & (np.r_[arr[1:], 0] == 0))[0]
        run_lengths.extend((ends - starts + 1).tolist())

        local = np.zeros(max_lag + 1, dtype=float)
        for lag in range(max_lag + 1):
            if lag >= len(arr):
                break
            local[lag] = float(np.dot(arr[: len(arr) - lag], arr[lag:]))
        intermittent_den += float(np.sum(arr))
        if intermittent_num is None:
            intermittent_num = local
        else:
            intermittent_num += local

    run_lengths_arr = np.asarray(run_lengths, dtype=int)
    continuous = survival_from_lengths(run_lengths_arr, max_lag=max_lag)
    intermittent = intermittent_num / max(intermittent_den, 1e-12)

    pd.DataFrame({"lag": np.arange(len(continuous)), "continuous_survival": continuous, "intermittent_correlation": intermittent}).to_csv(out_dir / "hb_lifetime_correlations.csv", index=False)
    pd.DataFrame({"continuous_run_length_frames": run_lengths_arr}).to_csv(out_dir / "hb_continuous_run_lengths.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(len(continuous)), continuous, label="S_HB(t) continuous")
    plt.plot(np.arange(len(intermittent)), intermittent, label="C_HB(t) intermittent")
    plt.xlabel("Lag (frames)")
    plt.ylabel("Correlation")
    plt.title("HB lifetimes")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hb_lifetime_correlations.png", dpi=300)
    plt.close()

    return {
        "n_pairs_seen": int(len(presence)),
        "mean_continuous_lifetime_frames": float(run_lengths_arr.mean()) if run_lengths_arr.size else 0.0,
        "median_continuous_lifetime_frames": float(np.median(run_lengths_arr)) if run_lengths_arr.size else 0.0,
    }


def per_molecule_count_autocorr(counts_df: pd.DataFrame, out_dir: Path, max_lag: Optional[int] = None) -> dict:
    pivot = counts_df.pivot(index="frame", columns="O_idx", values="n_hb").sort_index(axis=0).sort_index(axis=1)
    n_frames = pivot.shape[0]
    if max_lag is None or max_lag >= n_frames:
        max_lag = n_frames - 1
    acs = []
    for col in pivot.columns:
        ac = autocorrelation_mean_centered(pivot[col].to_numpy(dtype=float), max_lag=max_lag)
        acs.append(ac)
    acs_arr = np.vstack(acs) if acs else np.empty((0, max_lag + 1))
    mean_ac = acs_arr.mean(axis=0) if acs_arr.size else np.empty(0)
    std_ac = acs_arr.std(axis=0) if acs_arr.size else np.empty(0)
    pd.DataFrame({"lag": np.arange(len(mean_ac)), "mean_autocorr": mean_ac, "std_autocorr": std_ac}).to_csv(out_dir / "hb_count_autocorr.csv", index=False)

    per_mol_std = pivot.std(axis=0).rename("std_n_hb").reset_index()
    per_mol_std.to_csv(out_dir / "hb_count_per_molecule_fluctuation.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(len(mean_ac)), mean_ac)
    plt.xlabel("Lag (frames)")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation of per-molecule HB count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "hb_count_autocorr.png", dpi=300)
    plt.close()

    return {
        "mean_std_n_hb": float(per_mol_std["std_n_hb"].mean()) if len(per_mol_std) else 0.0,
        "lag1_autocorr": float(mean_ac[1]) if len(mean_ac) > 1 else 0.0,
    }


def partner_retention_and_exchange(edges_df: pd.DataFrame, counts_df: pd.DataFrame, out_dir: Path, max_lag: Optional[int] = None) -> dict:
    frames = np.sort(counts_df["frame"].unique())
    n_oxygen = int(counts_df["O_idx"].max()) + 1
    partners_per_frame: List[List[set[int]]] = [[set() for _ in range(n_oxygen)] for _ in range(len(frames))]
    frame_to_idx = {int(fr): i for i, fr in enumerate(frames)}
    for row in edges_df.itertuples(index=False):
        fi = frame_to_idx[int(row.frame)]
        i = int(row.O_idx_i)
        j = int(row.O_idx_j)
        partners_per_frame[fi][i].add(j)
        partners_per_frame[fi][j].add(i)

    if max_lag is None:
        max_lag = max(1, len(frames) - 1)
    retention = np.zeros(max_lag + 1, dtype=float)
    retention_counts = np.zeros(max_lag + 1, dtype=float)
    first_change_times: List[int] = []
    full_renewal_times: List[int] = []

    for o in range(n_oxygen):
        series = [partners_per_frame[t][o] for t in range(len(frames))]
        for t0, base in enumerate(series):
            if not base:
                continue
            changed_recorded = False
            full_recorded = False
            for lag in range(0, min(max_lag, len(frames) - 1 - t0) + 1):
                current = series[t0 + lag]
                retention[lag] += len(base & current) / max(len(base), 1)
                retention_counts[lag] += 1.0
                if lag > 0 and (not changed_recorded) and current != base:
                    first_change_times.append(lag)
                    changed_recorded = True
                if lag > 0 and (not full_recorded) and len(base & current) == 0:
                    full_renewal_times.append(lag)
                    full_recorded = True

    retention_curve = retention / np.maximum(retention_counts, 1.0)
    pd.DataFrame({"lag": np.arange(len(retention_curve)), "partner_retention": retention_curve}).to_csv(out_dir / "partner_retention_curve.csv", index=False)
    pd.DataFrame({"first_change_time_frames": first_change_times}).to_csv(out_dir / "partner_first_change_times.csv", index=False)
    pd.DataFrame({"full_renewal_time_frames": full_renewal_times}).to_csv(out_dir / "partner_full_renewal_times.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(len(retention_curve)), retention_curve)
    plt.xlabel("Lag (frames)")
    plt.ylabel("Retention fraction")
    plt.title("HB partner retention")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "partner_retention_curve.png", dpi=300)
    plt.close()

    return {
        "mean_first_change_time_frames": float(np.mean(first_change_times)) if first_change_times else 0.0,
        "mean_full_renewal_time_frames": float(np.mean(full_renewal_times)) if full_renewal_times else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Postprocess HOO HB network for analyses 1-4")
    p.add_argument("--hb_dir", required=True, help="Directory containing hb_edges.csv and hb_counts.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--old_counts_csv", default=None, help="Optional old-criterion hb_counts.csv for direct comparison")
    p.add_argument("--max_lag", type=int, default=None)
    args = p.parse_args()

    hb_dir = Path(args.hb_dir)
    out_dir = ensure_dir(args.out_dir)
    counts_df = load_counts(hb_dir / "hb_counts.csv")
    edges_df = load_edges(hb_dir / "hb_edges.csv")

    summary = {}
    summary["count_distribution"] = analyze_counts_distribution(counts_df, out_dir, old_counts_csv=args.old_counts_csv)
    summary["lifetimes"] = continuous_and_intermittent_lifetimes(edges_df, out_dir, max_lag=args.max_lag)
    summary["count_autocorr"] = per_molecule_count_autocorr(counts_df, out_dir, max_lag=args.max_lag)
    summary["partner_exchange"] = partner_retention_and_exchange(edges_df, counts_df, out_dir, max_lag=args.max_lag)

    with open(out_dir / "hb_postprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
