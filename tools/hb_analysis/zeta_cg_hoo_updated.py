from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class ZetaCgCalculator:
    """
    对 zeta 做空间粗粒化。

    兼容新的 zeta 输出：
    - 可以直接读 zeta_valid.csv（只含有效值）
    - 也可以读完整 zeta.csv（包含 status 列）；默认只使用 status == 'ok' 的值

    粗粒化定义：
        zeta_cg(i) = [zeta(i) + sum_j K(r_ij) zeta(j)] / [1 + sum_j K(r_ij)]
    其中 j 是 cutoff 内的氧邻居，K(r) 默认 exp(-r/L)
    """

    def __init__(
        self,
        universe: mda.Universe,
        zeta: pd.DataFrame,
        shear_rate: float = 0.0,
        time_step: float = 1.0,
        start_index: int = 0,
        end_index: Optional[int] = None,
        cutoff: float = 3.5,
        kernel_length: float = 3.0,
        valid_status: str = "ok",
    ) -> None:
        self.universe = universe
        self.n_frames_total = len(universe.trajectory)
        self.O_atoms = self.universe.select_atoms("type 1")
        self.O_indices = self.O_atoms.indices.astype(np.int64)
        self.global_to_local = {int(g): i for i, g in enumerate(self.O_indices)}

        self.start_index = int(start_index)
        self.end_index = (
            self.n_frames_total if end_index is None else min(int(end_index), self.n_frames_total)
        )
        if self.end_index <= self.start_index:
            raise ValueError("end_index 必须大于 start_index")

        self.frames = self.end_index - self.start_index
        self.coords = np.zeros((self.frames, len(self.O_atoms), 3), dtype=np.float32)
        self.boxes = np.zeros((self.frames, 6), dtype=np.float32)

        for ts in tqdm(
            self.universe.trajectory[self.start_index : self.end_index],
            desc="Loading trajectory data",
        ):
            local_frame = ts.frame - self.start_index
            self.coords[local_frame] = self.O_atoms.positions.astype(np.float32)
            self.boxes[local_frame] = ts.dimensions.astype(np.float32)

        if shear_rate != 0.0:
            self.shear_correction(shear_rate=shear_rate, time_step=time_step)

        self.distance_cutoff = float(cutoff)
        self.kernel_length = float(kernel_length)
        self.valid_status = valid_status
        self.zeta = self._normalize_zeta_table(zeta)
        self._precompute_zeta_mappings()

    @staticmethod
    def load_zeta_csv(zeta_file: str | Path, valid_status: str = "ok") -> pd.DataFrame:
        df = pd.read_csv(zeta_file)
        if "status" in df.columns:
            df = df[df["status"] == valid_status].copy()
        required = {"frame", "O_idx", "zeta"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"zeta 文件缺少列: {sorted(missing)}")
        df = df.loc[:, ["frame", "O_idx", "zeta"]].dropna(subset=["zeta"])
        df["frame"] = df["frame"].astype(int)
        df["O_idx"] = df["O_idx"].astype(int)
        df["zeta"] = df["zeta"].astype(float)
        return df

    def _normalize_zeta_table(self, zeta: pd.DataFrame) -> pd.DataFrame:
        df = zeta.copy()
        if "status" in df.columns:
            df = df[df["status"] == self.valid_status].copy()
        required = {"frame", "O_idx", "zeta"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"zeta DataFrame 缺少列: {sorted(missing)}")
        df = df.loc[:, ["frame", "O_idx", "zeta"]].dropna(subset=["zeta"])
        df["frame"] = df["frame"].astype(int)
        df["O_idx"] = df["O_idx"].astype(int)
        df["zeta"] = df["zeta"].astype(float)
        return df

    def shear_correction(self, shear_rate: float, time_step: float) -> None:
        """
        保留原脚本的逐步去仿射修正风格。
        如果你已经在轨迹导出前做过共剪切坐标变换，请把 shear_rate 设为 0。
        """
        for frame in tqdm(range(1, self.frames), desc="Applying shear correction"):
            y_prev = self.coords[frame - 1, :, 1]
            self.coords[frame:, :, 0] -= shear_rate * time_step * y_prev

    def _precompute_zeta_mappings(self) -> None:
        self.zeta_mappings: list[Dict[int, float]] = []
        grouped = {int(frame): g for frame, g in self.zeta.groupby("frame", sort=False)}
        for frame_idx in tqdm(range(self.frames), desc="Precomputing zeta mappings"):
            frame_number = frame_idx + self.start_index
            frame_df = grouped.get(frame_number)
            mapping: Dict[int, float] = {}
            if frame_df is not None:
                for row in frame_df.itertuples(index=False):
                    mapping[int(row.O_idx)] = float(row.zeta)
            self.zeta_mappings.append(mapping)

    def get_zeta_for_frame(self, frame_idx: int) -> Dict[int, float]:
        return self.zeta_mappings[frame_idx]

    def get_neighbor_indices(self, frame_idx: int) -> Dict[int, list[tuple[int, float]]]:
        box = self.boxes[frame_idx]
        positions = self.coords[frame_idx]
        ns = FastNS(cutoff=self.distance_cutoff, box=box, coords=positions, pbc=True)
        results = ns.self_search()
        neighbor_dict: Dict[int, list[tuple[int, float]]] = defaultdict(list)
        for pair, distance in zip(results.get_pairs(), results.get_pair_distances()):
            i, j = int(pair[0]), int(pair[1])
            gi, gj = int(self.O_indices[i]), int(self.O_indices[j])
            d = float(distance)
            neighbor_dict[gi].append((gj, d))
            neighbor_dict[gj].append((gi, d))
        return neighbor_dict

    @staticmethod
    def kernel_function(distance: float, kernel_length: float) -> float:
        return float(np.exp(-distance / kernel_length))

    def calculate_zeta_cg_for_frame(self, frame_idx: int) -> Dict[int, float]:
        zeta_values = self.get_zeta_for_frame(frame_idx)
        neighbor_dict = self.get_neighbor_indices(frame_idx)
        zeta_cg: Dict[int, float] = {}
        for atom_idx, zeta_value in zeta_values.items():
            weighted_sum = zeta_value
            weight_total = 1.0
            for neighbor_idx, distance in neighbor_dict.get(atom_idx, []):
                if neighbor_idx not in zeta_values:
                    continue
                w = self.kernel_function(distance, self.kernel_length)
                weighted_sum += zeta_values[neighbor_idx] * w
                weight_total += w
            zeta_cg[atom_idx] = weighted_sum / weight_total
        return zeta_cg

    def calculate_all_frames(self) -> Iterable[tuple[int, Dict[int, float]]]:
        for frame_idx in tqdm(range(self.frames), desc="Calculating zeta_cg"):
            yield frame_idx + self.start_index, self.calculate_zeta_cg_for_frame(frame_idx)

    def add_rolling_time_average(self, tau4_frames: int) -> Dict[int, list[float]]:
        atom_time_series: Dict[int, list[float]] = defaultdict(list)
        for _, zeta_cg in tqdm(
            list(self.calculate_all_frames()), desc="Collecting zeta_cg time series"
        ):
            for atom_idx, value in zeta_cg.items():
                atom_time_series[atom_idx].append(value)

        self.zeta_cg_smoothed: Dict[int, list[float]] = {}
        for atom_idx, values in atom_time_series.items():
            arr = pd.Series(values).rolling(window=tau4_frames, center=True, min_periods=1).mean()
            self.zeta_cg_smoothed[atom_idx] = arr.tolist()
        return self.zeta_cg_smoothed

    def get_smoothed_zeta_cg_distribution(self, target_frame: Optional[int] = None) -> np.ndarray:
        if not hasattr(self, "zeta_cg_smoothed"):
            raise ValueError("请先调用 add_rolling_time_average")
        vals = []
        for series in self.zeta_cg_smoothed.values():
            if target_frame is None:
                vals.extend(series)
            elif 0 <= target_frame < len(series):
                vals.append(series[target_frame])
        return np.asarray(vals, dtype=float)


def save_zeta_cg_csv(
    records: Iterable[tuple[int, Dict[int, float]]], output_csv: str | Path
) -> None:
    rows = []
    for frame_idx, mapping in records:
        for O_idx, zeta_cg in mapping.items():
            rows.append((frame_idx, O_idx, zeta_cg))
    out = pd.DataFrame(rows, columns=["frame", "O_idx", "zeta_cg"])
    out.to_csv(output_csv, index=False)


def plot_distribution(csv_file: str | Path, png_file: str | Path, bins: int = 300) -> None:
    df = pd.read_csv(csv_file)
    if "zeta_cg" not in df.columns or df.empty:
        raise ValueError("zeta_cg 结果为空，无法画图")
    plt.figure(figsize=(7, 5))
    plt.hist(df["zeta_cg"].to_numpy(), bins=bins, density=True)
    plt.xlabel("zeta_cg")
    plt.ylabel("Probability Density")
    plt.title("Distribution of zeta_cg")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_file, dpi=300)
    plt.close()


def main() -> None:
    """
    即便是剪切态，我们也不使用剪切修正进行处理，否则会破坏掉真实的瞬态结构特征
    """
    parser = argparse.ArgumentParser(
        description="Coarse-grain zeta computed with HOO hydrogen-bond criterion"
    )
    parser.add_argument("--dump_file", required=True)
    parser.add_argument("--zeta_file", required=True, help="zeta.csv or zeta_valid.csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_png", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=3.5)
    parser.add_argument("--kernel_length", type=float, default=3.0)
    parser.add_argument("--shear_rate", type=float, default=0.0)
    parser.add_argument("--time_step", type=float, default=1.0)
    parser.add_argument("--valid_status", default="ok")
    args = parser.parse_args()

    zeta_data = ZetaCgCalculator.load_zeta_csv(args.zeta_file, valid_status=args.valid_status)
    u = mda.Universe(args.dump_file, format="LAMMPSDUMP")
    calculator = ZetaCgCalculator(
        universe=u,
        zeta=zeta_data,
        shear_rate=args.shear_rate,
        time_step=args.time_step,
        start_index=args.start_index,
        end_index=args.end_index,
        cutoff=args.cutoff,
        kernel_length=args.kernel_length,
        valid_status=args.valid_status,
    )
    save_zeta_cg_csv(calculator.calculate_all_frames(), args.output_csv)
    if args.output_png:
        plot_distribution(args.output_csv, args.output_png)


if __name__ == "__main__":
    main()
