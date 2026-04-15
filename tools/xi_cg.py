from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.analysis import rdf
from numba import jit, prange
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class ZetaCgCalculator:
    def __init__(
        self,
        universe: mda.Universe,
        zeta: pd.DataFrame,
        shear_rate: float = 0.0,
        time_step: float = 1.0,
        start_index: int = 0,
        end_index: int = -1,
        cutoff: float = 3.5,
    ):
        self.universe = universe
        self.n_frames = len(universe.trajectory)
        self.n_particles = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        self.O_indices = self.O_atoms.indices
        # 预加载轨迹数据到内存，只针对O原子
        self.frames = self.n_frames - start_index
        self.coords = np.zeros((self.frames, len(self.O_atoms), 3), dtype=np.float32)
        self.boxs = np.zeros((self.frames, 6), dtype=np.float32)
        if end_index is not None or end_index != -1:
            for ts in tqdm(
                self.universe.trajectory[start_index:end_index], desc="Loading trajectory data"
            ):
                self.coords[ts.frame - start_index] = self.O_atoms.positions.copy()
                self.boxs[ts.frame - start_index] = ts.dimensions.copy()
        else:
            for ts in tqdm(self.universe.trajectory[start_index:], desc="Loading trajectory data"):
                self.coords[ts.frame - start_index] = self.O_atoms.positions.copy()
                self.boxs[ts.frame - start_index] = ts.dimensions.copy()
        if shear_rate != 0.0:
            self.shear_correction(shear_rate, time_step)
        self.zeta = zeta
        self.distance_cutoff = cutoff
        self.start_index = start_index
        self._precompute_neighbor_indices()

    def shear_correction(self, shear_rate, time_step):
        for frame in tqdm(range(self.frames), desc="Applying shear correction"):
            if frame == 0:
                continue
            y_positions = self.coords[frame - 1, :, 1]
            # 修正x坐标以消除剪切流影响
            self.coords[frame:, :, 0] -= shear_rate * time_step * y_positions

    def _precompute_neighbor_indices(self):
        self.zeta_mappings = []
        for frame_idx in tqdm(range(self.frames), desc="Precomputing neighbor indices"):
            frame_number = frame_idx + self.start_index
            frame_zeta_data = self.zeta[self.zeta["frame"] == frame_number]
            zeta_values = defaultdict(float)

            for _, row in frame_zeta_data.iterrows():
                O_idx = int(row["O_idx"])
                zeta_value = row["zeta"]
                zeta_values[O_idx] = zeta_value
            self.zeta_mappings.append(zeta_values)

    def get_zeta_for_frame(self, frame_idx: int):
        return self.zeta_mappings[frame_idx]

    def get_neighbor_indices(self, frame_idx: int):
        box = self.boxs[frame_idx]
        positions = self.coords[frame_idx]
        ns = FastNS(cutoff=self.distance_cutoff, box=box, coords=positions, pbc=True)
        results = ns.self_search()

        return self._build_neighbor_dict(results, self.O_indices)

    @staticmethod
    # @jit(nopython=True)
    def _build_neighbor_dict(results, O_indices):
        neighbor_dict = defaultdict(list)
        for distance, pair in zip(results.get_pair_distances(), results.get_pairs()):
            idx1, idx2 = pair
            O_idx1, O_idx2 = O_indices[idx1], O_indices[idx2]
            neighbor_dict[O_idx1].append((O_idx2, distance))
            neighbor_dict[O_idx2].append((O_idx1, distance))
        return neighbor_dict

    def calculate_zeta_cg_for_frame(self, frame_idx: int):
        zeta_values = self.get_zeta_for_frame(frame_idx)
        neighbor_dict = self.get_neighbor_indices(frame_idx)
        return self._compute_zeta_cg(zeta_values, neighbor_dict)

    @staticmethod
    @jit(nopython=True)
    def _kernel_function(x, L: float = 3.0):
        return np.exp(-(x / L))

    def _compute_zeta_cg(self, zeta_values, neighbor_dict):
        n_atoms = len(zeta_values)
        zeta_cg = defaultdict(float)
        for atom_idx, zeta_value in zeta_values.items():
            neighbors = neighbor_dict.get(atom_idx, [])
            weighted_sum = zeta_value
            weight_total = 1.0  # 包括自身的权重
            for neighbor_idx, distance in neighbors:
                if neighbor_idx in zeta_values:
                    weight = ZetaCgCalculator._kernel_function(distance)
                    weighted_sum += zeta_values[neighbor_idx] * weight
                    weight_total += weight
            zeta_cg[atom_idx] = weighted_sum / weight_total
        return zeta_cg

    def calculate_all_frames(self):
        for frame_idx in tqdm(range(self.frames), desc="Calculating zeta_cg for all frames"):
            zeta_cg = self.calculate_zeta_cg_for_frame(frame_idx)
            yield frame_idx + self.start_index, zeta_cg

    def add_rolling_time_average(self, tau4_frames: int):
        atom_time_series = defaultdict(list)

        for frame_idx in tqdm(range(self.frames), desc="Calculating rolling time average"):
            zeta_cg = self.calculate_zeta_cg_for_frame(frame_idx)
            for atom_idx, zeta_value in zeta_cg.items():
                atom_time_series[atom_idx].append(zeta_value)
        self.zeta_cg_smoothed = defaultdict()
        window_size = tau4_frames

        for atom_idx, values in atom_time_series.items():
            if len(values) >= window_size:
                series_array = pd.Series(values)
                smoothed_values = series_array.rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                self.zeta_cg_smoothed[atom_idx] = smoothed_values.tolist()
        return self.zeta_cg_smoothed

    def get_smoothed_zeta_cg_distribution(self, target_frame: int = None):
        if not hasattr(self, "zeta_cg_smoothed"):
            raise ValueError("请先调用add_rolling_time_average方法进行平滑处理")

        all_values = []
        for atom_idx, series in self.zeta_cg_smoothed.items():
            if target_frame is not None and target_frame < len(series):
                all_values.append(series[target_frame])
            elif target_frame is None:
                all_values.extend(series)

        return np.array(all_values)


class ZetaTimeCgCalculator:
    def __init__(
        self,
        universe: mda.Universe,
        zeta: pd.DataFrame,
        shear_rate: float = 0.0,
        time_step: float = 1.0,
        start_index: int = 0,
        time_window: int = 10,
    ):
        self.universe = universe
        self.n_frames = len(universe.trajectory)
        self.n_particles = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        self.O_indices = self.O_atoms.indices
        # 预加载轨迹数据到内存，只针对O原子
        self.frames = self.n_frames - start_index
        self.coords = np.zeros((self.frames, len(self.O_atoms), 3), dtype=np.float32)
        self.boxs = np.zeros((self.frames, 6), dtype=np.float32)
        for ts in tqdm(self.universe.trajectory[start_index:], desc="Loading trajectory data"):
            self.coords[ts.frame - start_index] = self.O_atoms.positions.copy()
            self.boxs[ts.frame - start_index] = ts.dimensions
        if shear_rate != 0.0:
            self.shear_correction(shear_rate, time_step)
        self.zeta = zeta
        self.start_index = start_index
        self.time_window = time_window

    def shear_correction(self, shear_rate, time_step):
        for frame in tqdm(range(self.frames), desc="Applying shear correction"):
            if frame == 0:
                continue
            y_positions = self.coords[frame - 1, :, 1]
            # 修正x坐标以消除剪切流影响
            self.coords[frame:, :, 0] -= shear_rate * time_step * y_positions

    def get_zeta_for_frame(self, frame_idx: int):
        frame_number = frame_idx + self.start_index
        frame_zeta_data = self.zeta[self.zeta["frame"] == frame_number]
        zeta_values = defaultdict(float)

        for _, row in frame_zeta_data.iterrows():
            O_idx = int(row["O_idx"])
            zeta_value = row["zeta"]
            zeta_values[O_idx] = zeta_value
        return zeta_values

    def calculate_zeta_time_cg_for_one_period(self, start_frame_idx: int):
        zeta_time_cg = defaultdict(float)
        for time in tqdm(range(start_frame_idx, start_frame_idx + self.time_window)):
            zeta_values = self.get_zeta_for_frame(time)
            for key, value in zeta_values.items():
                zeta_time_cg[key] += value
        for key in zeta_time_cg.keys():
            zeta_time_cg[key] /= self.time_window

        return zeta_time_cg

    def calculate_zeta_time_cg_all_periods(self):
        n_periods = self.frames // self.time_window
        for period in tqdm(range(n_periods), desc="Calculating zeta_time_cg for all periods"):
            start_frame_idx = period * self.time_window
            zeta_time_cg = self.calculate_zeta_time_cg_for_one_period(start_frame_idx)
            yield period, zeta_time_cg


if __name__ == "__main__":
    path_files = [
        "/home/debian/water/TIP4P/Ice/test/5e-6/traj_5e-6_225_100000.lammpstrj",
    ]
    zeta_file = "/home/debian/water/TIP4P/Ice/test/5e-6/zeta.csv"
    output_csv = "/home/debian/water/TIP4P/Ice/test/5e-6/zeta_cg.csv"
    zeta_data = pd.read_csv(zeta_file)
    u = mda.Universe(path_files[0], format="LAMMPSDUMP")
    # t_x = 68  # ps
    time_step = 0.025  # ps
    # tx_frame = int(t_x / time_step)
    calculator = ZetaCgCalculator(
        universe=u,
        zeta=zeta_data,
        shear_rate=0.0,
        time_step=time_step,
        start_index=0,
        # end_index=2,
        cutoff=3.5,
    )

    for frame_idx, zeta_cg in calculator.calculate_all_frames():
        with open(output_csv, "a") as f:
            for O_idx, zeta_value in zeta_cg.items():
                f.write(f"{frame_idx},{O_idx},{zeta_value}\n")
    print(f"Saved zeta_cg results to {output_csv}")

    df = pd.read_csv(output_csv, names=["frame", "O_idx", "zeta_cg"])
    plt.figure(figsize=(7, 5))
    plt.hist(df["zeta_cg"], bins=300, density=True)
    plt.xlabel("Zeta_cg")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Zeta_cg")
    plt.grid(True)
    plt.savefig("/home/debian/water/TIP4P/Ice/test/5e-6/zeta_cg.png", dpi=300)
    plt.show()
