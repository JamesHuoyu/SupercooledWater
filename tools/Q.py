import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import apply_PBC
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
from numba import jit, prange


class QCalculator:
    def __init__(
        self,
        universe: mda.Universe,
        shear_rate: float = 0.0,
        time_step: float = 1.0,
        start_index: int = 0,
    ):
        self.universe = universe
        self.n_frames = len(universe.trajectory)
        self.n_particles = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        # 预加载轨迹数据到内存，只针对O原子
        self.frames = self.n_frames - start_index
        self.coords = np.zeros((self.frames, len(self.O_atoms), 3))
        for ts in tqdm(self.universe.trajectory[start_index:], desc="Loading trajectory data"):
            self.coords[ts.frame - start_index] = self.O_atoms.positions.copy()
        if shear_rate != 0.0:
            self.shear_correction(shear_rate, time_step)

    def shear_correction(self, shear_rate, time_step):
        for frame in tqdm(range(self.frames), desc="Applying shear correction"):
            if frame == 0:
                continue
            y_positions = self.coords[frame - 1, :, 1]
            # 修正x坐标以消除剪切流影响
            self.coords[frame:, :, 0] -= shear_rate * time_step * y_positions

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _compute_overlap_numba(coords, t0, max_tau, n_particles, a=0.6):
        n_tau = min(coords.shape[0] - t0, max_tau)
        Q_t0 = np.zeros(n_tau)
        ref_pos = coords[t0]

        for tau in prange(n_tau):
            current_pos = coords[t0 + tau]  # shape(n_particles, 3)
            dr = current_pos - ref_pos
            dist = np.sqrt(np.sum(dr**2, axis=1))  # shape(n_particles,)
            overlap = np.sum(dist <= a) / n_particles
            Q_t0[tau] = overlap

        return Q_t0

    def compute_overlap_for_origin(self, t0: int):
        return self._compute_overlap_numba(
            self.coords, t0, self.frames - t0, len(self.O_atoms), a=0.65
        )

    def time_origin_average(self, max_tau: int = None) -> np.ndarray:
        max_tau = max_tau or self.frames
        Q_accum = np.zeros(max_tau)
        count = np.zeros(max_tau, dtype=int)

        for t0 in tqdm(range(self.frames), desc="Calculating Q(t)"):
            Q_t0 = self.compute_overlap_for_origin(t0)
            valid_tau = min(len(Q_t0), max_tau)

            Q_accum[:valid_tau] += Q_t0[:valid_tau]
            count[:valid_tau] += 1

        # 平均化
        valid_mask = count > 0
        Q_accum[valid_mask] /= count[valid_mask]

        return Q_accum

    def compute_Q(self, max_tau: int = None) -> np.ndarray:
        return self.time_origin_average(max_tau=max_tau)


if __name__ == "__main__":
    pathfiles = [
        # "/home/debian/water/TIP4P/Ice/test/traj_1e-6_225_100000.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/test/traj_5e-6_225_100000.lammpstrj",
        "/home/debian/water/TIP4P/Ice/test/traj_5e-5_225_100000.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/225/dump_225_test.lammpstrj",
    ]
    output_h5 = "/home/debian/water/TIP4P/Ice/test/Q_results.h5"

    store = pd.HDFStore(output_h5)

    # start_index = 2000  # 跳过前2000帧以避免初始非平衡影响
    start_index = 2000  # 跳过前7000帧以避免初始非平衡影响

    for pathfile in pathfiles:
        # time_step = 0.05  # ps
        time_step = 0.025  # ps
        u = mda.Universe(pathfile, format="LAMMPSDUMP")
        shear_rate = float(pathfile.split("traj_")[-1].split("_225")[0])  # 从文件名中提取剪切率
        shear_rate *= 1e3  # 转换为1/ps单位
        # shear_rate = 0.0  # 1/ps
        Q_calculator = QCalculator(
            u, shear_rate=shear_rate, time_step=time_step, start_index=start_index
        )  # shear_rate in 1/ps(7.5e-2 1/fs)
        # Q_calculator = QCalculator(u, time_step=time_step, start_index=start_index)
        Q = Q_calculator.compute_Q()
        times = np.arange(len(Q)) * time_step

        df = pd.DataFrame({"time_ps": times, "Q": Q})
        filename = pathfile.split("traj_")[-1].split("_225")[0]
        # filename = "equili"
        store.put(filename, df, format="table")
        print(f"Saved Q results to {output_h5} under key {filename}")
    store.close()
    print("All Q calculations completed.")
