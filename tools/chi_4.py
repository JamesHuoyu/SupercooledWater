import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import apply_PBC
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
from numba import jit, prange


class Chi4Calculator:
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

    def shear_correction(self, shear_rate, time_step, ref_y: float = 25.0):
        T = self.coords.shape[0]
        y = self.coords[:, :, 1] - ref_y  # shape (T, N)
        gamma_dt = shear_rate * time_step
        shear_disp = gamma_dt * np.cumsum(y, axis=0)  # shape (T, N)
        self.coords[:, :, 0] -= shear_disp

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _compute_overlap_numba(coords, t0, max_tau, n_particles, a=1.0):
        n_tau = min(coords.shape[0] - t0, max_tau)
        Q_t0 = np.zeros(n_tau)
        Q2_t0 = np.zeros(n_tau)
        ref_pos = coords[t0]

        for tau in prange(n_tau):
            current_pos = coords[t0 + tau]  # shape(n_particles, 3)
            dr = current_pos - ref_pos
            dist = np.sqrt(np.sum(dr**2, axis=1))  # shape(n_particles,)
            overlap = np.sum(dist <= a) / n_particles
            Q_t0[tau] = overlap
            Q2_t0[tau] = overlap**2

        return Q_t0, Q2_t0

    def compute_overlap_for_origin(self, t0: int):
        return self._compute_overlap_numba(
            self.coords, t0, self.frames - t0, len(self.O_atoms), a=0.65
        )

    def time_origin_average(self, max_tau: int = None) -> np.ndarray:
        max_tau = max_tau or self.frames
        Q_accum = np.zeros(max_tau)
        Q2_accum = np.zeros(max_tau)
        count = np.zeros(max_tau, dtype=int)

        for t0 in tqdm(range(self.frames), desc="Calculating chi4"):
            Q_t0, Q2_t0 = self.compute_overlap_for_origin(t0)
            valid_tau = min(len(Q_t0), max_tau)

            Q_accum[:valid_tau] += Q_t0[:valid_tau]
            Q2_accum[:valid_tau] += Q2_t0[:valid_tau]
            count[:valid_tau] += 1

        # 平均化
        valid_mask = count > 0
        Q_accum[valid_mask] /= count[valid_mask]
        Q2_accum[valid_mask] /= count[valid_mask]

        # 计算chi4
        chi4_values = np.zeros_like(Q_accum)
        chi4_values[valid_mask] = (Q2_accum[valid_mask] - Q_accum[valid_mask] ** 2) * (
            self.n_particles
        )

        return chi4_values


if __name__ == "__main__":
    pathfiles = [
        "/home/debian/water/TIP4P/Ice/225/dump_225_test.lammpstrj"
        # "/home/debian/water/TIP4P/Ice/test/traj_1e-6_225_100000.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/test/traj_5e-6_225_100000.lammpstrj",
        # # "/home/debian/water/TIP4P/Ice/test/traj_5e-5_225_100000.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/test/traj_1e-4_225_100000.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/225/shear/traj_5e-4_225.0.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/225/dump_225_test.lammpstrj"
        # "/home/debian/water/TIP4P/Ice/test/traj_1e-5_225_100000.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/test/traj_5e-5_225_100000.lammpstrj",
        # "/home/debian/water/TIP4P/Ice/test/traj_1e-4_225_100000.lammpstrj",
    ]
    output_h5 = "/home/debian/water/TIP4P/Ice/test/h5/chi4_results.h5"
    # output_h5 = "/home/debian/water/TIP4P/2005/Tanaka_2018/rst/equili_chi4_results.h5"

    store = pd.HDFStore(output_h5)

    start_index = 0  # 跳过前1500帧以避免初始非平衡影响
    # start_index = 25000
    # start_index = 0  # 不跳过任何帧

    for pathfile in pathfiles:
        time_step = 0.2  # 25fs
        u = mda.Universe(pathfile, format="LAMMPSDUMP")
        # shear_rate = float(pathfile.split("traj_")[-1].split("_225")[0]) * 1e3
        # print(f"shear_rate extracted: {shear_rate} 1/ps")
        # chi4_calculator = Chi4Calculator(
        #     u, shear_rate=shear_rate, time_step=time_step, start_index=start_index
        # )  # shear_rate in 1/ps(7.5e-2 1/fs)
        chi4_calculator = Chi4Calculator(u, start_index=start_index)  # 无剪切流
        chi4_values = chi4_calculator.time_origin_average()
        times = np.arange(len(chi4_values)) * time_step

        df = pd.DataFrame({"time_ps": times, "chi4": chi4_values})
        # filename = pathfile.split("traj_")[-1].split("_225")[0]
        filename = "equili"
        store.put(filename, df, format="table")
        print(f"Saved chi4 results to {output_h5} under key {filename}")
    store.close()
    print("All chi4 calculations completed.")
