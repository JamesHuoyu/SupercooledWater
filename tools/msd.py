import MDAnalysis as mda
import pandas as pd
import numpy as np
import argparse
from numba import jit, prange
from tqdm import tqdm


class MSDCalculator:
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
            self.shear_correction(shear_rate, time_step, ref_y=25.0)

    def shear_correction(self, shear_rate, time_step, ref_y: float = 25.0):
        T = self.coords.shape[0]
        y = self.coords[:, :, 1] - ref_y  # shape (T, N)
        gamma_dt = shear_rate * time_step
        shear_disp = gamma_dt * np.cumsum(y, axis=0)  # shape (T, N)
        self.coords[:, :, 0] -= shear_disp

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_msd_numba(zs, t0, max_tau, n_particles):
        n_tau = min(zs.shape[0] - t0, max_tau)
        msd_t0 = np.zeros(n_tau)
        ref_pos = zs[t0]

        for tau in prange(n_tau):
            current_pos = zs[t0 + tau]
            dr = current_pos - ref_pos
            sq_disp = 0.0
            for i in range(n_particles):
                disp = dr[i] ** 2
                sq_disp += disp
            msd_t0[tau] = sq_disp / n_particles
        return msd_t0

    def compute_msd_for_origin(self, t0: int, idx) -> np.ndarray:
        # 计算z方向的MSD
        return self.compute_msd_numba(
            self.coords[:, :, idx], t0, self.frames - t0, len(self.O_atoms)
        )

    def time_origin_average(self, idx, max_tau: int = None) -> np.ndarray:
        if max_tau is None:
            max_tau = self.frames
        msd_accum = np.zeros(max_tau)
        count = np.zeros(max_tau, dtype=int)

        for t0 in tqdm(range(self.frames), desc="Computing MSD time-origin average"):
            msd_t0 = self.compute_msd_for_origin(t0, idx)
            valid_tau = min(len(msd_t0), max_tau)

            msd_accum[:valid_tau] += msd_t0[:valid_tau]
            count[:valid_tau] += 1

        valid_mask = count > 0
        msd_avg = np.zeros(max_tau)
        msd_avg[valid_mask] = msd_accum[valid_mask] / count[valid_mask]
        return msd_avg


if __name__ == "__main__":
    pathfiles = [
        "/home/debian/water/TIP4P/Ice/test/traj_1e-6_225_100000.lammpstrj",
        "/home/debian/water/TIP4P/Ice/test/traj_5e-6_225_100000.lammpstrj",
    ]
    output_h5 = "/home/debian/water/TIP4P/Ice/test/msd_results.h5"
    # output_h5 = "test_msd_results.h5"
    store = pd.HDFStore(output_h5)

    start_index = 0  # 跳过前800帧以避免初始非平衡影响
    # start_index = 0

    for pathfile in pathfiles:
        time_step = 0.025  # ps
        u = mda.Universe(pathfile, format="LAMMPSDUMP")
        shear_rate = float(pathfile.split("traj_")[-1].split("_225")[0]) * 1e3
        # shear_rate = 0
        print(f"shear_rate extracted: {shear_rate} 1/ps")
        msd_calculator = MSDCalculator(
            u, shear_rate=shear_rate, time_step=time_step, start_index=start_index
        )  # shear_rate in 1/ps(1e-7 1/fs)
        # msd_calculator = MSDCalculator(u, start_index=start_index)  # 无剪切流
        msd = msd_calculator.time_origin_average(idx=2)
        times = np.arange(len(msd)) * time_step

        df = pd.DataFrame({"time_ps": times, "MSD_A2": msd})
        filename = pathfile.split("traj_")[-1].split("_225")[0]
        filename = f"{filename}-z"
        # filename = "equili"
        store.put(filename, df, format="table")
        print(f"Saved MSD results to key: {filename}")
    store.close()
    print(f"All MSD results saved to: {output_h5}")
