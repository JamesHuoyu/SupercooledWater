import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import argparse
import MDAnalysis as mda
import pandas as pd
from tqdm import tqdm
from MDAnalysis import Universe
from numba import jit, prange


class ISFCalculator:
    def __init__(
        self,
        universe: Universe,
        shear_rate: float = 0.0,
        time_step: float = 1.0,
        start_index: int = 0,
    ):
        """
        初始化ISF计算器
        :param universe: MDAnalysis Universe对象
        """
        self.universe = universe
        self.n_frames = len(universe.trajectory)
        self.n_particles = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        self.frames = self.n_frames - start_index
        # 预加载轨迹数据到内存，只针对O原子
        self.coords = np.zeros((self.frames, len(self.O_atoms), 3))
        for ts in tqdm(self.universe.trajectory[start_index:], desc="Loading trajectory"):
            self.coords[ts.frame - start_index] = self.O_atoms.positions.copy()
        # 应用剪切流修正
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
    def compute_phases_numba(coords, t0, q_vectors, max_tau, n_particles, n_q):
        """使用Numba加速的相位计算"""
        n_tau = min(coords.shape[0] - t0, max_tau)
        n_q = q_vectors.shape[0]
        isf_t0 = np.zeros((n_q, n_tau))
        ref_pos = coords[t0]

        for tau in prange(n_tau):
            current_pos = coords[t0 + tau]
            dr = current_pos - ref_pos

            for q_idx in range(n_q):
                q_dot_r_sum = 0.0
                for i in range(n_particles):
                    q_dot_r = (
                        dr[i, 0] * q_vectors[q_idx, 0]
                        + dr[i, 1] * q_vectors[q_idx, 1]
                        + dr[i, 2] * q_vectors[q_idx, 2]
                    )
                    q_dot_r_sum += np.cos(q_dot_r)  # 直接计算实部，避免复数运算

                isf_t0[q_idx, tau] = q_dot_r_sum / n_particles

        return isf_t0

    def compute_isf_for_origin(self, t0: int, q_vectors: np.ndarray) -> np.ndarray:
        return self.compute_phases_numba(
            self.coords, t0, q_vectors, self.frames - t0, len(self.O_atoms), q_vectors.shape[0]
        )

    def time_origin_average(self, q_vectors: np.ndarray, max_tau: int = None) -> np.ndarray:
        """
        执行时间原点平均计算ISF
        :param q_vectors: q矢量数组 (n_q, 3)
        :param max_tau: 最大时间延迟（可选）
        :return: 平均ISF (n_q, max_tau)
        """
        max_tau = max_tau or self.frames
        n_q = q_vectors.shape[0]
        isf_accum = np.zeros((n_q, max_tau))
        count = np.zeros(max_tau, dtype=int)

        for t0 in tqdm(range(self.frames), desc="Computing ISF"):
            isf_t0 = self.compute_isf_for_origin(t0, q_vectors)
            valid_tau = min(isf_t0.shape[1], max_tau)

            isf_accum[:, :valid_tau] += isf_t0[:, :valid_tau]
            count[:valid_tau] += 1

        # 处理可能存在的除零情况
        valid_mask = count > 0
        isf_avg = np.zeros((n_q, max_tau))
        isf_avg[:, valid_mask] = isf_accum[:, valid_mask] / count[valid_mask]

        return isf_avg


# 示例使用方式
if __name__ == "__main__":
    # 创建Universe对象（需提供拓扑和轨迹文件）
    pathfiles = [
    "/root/water/TIP4P/Ice/225/dump_225_test.lammpstrj"
    #     "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-4_246.lammpstrj",
    #     "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_7.5e-5_246.lammpstrj",
    #     "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-5_246.lammpstrj",
    #     "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_1e-5_246.lammpstrj",
    ]
    # pathfiles = ["/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-4_246_everystep.lammpstrj"]
    # pathfiles = [
    #     "/home/debian/water/TIP4P/Ice/test/5e-6/traj_5e-6_225_100000.lammpstrj",
    # ]
    # output_h5 = "test_isf_results.h5"
    output_h5 = "/root/water/TIP4P/Ice/test/h5/isf_results.h5"
    store = pd.HDFStore(output_h5)
    for pathfile in pathfiles:
        u = Universe(pathfile, format="LAMMPSDUMP")
        start_index = 0  # 跳过前2000帧以避免初始非平衡影响
        # start_index = 0  # 从头开始计算
        # 初始化计算器
        isf_calculator = ISFCalculator(u, start_index=start_index)

        # 定义q矢量（沿z方向的q矢量,垂直于xy平面，不受剪切流影响）
        q_magnitude = 2.25  # Å⁻¹
        q_vectors = np.array([[0, 0, q_magnitude]])

        # 计算时间原点平均的ISF
        isf = isf_calculator.time_origin_average(q_vectors)
        # time_steps = 0.05  # ps
        time_steps = 0.2  # ps
        times = np.arange(isf.shape[1]) * time_steps

        # 结果处理（保存）
        df = pd.DataFrame({"time_ps": times, "ISF": isf[0]})
        filename = pathfile.split("/")[-1].replace(".lammpstrj", "_isf.csv")
        # key = filename.split("_")[1]  # 对应的是剪切率部分例如"2.5e-5"
        key = "equili"  # 固定key以便对比
        store.put(key, df)
        print(f"ISF results saved to {output_h5} under key '{key}'")
    store.close()
    print("All ISF calculations completed.")
