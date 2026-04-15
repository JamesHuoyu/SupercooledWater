import numpy as np
from numba import njit, prange
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


import numpy as np
from numba import njit, prange
from tqdm import tqdm

# =========================================================
# Heaviside overlap w_n(a,t)
# =========================================================


@njit(fastmath=True)
def compute_overlap_mask(pos0, post, a2):
    N = pos0.shape[0]
    mask = np.zeros(N, dtype=np.int32)
    for i in range(N):
        dx = post[i, 0] - pos0[i, 0]
        dy = post[i, 1] - pos0[i, 1]
        dz = post[i, 2] - pos0[i, 2]
        if dx * dx + dy * dy + dz * dz < a2:
            mask[i] = 1
    return mask


# =========================================================
# 修改：计算 W(q,t) 的实部和虚部（用于方差形式）
# =========================================================
@njit(fastmath=True)
def compute_W_real_imag(qvecs, pos0, post, a2):
    """
    计算每个 q 向量的 W(q,t) 的实部和虚部。
    W(q,t) = sum_n w_n(t) exp[-i q · r_n(0)]，其中 w_n(t) 是重叠函数。
    """
    Nq = qvecs.shape[0]
    N = pos0.shape[0]
    Fr = np.zeros(Nq)  # W 的实部
    Fi = np.zeros(Nq)  # W 的虚部

    for i in range(N):
        dx = post[i, 0] - pos0[i, 0]
        dy = post[i, 1] - pos0[i, 1]
        dz = post[i, 2] - pos0[i, 2]
        if dx * dx + dy * dy + dz * dz < a2:
            x0, y0, z0 = pos0[i]
            for q in range(Nq):
                qx, qy, qz = qvecs[q]
                # 相位：-q · r_n(0)，以匹配定义 W(q,t) = sum_n w_n exp[-i q · r_n(0)]
                phase = -(qx * x0 + qy * y0 + qz * z0)
                Fr[q] += np.cos(phase)
                Fi[q] += np.sin(phase)
    return Fr, Fi


# =========================================================
# 单窗口 S4(q≠0)
# =========================================================


# @njit(fastmath=True)
# def S4_window_multi_q(qvecs, pos0, post, a2):
#     Nq = qvecs.shape[0]
#     N = pos0.shape[0]

#     Fr = np.zeros(Nq)
#     Fi = np.zeros(Nq)

#     for i in range(N):
#         dx = post[i, 0] - pos0[i, 0]
#         dy = post[i, 1] - pos0[i, 1]
#         dz = post[i, 2] - pos0[i, 2]

#         if dx * dx + dy * dy + dz * dz < a2:
#             x0, y0, z0 = pos0[i]
#             for q in range(Nq):
#                 qx, qy, qz = qvecs[q]
#                 phase = qx * x0 + qy * y0 + qz * z0
#                 Fr[q] += np.cos(phase)
#                 Fi[q] += np.sin(phase)

#     S4 = np.empty(Nq)
#     for q in range(Nq):
#         S4[q] = (Fr[q] * Fr[q] + Fi[q] * Fi[q]) / N

#     return S4


# =========================================================
# q ≠ 0：所有时间窗口平均
# =========================================================


@njit(parallel=True, fastmath=True)
def S4_all_windows(qvecs, traj, t, a2):
    N_frames = traj.shape[0]
    n_windows = N_frames - t
    Nq = qvecs.shape[0]
    N = traj.shape[1]

    S4_accum = np.zeros(Nq)
    sum_Wr = np.zeros(Nq)
    sum_Wi = np.zeros(Nq)
    sum_W2 = np.zeros(Nq)

    for start in prange(n_windows):
        pos0 = traj[start]
        post = traj[start + t]
        Wr, Wi = compute_W_real_imag(qvecs, pos0, post, a2)
        for q in range(Nq):
            sum_Wr[q] += Wr[q]
            sum_Wi[q] += Wi[q]
            sum_W2[q] += Wr[q] ** 2 + Wi[q] ** 2
        #  S4_accum += S4_window_multi_q(qvecs, pos0, post, a2)
    avg_Wr = sum_Wr / n_windows
    avg_Wi = sum_Wi / n_windows
    avg_W2 = sum_W2 / n_windows

    for q in range(Nq):
        S4_accum[q] = (avg_W2[q] - (avg_Wr[q] ** 2 + avg_Wi[q] ** 2)) / N

    return S4_accum


# =========================================================
# q = 0 特殊：精确 χ₄(t)
# =========================================================


@njit(parallel=True)
def chi4_from_windows(traj, t, a2):
    """
    计算方差形式的 χ₄(t) for q=0。
    χ₄(t) = (1/N) [ ⟨N_s²⟩ - ⟨N_s⟩² ]，其中 N_s = sum_n w_n(t) 是慢粒子数。
    """
    N_frames = traj.shape[0]
    n_windows = N_frames - t
    N = traj.shape[1]

    total_Ns = 0.0
    total_Ns2 = 0.0
    for start in prange(n_windows):
        pos0 = traj[start]
        post = traj[start + t]
        N_s = 0  # 当前窗口的慢粒子数
        for i in range(N):
            dx = post[i, 0] - pos0[i, 0]
            dy = post[i, 1] - pos0[i, 1]
            dz = post[i, 2] - pos0[i, 2]
            if dx * dx + dy * dy + dz * dz < a2:
                N_s += 1
        total_Ns += N_s
        total_Ns2 += N_s * N_s

    avg_Ns = total_Ns / n_windows
    avg_Ns2 = total_Ns2 / n_windows
    chi4 = (avg_Ns2 - avg_Ns**2) / N  # 方差形式

    return chi4


# =========================================================
# q-shell 生成（允许 q=0）
# =========================================================


def q_shell_vectors(Lx, Ly, Lz, q_target, dq):
    if abs(q_target) < 1e-12:
        return np.array([[0.0, 0.0, 0.0]])

    qx0, qy0, qz0 = 2 * np.pi / Lx, 2 * np.pi / Ly, 2 * np.pi / Lz
    nmax = int((q_target + dq) / min(qx0, qy0, qz0)) + 2

    n = np.arange(-nmax, nmax + 1)
    nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")

    qx = nx * qx0
    qy = ny * qy0
    qz = nz * qz0
    qmag = np.sqrt(qx * qx + qy * qy + qz * qz)

    mask = np.abs(qmag - q_target) <= dq
    qvecs = np.column_stack((qx[mask], qy[mask], qz[mask]))

    return np.unique(np.round(qvecs, 12), axis=0)


# =========================================================
# 主接口：S4(q,t)
# =========================================================


def compute_S4_vs_q(q_values, dq, traj, t, Lx, Ly, Lz, a=1.0):
    a2 = a * a
    S4_results = []

    for q in q_values:
        if abs(q) < 1e-12:
            chi4 = chi4_from_windows(traj, t, a2)
            S4_results.append(chi4)
            continue

        qs = q_shell_vectors(Lx, Ly, Lz, q, dq)
        print(f"q={q:.3f}, q-vectors={len(qs)}")
        S4_q = S4_all_windows(qs, traj, t, a2)
        S4_results.append(np.mean(S4_q))

    return np.array(S4_results)


def compute_S4_vs_t(q_value, dq, traj, t_values, Lx, Ly, Lz, a=1.0):
    """
    计算 S4(q,t) 随时间 t 的变化（方差形式）。

    参数:
        q_value: float, 固定的波矢模长 |q|。如果为0，则计算 χ₄(t)（q=0 极限）。
        dq: float, q-shell 的宽度容差。
        traj: numpy array, 轨迹数据，形状为 (N_frames, N_particles, 3)。
        t_values: list 或 array, 时间延迟 t 的序列。
        Lx, Ly, Lz: float, 系统尺寸。
        a: float, 重叠阈值（默认1.0）。

    返回:
        S4_t: numpy array, 每个 t 对应的 S4(q,t) 值。
    """
    a2 = a * a
    S4_t = np.zeros(len(t_values))

    # 预计算 q-shell 向量（如果 q ≠ 0）
    if abs(q_value) < 1e-12:
        # q=0 情况：直接使用 χ₄(t) 计算
        for i, t in enumerate(tqdm(t_values, desc="Computing χ₄(t) vs t")):
            S4_t[i] = chi4_from_windows(traj, t, a2)
    else:
        # q ≠ 0 情况：生成 q-shell 向量并计算 S4(q,t)
        qs = q_shell_vectors(Lx, Ly, Lz, q_value, dq)
        print(f"使用 q={q_value:.3f} 的 q-shell，包含 {len(qs)} 个向量")
        for i, t in enumerate(tqdm(t_values, desc="Computing S4(q,t) vs t")):
            S4_q = S4_all_windows(qs, traj, t, a2)  # 返回每个 q 向量的 S4 值
            S4_t[i] = np.mean(S4_q)  # 对 q-shell 取平均

    return S4_t


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # ===============================
    # Test 6: 单个球形 slow 团簇
    # ===============================

    np.random.seed(0)

    N = 4000
    L = 20.0
    R = 4.0
    center = np.array([L / 2, L / 2, L / 2])

    traj = np.zeros((2, N, 3))
    traj[0] = np.random.rand(N, 3) * L
    traj[1] = traj[0]

    for i in range(N):
        if np.linalg.norm(traj[0, i] - center) > R:
            traj[1, i] += 5.0  # fast

    # q 包含 0
    q_values = np.linspace(0.0, 3.0, 25)

    results = compute_S4_vs_q(q_values, dq=0.15, traj=traj, t=1, Lx=L, Ly=L, Lz=L, a=1.0)

    plt.plot(q_values, results, "o-")
    plt.xlabel("q")
    plt.ylabel("S4(q)")
    plt.title("Test 6: Single Cluster")
    plt.grid()
    plt.show()

    print("q=0 χ4 =", results[0])
