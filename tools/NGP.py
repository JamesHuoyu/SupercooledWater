"""
计算 non-trival displacement 的工具
"""

import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
import pandas as pd
import h5py


# 计算对于单粒子的 non-trival displacement
def apply_shear_correction(coords, shear_rate, time_step, ref_y=25.0):
    T = coords.shape[0]
    y = coords[:, :, 1] - ref_y  # shape (T, N)
    gamma_dt = shear_rate * time_step
    shear_disp = gamma_dt * np.cumsum(y, axis=0)  # shape (T, N)
    coords[:, :, 0] -= shear_disp
    return coords


def compute_p_hop_fast(positions):
    """
    positions: (T, 3)
    return: p_hop array (T,)
    """
    R = positions
    T = R.shape[0]

    # cumulative sums
    cum_R = np.cumsum(R, axis=0)
    cum_R2 = np.cumsum(R**2, axis=0)

    p_hop = np.zeros(T)

    for t_c in range(1, T - 1):
        n1 = t_c
        n2 = T - t_c

        # centroids
        c1 = cum_R[t_c - 1] / n1
        c2 = (cum_R[-1] - cum_R[t_c - 1]) / n2

        # <|r|^2>
        mean_r2_S1 = cum_R2[t_c - 1] / n1
        mean_r2_S2 = (cum_R2[-1] - cum_R2[t_c - 1]) / n2

        # <|r - c|^2> using analytic formula
        d2_S1 = mean_r2_S1 - 2.0 * c2 * (cum_R[t_c - 1] / n1) + c2**2
        d1_S2 = mean_r2_S2 - 2.0 * c1 * ((cum_R[-1] - cum_R[t_c - 1]) / n2) + c1**2

        mean_d2 = np.sum(d2_S1)
        mean_d1 = np.sum(d1_S2)

        zeta = np.sqrt((t_c / T) * (1.0 - t_c / T))
        p_hop[t_c] = zeta * np.sqrt(mean_d2 * mean_d1)

    return p_hop


def compute_non_trival_iterable(positions, threshold):
    p_hop = compute_p_hop_fast(positions)
    maxmimum, max_index = np.max(p_hop), np.argmax(p_hop)
    if maxmimum > threshold:
        positions_a = positions[: max_index + 1]
        positions_b = positions[max_index:]
        indices_a = compute_non_trival_iterable(positions_a, threshold)
        indices_b = compute_non_trival_iterable(positions_b, threshold)
        indices_b_adjusted = [idx + max_index for idx in indices_b] if indices_b else []
        return indices_a + [max_index] + indices_b_adjusted
    else:
        return []


def save_O_dict_to_h5(h5_file, strain_rate, O_dict):
    with pd.HDFStore(h5_file) as store:
        data_list = []
        for o_idx, indices in O_dict.items():
            if indices:
                for event_idx in indices:
                    data_list.append({"O_index": o_idx, "event_frame": event_idx})
        if data_list:
            df = pd.DataFrame(data_list)
            key = f"{strain_rate}"
            store.put(key, df, format="table", data_columns=True)


# -----------------------------------------------------------------------------------------------
# original version
# def compute_p_tc(positions, t_c):
#     T = positions.shape[0]

#     S1 = positions[:t_c]
#     S2 = positions[t_c:]

#     centroid1 = np.mean(S1, axis=0)
#     centroid2 = np.mean(S2, axis=0)

#     d2_S1 = np.linalg.norm(S1 - centroid2, axis=1)
#     d1_S2 = np.linalg.norm(S2 - centroid1, axis=1)

#     mean_d2_sq = np.mean(d2_S1**2)  # <d2^2(t1)>
#     mean_d1_sq = np.mean(d1_S2**2)  # <d1^2(t2)>

#     zeta = np.sqrt(t_c / T * (1 - t_c / T))

#     p_tc = zeta * np.sqrt(mean_d2_sq * mean_d1_sq)
#     return p_tc


# def compute_non_trival_disp(positions):
#     """
#     计算单粒子的 non-trival displacement，不使用时间间隔参数,而是使用整个时间序列

#     参数:
#     positions: np.ndarray, 形状为 (T, 3)，表示粒子在 T 个时间步的三维位置

#     返回:
#     p_hop_list: np.ndarray, 形状为 (T,)，表示每个时间步的 non-trival displacement
#     """
#     p_hop_list = np.zeros(shape=(positions.shape[0],))
#     for t_c in range(1, positions.shape[0] - 1):
#         p_tc = compute_p_tc(positions, t_c)
#         p_hop_list[t_c] = p_tc
#     return p_hop_list


# def compute_non_trival_iterable(positions, threshold):
#     p_hop = compute_non_trival_disp(positions)
#     maxmimum, max_index = np.max(p_hop), np.argmax(p_hop)
#     if maxmimum > threshold:
#         positions_a = positions[: max_index + 1]
#         positions_b = positions[max_index:]
#         indices_a = compute_non_trival_iterable(positions_a, threshold)
#         indices_b = compute_non_trival_iterable(positions_b, threshold)
#         indices_b_adjusted = [idx + max_index for idx in indices_b] if indices_b else []
#         return indices_a + [max_index] + indices_b_adjusted
#     else:
#         return []


# def save_O_dict_to_h5(h5_file, strain_rate, O_dict):
#     with pd.HDFStore(h5_file) as store:
#         data_list = []
#         for o_idx, indices in O_dict.items():
#             if indices:
#                 for event_idx in indices:
#                     data_list.append({"O_index": o_idx, "event_frame": event_idx})
#         if data_list:
#             df = pd.DataFrame(data_list)
#             key = f"{strain_rate}"
#             store.put(key, df, format="table", data_columns=True)


for threshold in [0.25, 0.3, 0.35, 0.4]:
    # 示例计算和可视化
    file_paths = [
        "/home/debian/water/TIP4P/Ice/test/traj_5e-5_225_100000.lammpstrj",
        "/home/debian/water/TIP4P/Ice/test/traj_5e-6_225_100000.lammpstrj",
    ]
    # file_path = "/home/debian/water/TIP4P/2005/dump_H2O_246_10.lammpstrj"
    for file_path in file_paths:
        u = mda.Universe(file_path, format="LAMMPSDUMP")
        O_atoms = u.select_atoms("type 1")
        start_frame = 0  # 跳过前800帧以避免初始非平衡影响
        coords = np.zeros(shape=(len(u.trajectory[start_frame:]), O_atoms.n_atoms, 3))
        for ts in tqdm(u.trajectory[start_frame:]):
            coords[ts.frame - start_frame] = O_atoms.positions.copy()
        shear_rate = float(file_path.split("traj_")[-1].split("_225")[0]) * 1e3  # 1/ps
        # shear_rate = 0  # 1/ps
        time_step = 0.025  # ps
        # time_step = 0.02  # ps
        print(f"shear_rate extracted: {shear_rate} 1/ps")
        coords = apply_shear_correction(coords, shear_rate, time_step, ref_y=25.0)

        O_dict = {}
        for i in tqdm(
            range(coords.shape[1]), desc="Computing non-trival displacements for O atoms"
        ):
            positions = coords[:, i, :]
            O_idx = O_atoms.indices[i]
            computed_indices = compute_non_trival_iterable(positions, threshold=threshold)  # Å
            O_dict[O_idx] = computed_indices

        output_h5 = "/home/debian/water/TIP4P/Ice/test/test_non_trival_displacement_results.h5"
        key = file_path.split("traj_")[-1].split("_225")[0]
        key = f"{key}-{threshold}"
        # key = "equili"
        print(f"Saving non-trival displacement results to {output_h5} under key '{key}'")
        save_O_dict_to_h5(output_h5, key, O_dict)
# # 做成上下两部分的图，上半部分是位移，下半部分是 non-trival displacement
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# ax1.plot(displacement, label="Displacement", color="blue")
# ax1.set_ylabel("Displacement (Å)")
# ax1.legend()
# ax2.plot(non_trival_displacement, label="Non-trival Displacement", color="orange")
# ax2.axvline(x=time_interval, color="red", linestyle="--", label="Time Interval")
# ax2.axvline(
#     x=len(positions) - time_interval,
#     color="green",
#     linestyle="--",
#     label="Total Steps - Time Interval",
# )
# ax2.plot(maximum_index, maximum, "ro", label="Maximum Point")
# ax2.axhline(y=0.27, color="purple", linestyle=":", label="Threshold (0.27 Å)")
# ax2.set_ylabel("Non-trival Displacement (Å)")
# ax2.legend()
# ax2.set_xlabel("Time Step")
# plt.tight_layout()
# plt.show()
