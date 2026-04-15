import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import MDAnalysis as mda
from collections import defaultdict
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from multiprocessing import Pool, cpu_count
from scipy import stats
import os
import h5py
import gc
import pickle

file_path = "/home/debian/water/TIP4P/2005/nvt/dump_225_test.lammpstrj"
HB_H5_PATH = "/home/debian/water/TIP4P/2005/nvt/rst/equili/hbonds.h5"
COORDS_PATH = "shared_coords.npy"
# ----------------------------------------------------------------------
# 参数设置
# ----------------------------------------------------------------------
BATCH_SIZE = 10000
MIN_HBONDS = 4  # 定义笼状结构所需的最小氢键数
JUMP_THRESHOLD = 0.0  # 跳跃距离阈值，单位Å
N_WORKERS = min(8, cpu_count())  # 使用的并行进程数


u = mda.Universe(file_path, format="LAMMPSDUMP")
O_atoms = u.select_atoms("type 1")  # 选择氧原子
n_O_atoms = len(O_atoms)
start_frame = 0
n_frames = len(u.trajectory)
end_frame = n_frames + 1  # 使用所有帧


# ----------------------------------------------------------------------
# 流式化处理氢键数据以节省内存
# ----------------------------------------------------------------------
def get_atom_hbonds_streaming(atom_idx):
    atom_hb = defaultdict(set)

    with h5py.File(HB_H5_PATH, "r") as h5f:
        hb = h5f["hbonds"]
        for i in range(0, hb.shape[0], BATCH_SIZE):
            batch = hb[i : i + BATCH_SIZE]
            for r in batch:
                frame = r["frame"]
                if frame >= n_frames:
                    continue

                d, a = r["donor"], r["acceptor"]
                if d == atom_idx:
                    atom_hb[frame].add(a)
                elif a == atom_idx:
                    atom_hb[frame].add(d)

    return atom_hb


# ---------------------------------------------------------
# 单粒子cage jump分析函数
# ---------------------------------------------------------


def apply_shear_correction(coords, shear_rate, time_step):
    for frame in tqdm(range(coords.shape[0]), desc="Applying shear correction"):
        if frame == 0:
            continue
        y_positions = coords[frame - 1, :, 1]
        # 修正x坐标以消除剪切流影响
        coords[frame:, :, 0] -= shear_rate * time_step * y_positions
    return coords


# 此时我们定义非平凡位移的标准是根据氧原子从四面体结构中跳出来进行计算。从氧原子形成4个氢键的结构开始，4个氢键最终会完全断裂，并和新的原子形成4个氢键，此时认为发生了非平凡位移。
def compute_non_trival_time_for_one_atom(atom_idx, coords, min_hbonds=4):
    """
    计算单个氧原子的非平凡时间（笼状结构寿命和跳跃时间）
    根据图示修正：笼状结构结束是指与原来四个氢键伙伴的所有氢键完全断裂

    参数:
    atom_idx: 氧原子索引
    coords: 所有氧原子的坐标 (n_frames, n_atoms, 3)
    min_hbonds: 定义笼状结构所需的最小氢键数，默认为4

    返回:
    cage_lifetimes: 笼状结构寿命列表
    jump_times: 跳跃时间列表
    cage_info: 详细的笼状结构信息
    """

    n_frames = coords.shape[0]
    cage_lifetimes = []
    jump_times = []
    cage_info = []

    # 获取该原子的氢键信息
    atom_hbonds = get_atom_hbonds_streaming(atom_idx)

    # 遍历每一帧，检测笼状结构的开始和结束
    in_cage = False
    cage_start = None
    cage_partners = set()  # 当前笼状结构的氢键伙伴
    prev_cage_partners = set()  # 上一个笼状结构的氢键伙伴

    for frame in range(n_frames):
        # 获取当前帧的氢键伙伴
        current_partners = atom_hbonds.get(frame, set())
        num_hbonds = len(current_partners)

        # print(f"原子 {atom_idx}: 帧 {frame}, 当前氢键伙伴: {current_partners}, 数量: {num_hbonds}")
        # 检测笼状结构开始
        if not in_cage:
            if num_hbonds == min_hbonds:
                common_with_prev = current_partners.intersection(prev_cage_partners)
                # 如果与上一个笼状结构的伙伴没有交集，说明是新的
                if len(common_with_prev) <= 0:
                    in_cage = True
                    cage_start = frame
                    cage_partners = current_partners.copy()  # 记录初始氢键伙伴

                    # print(
                    #     f"原子 {atom_idx}: 帧 {frame} 开始笼状结构, 初始氢键伙伴: {cage_partners}"
                    # )
        else:
            remaining_with_initial = current_partners.intersection(cage_partners)
            if len(remaining_with_initial) <= 0:
                # 与初始伙伴的所有氢键都已断裂，笼状结构结束
                in_cage = False
                cage_end = frame
                cage_lifetime = cage_end - cage_start

                if cage_start > 0 and cage_end < n_frames - 1:
                    # 记录笼状结构信息
                    cage_lifetimes.append(cage_lifetime)
                    cage_info.append(
                        {
                            "atom_idx": atom_idx,
                            "start_frame": cage_start,
                            "end_frame": cage_end,
                            "lifetime": cage_lifetime,
                            "initial_partners": cage_partners.copy(),  # 初始氢键伙伴
                            "final_partners": current_partners.copy(),  # 结束时的氢键伙伴（可能与新伙伴形成）
                        }
                    )

                    # print(f"原子 {atom_idx}: 帧 {frame} 笼状结构结束, 寿命: {cage_lifetime} 帧")
                    # print(f"  初始伙伴: {cage_partners}")
                    # print(f"  当前伙伴: {current_partners} (可能包含新伙伴)")
                prev_cage_partners = cage_partners.copy()
                # 重置
                cage_start = None
                cage_partners = set()

    return cage_lifetimes, jump_times, cage_info


def compute_jump_times(cage_info, coords, hbond_dict, jump_threshold=0, min_new_hbonds=4):
    """
    计算跳跃时间（基于笼状结构之间的位置变化和氢键伙伴更换）

    参数:
    cage_info: 笼状结构信息列表
    coords: 原子坐标 (n_frames, n_atoms, 3)
    hbond_dict: 氢键字典
    jump_threshold: 跳跃距离阈值 (Å)
    min_new_hbonds: 新笼状结构所需的最小新氢键数

    返回:
    jump_times: 跳跃时间列表
    jump_info: 详细的跳跃信息
    """

    jump_times = []
    jump_info = []

    # 对每个连续的笼状结构对计算跳跃
    for i in range(len(cage_info) - 1):
        current_cage = cage_info[i]
        next_cage = cage_info[i + 1]

        # 跳跃开始时间：当前笼状结构结束
        jump_start = current_cage["end_frame"]
        # 跳跃结束时间：下一个笼状结构开始
        jump_end = next_cage["start_frame"]

        jump_duration = jump_end - jump_start

        if jump_duration > 0:  # 确保有跳跃过程
            # 检查氢键伙伴是否完全更换（根据图示要求）
            initial_partners_current = current_cage["initial_partners"]
            initial_partners_next = next_cage["initial_partners"]

            # 计算伙伴更换比例
            common_partners = initial_partners_current.intersection(initial_partners_next)
            partner_change_ratio = 1 - len(common_partners) / len(initial_partners_next)

            # 只有当伙伴完全更换（或大部分更换）且跳跃距离超过阈值时才认为是有效跳跃
            if partner_change_ratio >= 1:  # 至少75%的伙伴更换
                cage1_coord = coords[current_cage["end_frame"] - 1, current_cage["atom_idx"], :]
                cage2_coord = coords[next_cage["start_frame"], next_cage["atom_idx"], :]

                jump_distance = np.linalg.norm(cage2_coord - cage1_coord)

                if jump_distance >= jump_threshold:
                    jump_times.append(jump_duration)
                    jump_info.append(
                        {
                            "jump_start": jump_start,
                            "jump_end": jump_end,
                            "duration": jump_duration,
                            "distance": jump_distance,
                            "partner_change_ratio": partner_change_ratio,
                            "from_cage": current_cage,
                            "to_cage": next_cage,
                        }
                    )

                    print(f"跳跃检测: 帧 {jump_start}-{jump_end}, 持续时间: {jump_duration} 帧")
                    print(
                        f"  距离: {jump_distance:.2f} Å, 伙伴更换比例: {partner_change_ratio:.2f}"
                    )

    return jump_times, jump_info


def init_worker(shared_coords):
    global coords
    coords = shared_coords


def worker(atom_idx):
    return compute_non_trival_time_for_one_atom(atom_idx, coords, MIN_HBONDS)


def run_parallel_analysis():
    with Pool(
        N_WORKERS, initializer=init_worker, initargs=(np.load(COORDS_PATH, mmap_mode="r"),)
    ) as pool:
        results = list(
            tqdm(
                pool.imap(worker, range(n_O_atoms)),
                total=n_O_atoms,
                desc="Parallel analysis of oxygen atoms",
            )
        )
        cage_times = []
        jump_times = []
        cage_infos = []
        jump_infos = []
        for cage_lifetimes, jump_times_atom, cage_info in results:
            cage_times.extend(cage_lifetimes)
            jump_times.extend(jump_times_atom)
            cage_infos.extend(cage_info)
        return cage_times, jump_times, cage_infos, jump_infos


def save_results(cage_times, jump_times):
    np.save("cage_lifetimes.npy", np.array(cage_times))
    np.save("jump_times.npy", np.array(jump_times))
    pickle.dump((cage_times, jump_times), open("cage_jump.pkl", "wb"))
    print("Saved results → cage_lifetimes.npy, jump_times.npy, cage_jump.pkl")


def plot_results(cage_times, jump_times):
    plt.figure()
    plt.hist(cage_times, bins=50, density=True)
    plt.yscale("log")
    plt.xlabel("Cage lifetime (frames)")
    plt.title("Distribution of cage lifetimes")
    plt.savefig("cage_lifetime_distribution.png")

    plt.figure()
    plt.hist(jump_times, bins=50, density=True)
    plt.yscale("log")
    plt.xlabel("Jump length (nm)")
    plt.title("Distribution of jump lengths")
    plt.savefig("jump_length_distribution.png")

    print("Saved plots.")


if __name__ == "__main__":
    print("Starting parallel analysis...")
    cage_times, jump_times, cage_infos, jump_infos = run_parallel_analysis()

    print("Done.")
    print(f"Total cage lifetimes: {len(cage_times)}")
    print(f"Total jump times: {len(jump_times)}")

    save_results(cage_times, jump_times)
    plot_results(cage_times, jump_times)
