# 读取hbonds.h5文件，获取平均氢键长度的信息：
#     做氢键长度分布图
#     做氢键长度随时间变化图
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from tqdm import tqdm
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.distances import apply_PBC

"""
绘制氢键长度分布图和随时间变化图，在单剪切率下的结果
"""


class HBondAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.hbonds_df = pd.read_hdf(file_path, key="hbonds")
        self.output_prefix = file_path.replace(".h5", "")

    def plot_hb_lengths(self):
        hbond_lengths = self.hbonds_df.groupby("frame")["distance"].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(hbond_lengths.index, hbond_lengths.values, marker="o", linestyle="-")
        plt.title("Average Hydrogen Bond Length Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Average Hydrogen Bond Length (Å)")
        plt.grid()
        plt.savefig(f"{self.output_prefix}_avg_hbond_length.png", dpi=300)
        plt.show()

    def plot_hb_length_distribution(self, specific_frames):
        plt.figure(figsize=(15, 10))
        for i, frame in enumerate(specific_frames, 1):
            frame_data = self.hbonds_df[self.hbonds_df["frame"] == frame]
            plt.subplot(2, 3, i)
            plt.hist(frame_data["distance"], bins=100, color="skyblue", edgecolor="black")
            plt.title(f"Frame {frame}")
            plt.xlabel("Hydrogen Bond Length (Å)")
            plt.ylabel("Frequency")
            plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_overall_distribution(self, threshold: int = None):
        steady_state_frames = (
            self.hbonds_df[self.hbonds_df["frame"] >= threshold]
            if threshold is not None
            else self.hbonds_df
        )
        plt.figure(figsize=(8, 6))
        plt.hist(steady_state_frames["distance"], bins=100, color="salmon", edgecolor="black")
        plt.title("Overall Hydrogen Bond Length Distribution")
        plt.xlabel("Hydrogen Bond Length (Å)")
        plt.ylabel("Frequency")
        plt.grid()
        # plt.show()
        return steady_state_frames["distance"].mean()


# 写一个继承HBondAnalyzer的类，专门用来分析氢键网络的性质，比如平均连接数，聚集体大小分布等
class NetworkAnalyzer(HBondAnalyzer):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def _resolve_columns(self) -> tuple[str, str, str | None]:
        """自动识别 hbonds_df 中 donor/acceptor/hydrogen 列名。

        你的数据描述里是：frame, donor, hydrogen, acceptor, distance, angle。
        代码里也常见 donor_id/acceptor_id/hydrogen_id。
        """

        cols = set(self.hbonds_df.columns)

        donor_col = "donor_id" if "donor_id" in cols else ("donor" if "donor" in cols else None)
        acceptor_col = (
            "acceptor_id" if "acceptor_id" in cols else ("acceptor" if "acceptor" in cols else None)
        )
        hydrogen_col = (
            "hydrogen_id" if "hydrogen_id" in cols else ("hydrogen" if "hydrogen" in cols else None)
        )

        if donor_col is None or acceptor_col is None:
            raise KeyError(
                "hbonds_df 必须包含 donor/donor_id 和 acceptor/acceptor_id 列；"
                f"当前列: {sorted(cols)}"
            )
        return donor_col, acceptor_col, hydrogen_col

    def _frame_group(self, frame: int) -> pd.DataFrame:
        group = self.hbonds_df[self.hbonds_df["frame"] == frame]
        if group.empty:
            raise ValueError(f"frame={frame} 不存在或无氢键记录")
        return group

    def analyze_hbond_network(
        self,
        max_depth: int = 12,
        compute_cycles: bool = False,
    ) -> pd.DataFrame:
        """分析每个 frame 的氢键网络性质。

        说明（对你当前代码的关键修正点）：
        - 你原来的 DFS + 全局 visited 统计的是“有向可达子树大小”，而不是聚集体/连通分量。
          在有向图里它会因为遍历顺序产生偏差，并且会漏掉只有入边、没有出边的节点。
        - 如果你的“cluster/聚集体”想表达网络连通性，通常更合理的是把氢键边当作无向边，
          统计无向连通分量大小分布（并查集/Union-Find 更快、更稳）。

        参数:
            max_depth: 环路遍历/检测的最大深度（只在 compute_cycles=True 时使用）。
            compute_cycles: 是否按“沿有向边遍历，回到起点则成环，最大深度 max_depth”统计环路。

        返回列（兼容原始输出）:
            - frame
            - avg_cluster_size: 无向连通分量大小均值
            - num_clusters: 无向连通分量个数
        同时会额外返回一些常用网络指标:
            - num_nodes, num_edges, avg_out_degree
            - max_cluster_size
            - num_cycles, avg_cycle_length（若 compute_cycles=True）
        """

        donor_col, acceptor_col, _ = self._resolve_columns()
        hbonds_grouped = self.hbonds_df.groupby("frame", sort=True)
        network_properties: list[dict] = []

        for frame, group in hbonds_grouped:
            donors = group[donor_col].to_numpy()
            acceptors = group[acceptor_col].to_numpy()

            # 节点集合必须包含 donor 和 acceptor，避免漏掉只有入边的节点
            nodes = pd.unique(np.concatenate([donors, acceptors]))
            num_nodes = int(nodes.size)
            num_edges = int(len(group))

            # ---- 1) 无向连通分量（聚集体）统计：并查集更高效、更符合“cluster”语义 ----
            # 将节点 ID 映射到 0..N-1
            node_index = {int(node_id): idx for idx, node_id in enumerate(nodes)}

            parent = np.arange(num_nodes, dtype=np.int64)
            size = np.ones(num_nodes, dtype=np.int64)

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra = find(a)
                rb = find(b)
                if ra == rb:
                    return
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]

            for d, a in zip(donors, acceptors):
                union(node_index[int(d)], node_index[int(a)])

            # 统计每个根的分量大小
            roots = np.array([find(i) for i in range(num_nodes)], dtype=np.int64)
            # roots 可能不是连续的，用 pandas 计数最直观
            comp_sizes = pd.Series(roots).value_counts().to_numpy(dtype=np.int64)
            cluster_sizes = comp_sizes.tolist()

            avg_cluster_size = float(np.mean(comp_sizes)) if comp_sizes.size else 0.0
            max_cluster_size = int(np.max(comp_sizes)) if comp_sizes.size else 0
            num_clusters = int(comp_sizes.size)

            # ---- 2) 出度（有向）统计：反映“平均连接数” ----
            # avg_out_degree：按所有节点平均（没有出边的节点也算 0）
            out_counts = pd.Series(donors).value_counts()
            total_out = float(out_counts.sum())
            avg_out_degree = total_out / num_nodes if num_nodes else 0.0

            # ---- 3) 可选：按深度限制统计有向环路（简单环） ----
            num_cycles = None
            avg_cycle_length = None
            if compute_cycles:
                adjacency: dict[int, list[int]] = {}
                for d, a in zip(donors, acceptors):
                    di = int(d)
                    ai = int(a)
                    adjacency.setdefault(di, []).append(ai)

                # 仅检测长度<=max_depth 的简单环；用规范化去重（同一环不同起点/方向）
                cycles: set[tuple[int, ...]] = set()

                def canonicalize_cycle(cycle: list[int]) -> tuple[int, ...]:
                    # cycle: [v0, v1, ..., vk, v0]，去掉末尾重复 v0
                    core = cycle[:-1]
                    # 旋转，使最小节点在开头，减少重复
                    min_idx = min(range(len(core)), key=lambda i: core[i])
                    rotated = core[min_idx:] + core[:min_idx]
                    # 同时考虑反向
                    rev = list(reversed(core))
                    min_idx_r = min(range(len(rev)), key=lambda i: rev[i])
                    rotated_r = rev[min_idx_r:] + rev[:min_idx_r]
                    return (
                        tuple(rotated) if tuple(rotated) <= tuple(rotated_r) else tuple(rotated_r)
                    )

                def dfs_cycles(start: int) -> None:
                    stack: list[tuple[int, list[int], set[int]]] = [(start, [start], {start})]
                    while stack:
                        node, path, in_path = stack.pop()
                        if len(path) > max_depth:
                            continue
                        for nxt in adjacency.get(node, []):
                            if nxt == start and len(path) >= 2:
                                cyc = path + [start]
                                cycles.add(canonicalize_cycle(cyc))
                                continue
                            if nxt in in_path:
                                continue
                            stack.append((nxt, path + [nxt], in_path | {nxt}))

                for node in nodes:
                    dfs_cycles(int(node))

                num_cycles = int(len(cycles))
                avg_cycle_length = float(np.mean([len(c) for c in cycles])) if cycles else 0.0

            row = {
                "frame": frame,
                "avg_cluster_size": avg_cluster_size,
                "num_clusters": num_clusters,
                "max_cluster_size": max_cluster_size,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "avg_out_degree": avg_out_degree,
            }
            if compute_cycles:
                row.update({"num_cycles": num_cycles, "avg_cycle_length": avg_cycle_length})
            network_properties.append(row)

        return pd.DataFrame(network_properties)

    def check_frame_consistency(self, frame: int) -> dict:
        """对单帧网络统计做快速一致性检查（用于判断构网是否“算对了”）。

        返回包含若干布尔项与关键计数，便于你在 notebook 里直接 print/断言。
        """

        donor_col, acceptor_col, _ = self._resolve_columns()
        group = self._frame_group(frame)

        donors = group[donor_col].to_numpy()
        acceptors = group[acceptor_col].to_numpy()
        nodes = pd.unique(np.concatenate([donors, acceptors]))

        num_nodes = int(nodes.size)
        num_edges = int(len(group))
        # 增加调试：
        if num_nodes != 4096:
            # 给出是哪些氧原子序号没有在node里面
            print(num_nodes)
            idxs = nodes // 3
            for i in range(4096):
                if i in set(idxs):
                    continue
                print(i)
            # print(f"缺失的氧原子序号为：")
        out_degree_sum = int(pd.Series(donors).value_counts().sum())

        # 复用 analyze_hbond_network 的并查集逻辑（在单帧上跑一遍）
        node_index = {int(node_id): idx for idx, node_id in enumerate(nodes)}
        parent = np.arange(num_nodes, dtype=np.int64)
        size = np.ones(num_nodes, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        for d, a in zip(donors, acceptors):
            union(node_index[int(d)], node_index[int(a)])

        roots = np.array([find(i) for i in range(num_nodes)], dtype=np.int64)
        comp_sizes = pd.Series(roots).value_counts().to_numpy(dtype=np.int64)

        sum_comp_sizes = int(comp_sizes.sum())
        num_clusters = int(comp_sizes.size)

        checks = {
            "frame": int(frame),
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_clusters": num_clusters,
            "sum_component_sizes": sum_comp_sizes,
            "out_degree_sum": out_degree_sum,
            # 守恒关系：分量大小之和必须等于节点数
            "ok_components_sum_to_nodes": sum_comp_sizes == num_nodes,
            # 出度求和必须等于边数（每条边贡献一个 donor 出度）
            "ok_out_degree_sum_to_edges": out_degree_sum == num_edges,
            # 平均出度应等于 E/N（按所有节点平均）
            "avg_out_degree": (num_edges / num_nodes) if num_nodes else 0.0,
        }
        return checks

    def plot_frame_network_diagnostics(
        self,
        frame: int,
        max_nodes_for_graph: int = 0,
        seed: int = 0,
        bins: int = 50,
    ) -> None:
        """对单帧做简单可视化：度分布 + 聚集体大小分布 + （可选）网络示意图。

        max_nodes_for_graph:
            - 0：不画网络图（推荐默认，最快）
            - >0：如果安装了 networkx，则抽取最多该数量节点画 spring layout 示意
        """

        donor_col, acceptor_col, _ = self._resolve_columns()
        group = self._frame_group(frame)
        donors = group[donor_col].to_numpy()
        acceptors = group[acceptor_col].to_numpy()

        nodes = pd.unique(np.concatenate([donors, acceptors]))
        num_nodes = int(nodes.size)
        num_edges = int(len(group))

        # 度：按有向出度统计（donor -> acceptor）
        out_counts = pd.Series(donors).value_counts()
        out_degrees = np.array([int(out_counts.get(node, 0)) for node in nodes], dtype=np.int64)

        # 聚集体：无向连通分量大小（并查集）
        node_index = {int(node_id): idx for idx, node_id in enumerate(nodes)}
        parent = np.arange(num_nodes, dtype=np.int64)
        size = np.ones(num_nodes, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        for d, a in zip(donors, acceptors):
            union(node_index[int(d)], node_index[int(a)])

        roots = np.array([find(i) for i in range(num_nodes)], dtype=np.int64)
        comp_sizes = pd.Series(roots).value_counts().to_numpy(dtype=np.int64)

        fig = plt.figure(figsize=(14, 4))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.hist(out_degrees, bins=bins, color="skyblue", edgecolor="black")
        ax1.set_title(f"Frame {frame}: Out-degree distribution")
        ax1.set_xlabel("Out-degree")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.hist(
            comp_sizes,
            bins=min(bins, max(5, int(comp_sizes.size))),
            color="salmon",
            edgecolor="black",
        )
        ax2.set_title(f"Frame {frame}: Component size distribution")
        ax2.set_xlabel("Component size")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.axis("off")
        ax3.set_title("Graph view (optional)")
        ax3.text(
            0.01,
            0.95,
            f"nodes={num_nodes}\nedges={num_edges}\n"
            f"avg_out={num_edges/num_nodes if num_nodes else 0:.3f}\n"
            f"components={int(comp_sizes.size)}\nmax_comp={int(comp_sizes.max()) if comp_sizes.size else 0}",
            va="top",
        )

        if max_nodes_for_graph and num_nodes:
            try:
                import networkx as nx

                rng = np.random.default_rng(seed)
                keep_n = min(max_nodes_for_graph, num_nodes)
                keep_nodes = set(rng.choice(nodes.astype(int), size=keep_n, replace=False).tolist())
                # 为了连通性示意，把 keep_nodes 对应的边拿出来
                edge_list = [
                    (int(d), int(a))
                    for d, a in zip(donors, acceptors)
                    if int(d) in keep_nodes and int(a) in keep_nodes
                ]
                G = nx.DiGraph()
                G.add_nodes_from(list(keep_nodes))
                G.add_edges_from(edge_list)
                pos = nx.spring_layout(G, seed=seed)
                ax3.clear()
                ax3.axis("off")
                ax3.set_title(f"Graph view (n={keep_n})")
                nx.draw_networkx_nodes(G, pos, ax=ax3, node_size=20)
                nx.draw_networkx_edges(G, pos, ax=ax3, arrows=False, width=0.5, alpha=0.5)
            except ImportError:
                ax3.text(
                    0.01, 0.65, "networkx 未安装，跳过网络图。\n可只看前两幅分布图。", va="top"
                )

        plt.tight_layout()
        plt.show()

    def ring_size_distribution(
        self,
        frame: int,
        min_ring: int = 3,
        max_ring: int = 10,
        sample_edges: int | None = None,
        seed: int = 0,
        normalize: bool = True,
    ) -> pd.Series:
        """估计无向氢键网络的环尺寸分布（更接近常见“3–10 环”文献口径）。

        重要说明：
        - 你之前的 compute_cycles=True 找的是“有向 donor→acceptor 的简单有向环”，
          这和文献里常用的“无向 O-O 环/原始环”并不是一回事，因此很容易出现“几乎没环”或环长异常。
        - 这里采用一种可扩展的近似统计：对每条无向边 (u,v)，删除该边后在限定深度内
          寻找 u 到 v 的最短路径长度 L，则得到一个最短环长度 k=L+1。
          该统计会对同一个环按边重复计数；若 normalize=True，会用 count/k 做粗略去重。

        参数:
            frame: 帧号
            min_ring/max_ring: 只统计该范围内的环长
            sample_edges: 若指定，则随机抽样该数量的无向边以加速（4096 节点建议先抽样）
            normalize: True 时对每个环长 k 返回 count/k（近似按“每个环被 k 条边重复计数”校正）

        返回:
            index 为环长 (min_ring..max_ring)，value 为估计的环数量（或归一化后的估计值）。
        """

        if min_ring < 3:
            raise ValueError("min_ring 必须 >= 3")
        if max_ring < min_ring:
            raise ValueError("max_ring 必须 >= min_ring")

        donor_col, acceptor_col, _ = self._resolve_columns()
        group = self._frame_group(frame)
        donors = group[donor_col].to_numpy()
        acceptors = group[acceptor_col].to_numpy()

        # 构建无向邻接（用 set 去重）
        adjacency: dict[int, set[int]] = {}
        undirected_edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()

        for d, a in zip(donors, acceptors):
            u = int(d)
            v = int(a)
            if u == v:
                continue
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)
            e = (u, v) if u < v else (v, u)
            if e not in seen_edges:
                seen_edges.add(e)
                undirected_edges.append(e)

        if not undirected_edges:
            return pd.Series(
                {k: 0.0 for k in range(min_ring, max_ring + 1)},
                name=f"frame_{frame}",
                dtype=float,
            )

        edges_to_check = undirected_edges
        if sample_edges is not None and sample_edges < len(undirected_edges):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(undirected_edges), size=int(sample_edges), replace=False)
            edges_to_check = [undirected_edges[i] for i in idx]

        max_depth = max_ring - 1  # 最短路径长度上限
        counts = {k: 0 for k in range(min_ring, max_ring + 1)}

        # 限深 BFS：寻找在移除边 (u,v) 后 u->v 的最短路径长度
        from collections import deque

        def shortest_path_len_without_edge(u: int, v: int) -> int | None:
            # BFS from u, stop when reaching v or exceeding max_depth
            q = deque([(u, 0)])
            visited = {u}
            while q:
                node, depth = q.popleft()
                if depth >= max_depth:
                    continue
                for nxt in adjacency.get(node, ()):  # type: ignore[arg-type]
                    # 跳过被“删除”的边
                    if (node == u and nxt == v) or (node == v and nxt == u):
                        continue
                    if nxt in visited:
                        continue
                    if nxt == v:
                        return depth + 1
                    visited.add(nxt)
                    q.append((nxt, depth + 1))
            return None

        for u, v in edges_to_check:
            L = shortest_path_len_without_edge(u, v)
            if L is None:
                continue
            ring_len = L + 1
            if min_ring <= ring_len <= max_ring:
                counts[ring_len] += 1

        values = {}
        for k in range(min_ring, max_ring + 1):
            values[k] = (counts[k] / k) if normalize else float(counts[k])

        return pd.Series(values, name=f"frame_{frame}", dtype=float)

    def plot_ring_size_distribution(
        self,
        frame: int,
        min_ring: int = 3,
        max_ring: int = 10,
        sample_edges: int | None = 20000,
        seed: int = 0,
        normalize: bool = True,
    ) -> pd.Series:
        """计算并绘制单帧 3–10 环（无向）分布。"""

        dist = self.ring_size_distribution(
            frame=frame,
            min_ring=min_ring,
            max_ring=max_ring,
            sample_edges=sample_edges,
            seed=seed,
            normalize=normalize,
        )
        plt.figure(figsize=(7, 4))
        plt.bar(dist.index.astype(int), dist.values, color="mediumpurple", edgecolor="black")
        plt.title(f"Frame {frame}: Ring size distribution ({min_ring}-{max_ring})")
        plt.xlabel("Ring size")
        plt.ylabel("Estimated count" + (" (normalized)" if normalize else ""))
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
        return dist

    def particle_ring_table(
        self,
        frame: int,
        min_ring: int = 3,
        max_ring: int = 10,
        max_depth: int | None = None,
    ) -> pd.Series:
        """
        按文献定义的 particle-ring（以分子为中心）统计环尺寸分布。

        定义：
        - 对每个分子 i
        - 对其一阶邻居对 (j,k)
        - 在删除 i 的无向网络中，找 j->k 的最短路径 L
        - particle-ring 长度 = L + 2

        返回：
        - index: ring length
        - value: ring count（所有分子累积）
        """

        donor_col, acceptor_col, _ = self._resolve_columns()
        group = self._frame_group(frame)

        donors = group[donor_col].to_numpy(dtype=int)
        acceptors = group[acceptor_col].to_numpy(dtype=int)

        # ---------- 构建无向邻接 ----------
        adjacency: dict[int, set[int]] = {}
        for d, a in zip(donors, acceptors):
            if d == a:
                continue
            adjacency.setdefault(d, set()).add(a)
            adjacency.setdefault(a, set()).add(d)

        if max_depth is None:
            max_depth = max_ring - 2

        from collections import deque
        from itertools import combinations

        # counts = {k: 0 for k in range(min_ring, max_ring + 1)}
        records: list[dict] = []

        # ---------- 中心分子 ----------
        for i, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue

            neighbors = list(neighbors)

            for j, k in combinations(neighbors, 2):

                # BFS，记录路径
                q = deque([(j, [j])])
                visited = {j, i}

                found_path = None

                while q:
                    node, path = q.popleft()
                    if len(path) - 1 >= max_depth:
                        continue
                    for nxt in adjacency.get(node, ()):
                        if nxt in visited:
                            continue
                        if nxt == k:
                            found_path = path + [k]
                            q.clear()
                            break
                        visited.add(nxt)
                        q.append((nxt, path + [nxt]))

                if found_path is None:
                    continue

                ring_nodes = (i,) + tuple(found_path) + (i,)
                ring_len = len(ring_nodes) - 1

                if min_ring <= ring_len <= max_ring:
                    records.append(
                        {
                            "frame": frame,
                            "center": i,
                            "ring_len": ring_len,
                            "ring_nodes": ring_nodes,
                            "pair": (j, k),
                        }
                    )

        return pd.DataFrame.from_records(records)

    def particle_ring_distribution_from_table(df: pd.DataFrame) -> pd.Series:
        return df["ring_len"].value_counts().sort_index()


class SingleNetworkAnalyzer:
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
        self.boxes = np.zeros((self.frames, 6))
        for ts in tqdm(self.universe.trajectory[start_index:], desc="Loading trajectory data"):
            self.coords[ts.frame - start_index] = self.O_atoms.positions.copy()
            self.boxes[ts.frame - start_index] = ts.dimensions.copy()

    def ring_shape_and_axis(
        self, positions: np.ndarray, box: np.ndarray
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        返回:
            elongation
            asphericity
            eigenvalues (λ1 ≤ λ2 ≤ λ3)
            main_axis (对应 λ3 的单位向量)
        """
        # ---------- PBC 最小映射 ----------
        ref = positions[0]
        rel_vectors = minimize_vectors(positions - ref, box)
        unwrapped = ref + rel_vectors

        # ---------- 质心 ----------
        r0 = unwrapped.mean(axis=0)
        r = unwrapped - r0

        # ---------- 惯性张量 ----------
        I = np.zeros((3, 3))
        for ri in r:
            r2 = np.dot(ri, ri)
            I += np.eye(3) * r2 - np.outer(ri, ri)

        # ---------- 特征分解 ----------
        eigvals, eigvecs = np.linalg.eigh(I)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        l1, l2, l3 = eigvals
        main_axis = eigvecs[:, 2]
        main_axis /= np.linalg.norm(main_axis)

        # ---------- elongation ----------
        elongation = (l2 - l1) / l3 if l3 > 0 else 0.0

        # ---------- asphericity ----------
        normalized = eigvals / l3 if l3 > 0 else eigvals
        asphericity = np.sqrt(((normalized - normalized.mean()) ** 2).sum() / 3.0)

        return elongation, asphericity, eigvals, main_axis

    def add_ring_shape_columns(
        self, df: pd.DataFrame, coords: np.ndarray, box: np.ndarray
    ) -> pd.DataFrame:

        elongations = []
        asphericities = []
        cos2_x_list = []
        cos2_plus_list = []

        ex = np.array([1.0, 0.0, 0.0])
        e_plus = np.array([1.0, 1.0, 0.0])
        e_plus /= np.linalg.norm(e_plus)

        for ring_nodes in df["ring_nodes"]:
            nodes = ring_nodes[:-1]
            particle_indices = [n // 3 for n in nodes]
            positions = coords[particle_indices]

            e, b, _, axis = self.ring_shape_and_axis(positions, box)

            elongations.append(e)
            asphericities.append(b)

            cos2_x_list.append((np.dot(axis, ex)) ** 2)
            cos2_plus_list.append((np.dot(axis, e_plus)) ** 2)

        df = df.copy()
        df["elongation"] = elongations
        df["asphericity"] = asphericities
        df["cos2_flow"] = cos2_x_list
        df["cos2_extensional"] = cos2_plus_list

        return df

    def ring_center_calculate(
        self, positions: np.ndarray, box: np.ndarray
    ) -> tuple[float, float, float]:
        """
        positions: (N, 3) array, ring 粒子的坐标（已去重）
        考虑周期性边界条件带来的影响
        返回:
            每个环质心的位置（放缩回盒子中展示）
        """
        ref = positions[0]
        rel_vecs = minimize_vectors(positions - ref, box)
        unwrapped = rel_vecs + ref
        r0 = unwrapped.mean(axis=0)
        # apply_PBC
        r0_inbox = apply_PBC(r0, box)
        return r0_inbox

    def find_ring_centers(
        self, df: pd.DataFrame, coords: np.ndarray, box: np.ndarray
    ) -> pd.DataFrame:
        ring_centers = []
        for ring_nodes in df["ring_nodes"]:
            nodes = ring_nodes[:-1]
            particle_indices = [n // 3 for n in nodes]
            positions = coords[particle_indices]
            ring_center = self.ring_center_calculate(positions, box)
            ring_centers.append(ring_center)

        df = df.copy()
        df["ring_center"] = ring_centers
        return df


# if __name__ == "__main__":
#     file_paths = [
#         "/home/debian/water/TIP4P/Ice/225/shear/rst/1e-6/hbonds.h5",
#         "/home/debian/water/TIP4P/Ice/225/shear/rst/5e-6/hbonds.h5",
#         "/home/debian/water/TIP4P/Ice/225/shear/rst/5e-5/hbonds.h5",
#         "/home/debian/water/TIP4P/Ice/225/shear/rst/1e-4/hbonds.h5",
#         "/home/debian/water/TIP4P/Ice/225/shear/rst/5e-4/hbonds.h5",
#     ]
#     lengths_dict = {}
#     for file_path in file_paths:
#         analyzer = HBondAnalyzer(file_path)
#         # 给出初始状态下的平均氢键长度
#         print("Initial Average H-Bond Length:")
#         initial_avg_length = analyzer.hbonds_df[analyzer.hbonds_df["frame"] == 0]["distance"].mean()
#         print(f"{initial_avg_length:.4f} Å")
#         # 给出稳态下的平均氢键长度
#         # analyzer.plot_hb_lengths()
#         avg_length = analyzer.plot_overall_distribution(threshold=1500)
#         shear_rate = float(file_path.split("/")[-2])
#         print(f"Shear Rate: {shear_rate}, Average H-Bond Length: {avg_length:.4f} Å")
#         lengths_dict[shear_rate] = avg_length
#     # 绘制不同剪切率下的平均氢键长度对比图
#     shear_rates = list(lengths_dict.keys())
#     avg_lengths = list(lengths_dict.values())
#     delta_avg_lengths = [l - 2.814 for l in avg_lengths]  # 初始氢键长度约为2.814Å
#     plt.figure(figsize=(8, 6))
#     # plt.plot(shear_rates, avg_lengths, marker="o", linestyle="-")
#     plt.plot(shear_rates, delta_avg_lengths, marker="o", linestyle="-")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.title("Average Hydrogen Bond Length vs Shear Rate")
#     plt.xlabel("Shear Rate")
#     plt.ylabel("Average Hydrogen Bond Length (Å)")
#     plt.grid()
#     plt.show()
if __name__ == "__main__":
    file_path = "/home/debian/water/TIP4P/Ice/225/shear/rst/1e-4/hbonds.h5"
    analyzer = NetworkAnalyzer(file_path)
    # print(analyzer.check_frame_consistency(2000))
    particle_ring_df = analyzer.particle_ring_table(frame=0, min_ring=3, max_ring=10)

    # ring_distribution = NetworkAnalyzer.particle_ring_distribution_from_table(particle_ring_df)
    # plt.figure(figsize=(7, 4))
    # plt.bar(
    #     ring_distribution.index.astype(int),
    #     ring_distribution.values,
    #     color="mediumpurple",
    #     edgecolor="black",
    # )
    # plt.title(f"Frame 0: Particle Ring Size Distribution (3-10)")
    # plt.xlabel("Ring size")
    # plt.ylabel("Count")
    # plt.grid(True, axis="y", alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    # analyzer.plot_frame_network_diagnostics(frame=0, max_nodes_for_graph=4, seed=42)
    # 画单帧3-10环分布（无向）
    # analyzer.plot_ring_size_distribution(
    #     frame=0, min_ring=3, max_ring=15, sample_edges=None, seed=0
    # )
    # analyzer.plot_ring_size_distribution(
    #     frame=2000, min_ring=3, max_ring=15, sample_edges=None, seed=0
    # )
    # network_df = analyzer.analyze_hbond_network(compute_cycles=True)
    # print(network_df.head())

    # # 检查某一帧的一致性
    # frame_to_check = 2000
    # consistency_report = analyzer.check_frame_consistency(frame_to_check)
    # print(f"Consistency report for frame {frame_to_check}:")
    # for key, value in consistency_report.items():
    #     print(f"  {key}: {value}")

    # # 绘制某一帧的网络诊断图
    # analyzer.plot_frame_network_diagnostics(frame=frame_to_check, max_nodes_for_graph=100)
