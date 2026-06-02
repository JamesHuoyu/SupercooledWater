# zeta_cluster_toolkit


## v4 update: MDAnalysis-style triclinic PBC

This version delegates triclinic PBC operations to MDAnalysis:

- `minimum_image(...)` calls `MDAnalysis.lib.distances.minimize_vectors`;
- `wrap_positions_pbc(...)` calls `MDAnalysis.lib.distances.apply_PBC`;
- `query_pairs_pbc(...)` calls `MDAnalysis.lib.distances.self_capped_distance`, with `method="nsgrid"` by default;
- the cluster centroid is computed by unwrapping compact members around a reference with MDAnalysis MIC vectors, not by circular averaging in fractional coordinates.

Therefore, `zeta_cg` cluster connectivity and track centroid matching no longer use the simplified `frac -= round(frac)` implementation from v3.


这个小包是从 `stress_5e_6.ipynb` 中拆出来的团簇分析工具，目标是把 notebook 里的临时分析逻辑变成可复用模块。当前版本同时保留两条圈团路线：

1. **direct zeta_cg domains**：直接根据每个分子的 `zeta_cg` 圈出 high-ζ / low-ζ 结构团簇，不再做额外粗粒化或卷积。
2. **joint P3D-zeta domains**：保留原来的 `P_i^{3D}` 与 `zeta_cg` 联合圈团方法，内部仍会做局部平均与 score 构造。

---

## 模块结构

```text
zeta_cluster_toolkit/
├── __init__.py
├── geometry.py           # PBC、wrap、minimum image、PBC centroid、slab mask
├── io_helpers.py         # 从 ZOP 对象抽取 frame payload，并批量运行圈团
├── propensity.py         # iso-configurational ensemble 的 P_i^3D 计算
├── domain_extraction.py  # direct zeta_cg 圈团 + joint P3D-zeta 圈团
├── visualization.py      # zeta_cg 切片、domain 切片、Plotly/Matplotlib 3D 图
└── tracking.py           # 跨帧团簇连续性追踪，以及 track observable/stress hook
```

---

## 1. 单帧 zeta_cg 切片展示

```python
import numpy as np
from zeta_cluster_toolkit import frame_payload_from_zop, plot_zeta_cg_slice

box = np.array([50.0, 50.0, 50.0])
frame = 0

payload = frame_payload_from_zop(zop, frame=frame, box=box, field="zeta_cg")

fig, ax = plot_zeta_cg_slice(
    positions=payload["positions"],
    zeta_cg=payload["zeta"],
    box=box,
    axis=2,             # 2 表示 z-normal，即 XY slice
    slice_center=25.0,
    slice_width=6.0,
)
```

---

## 2. 推荐主线：直接根据 zeta_cg 圈结构团簇

如果 `zeta_cg` 已经由 ZOP 中的局部粗粒化得到，就不需要再做一次空间卷积。新的入口是：

```python
from zeta_cluster_toolkit import (
    extract_zeta_domains_for_frame,
    print_domain_summary,
    plot_zeta_domain_slice,
    plot_domains_3d_matplotlib,
)

result_z = extract_zeta_domains_for_frame(
    zop=zop,
    frame=0,
    box=box,
    r_connect=3.6,
    high_seed_pct=90,
    high_grow_pct=75,
    low_seed_pct=10,
    low_grow_pct=25,
    min_cluster_size=10,
    min_core_size=3,
)

print_domain_summary(result_z)

fig, ax = plot_zeta_domain_slice(
    result_z,
    axis=2,
    slice_center=25.0,
    slice_width=6.0,
)

fig3d, ax3d = plot_domains_3d_matplotlib(
    result_z,
    show_background=True,
    max_background=1500,
    show_hulls=False,
    background_color_by="zeta_cg",
)
```

### direct zeta_cg 圈团定义

high-ζ / slow-structured domain：

```text
core: zeta_cg >= percentile(zeta_cg, high_seed_pct)
halo: 与 core 连通，且 zeta_cg >= percentile(zeta_cg, high_grow_pct)
```

low-ζ / fast-disordered domain：

```text
core: zeta_cg <= percentile(zeta_cg, low_seed_pct)
halo: 与 core 连通，且 zeta_cg <= percentile(zeta_cg, low_grow_pct)
```

默认返回结构仍然和原来的联合方法兼容：

```python
result_z["clusters"]["slow"]  # high-zeta structural domains
result_z["clusters"]["fast"]  # low-zeta disordered domains
result_z["masks"]["slow_core"]
result_z["masks"]["slow_halo"]
result_z["masks"]["fast_core"]
result_z["masks"]["fast_halo"]
```

这样后续的可视化和 tracking 函数都可以继续使用。

---

## 3. 保留原来的 P3D-zeta 联合圈团方法

如果你希望圈出“低运动倾向 + 高 ζ_cg”的慢结构域，或者“高运动倾向 + 低 ζ_cg”的快结构域，可以继续使用原来的联合方法：

```python
from zeta_cluster_toolkit import extract_domains_for_frame

result_joint = extract_domains_for_frame(
    zop=zop,
    frame=0,
    P3D=P3D,
    box=box,
    local_length=3.5,
    local_cutoff=6.0,
    r_connect=3.6,
    alpha=1.0,
    mode="score",
    slow_seed_pct=90,
    slow_grow_pct=75,
    fast_seed_pct=90,
    fast_grow_pct=75,
    min_cluster_size=10,
    min_core_size=3,
)
```

联合方法内部使用：

```text
slow_score = Z(local zeta_cg) - alpha * Z(local P3D)
fast_score = Z(local P3D) - alpha * Z(local zeta_cg)
```

因此它会做 `P3D` 和 `zeta_cg` 的局部平均。这个方法适合做结构-动力学耦合分析，但不建议作为结构团簇生命周期追踪的唯一主线。

---

## 4. 多帧 direct zeta_cg 圈团与连续性追踪

```python
from zeta_cluster_toolkit import (
    extract_zeta_domains_over_frames,
    track_clusters_over_time,
    summarize_tracks,
)

frames = range(0, 800, 10)

results_z = extract_zeta_domains_over_frames(
    zop=zop,
    frames=frames,
    box=box,
    r_connect=3.6,
    high_seed_pct=90,
    high_grow_pct=75,
    low_seed_pct=10,
    low_grow_pct=25,
    min_cluster_size=10,
    min_core_size=3,
)

tracks = track_clusters_over_time(
    results_z,
    min_overlap=3,
    min_jaccard=0.15,
    min_containment=0.35,
    max_centroid_distance=8.0,
    max_gap_steps=1,
)

track_table = summarize_tracks(
    tracks,
    dt_per_sample=10 * 0.025,  # ps
)
track_table
```

---

## 5. 团簇应力跟踪接口

后续如果你已经有了逐帧、逐分子的应力数组，例如：

```python
pxy_mol.shape == (n_sampled_frames, n_molecules)
```

那么可以沿 track 提取团簇的应力贡献：

```python
from zeta_cluster_toolkit import track_observable

stress_track = track_observable(
    tracks,
    frame_values=pxy_mol,
    index_set="member_indices",  # 或 "core_indices"
    reducer="sum",
)

stress_amp_track = track_observable(
    tracks,
    frame_values=pxy_mol,
    index_set="member_indices",
    reducer="abs_sum",
)
```

建议后续至少保留三条曲线：

1. `sum(pxy_i)`：团簇对体系剪切应力的符号贡献；
2. `sum(abs(pxy_i))` 或 `rms(pxy_i)`：局部应力幅度；
3. `N_cluster(t)` 或 `Rg(t)`：团簇大小/几何尺度，用于判断应力变化是否只是粒子数变化导致。

---

## 推荐工作流

```python
# A. 先看结构场
payload = frame_payload_from_zop(zop, frame=0, box=box)
plot_zeta_cg_slice(payload["positions"], payload["zeta"], box, axis=2)

# B. 单帧 direct zeta_cg 圈团调参数
result0 = extract_zeta_domains_for_frame(
    zop, 0, box=box,
    high_seed_pct=90,
    high_grow_pct=75,
    low_seed_pct=10,
    low_grow_pct=25,
)
plot_zeta_domain_slice(result0, axis=2)
print_domain_summary(result0)

# C. 固定参数后跑多帧
results = extract_zeta_domains_over_frames(
    zop, frames=range(0, 800, 10), box=box,
    high_seed_pct=90,
    high_grow_pct=75,
)

# D. 追踪团簇生命周期
tracks = track_clusters_over_time(results, max_gap_steps=1)
summary = summarize_tracks(tracks, dt_per_sample=0.25)

# E. 与应力耦合
stress_track = track_observable(tracks, pxy_mol, reducer="sum")
```

---

## 物理解释建议

- 以结构生命周期为主线时，优先使用 **direct zeta_cg domains**。这样团簇定义只依赖结构变量本身，不会被某个动力学窗口中的 `P3D` 重新加权。
- `high-zeta` 团簇可解释为更 LDL-like、更局域有序、更高氢键连通性的结构域。
- `low-zeta` 团簇可作为 HDL-like / disordered 对照区域。
- `core_indices` 更适合追踪稳定结构主体；`halo_indices` 更适合研究边界交换、重排和应力释放。
- 如果目标是“结构域的应力积累与耗散”，建议分别计算 core、halo、whole-cluster 三套应力时间序列。

## v3 note: triclinic/sheared-box PBC

For shear simulations generated by LAMMPS `fix deform`, do not use only
`[Lx, Ly, Lz]` if the box has a nonzero tilt factor. The v3 geometry layer accepts
triclinic boxes and applies the minimum-image convention in fractional coordinates.

Supported `box` formats:

```python
# Orthorhombic
box = np.array([Lx, Ly, Lz])

# MDAnalysis-style triclinic dimensions, if your Universe exposes them
box = u.dimensions.copy()  # [lx, ly, lz, alpha, beta, gamma]

# LAMMPS restricted triclinic matrix
from zeta_cluster_toolkit import lammps_triclinic_matrix
box = lammps_triclinic_matrix(lx, ly, lz, xy=xy, xz=xz, yz=yz)
```

The direct `zeta_cg` extraction route still does not apply an additional spatial
convolution. It only uses the triclinic PBC geometry to define which grow/core
particles are connected.

To inspect how a tracked domain boundary changes over time:

```python
from zeta_cluster_toolkit import plot_track_boundary_evolution

fig, ax = plot_track_boundary_evolution(
    results_z,
    tracks[0],
    axis=2,                       # XY footprint
    coordinate_system="unsheared", # removes trivial triclinic skew
    use_core=False,
    draw_hull=True,
)
```

`coordinate_system="unsheared"` maps positions to fractional coordinates multiplied
by the cell-vector lengths. This is usually better for comparing intrinsic boundary
changes, because it removes the affine shear deformation of the simulation cell.
## v5 note: multi-frame extraction under shear

For a sheared/triclinic trajectory, do **not** pass a single fixed box to
`extract_zeta_domains_over_frames` unless the box is truly constant. The `box`
argument now accepts a callable, a dict keyed by frame, or a frame-aligned array.

Recommended MDAnalysis usage:

```python
frames = range(4000, 4800, 10)

def box_getter(frame):
    u.trajectory[int(frame)]
    return u.dimensions.copy()   # [lx, ly, lz, alpha, beta, gamma]

results_z = extract_zeta_domains_over_frames(
    zop=zop,
    frames=frames,
    box=box_getter,
    r_connect=3.6,
    high_seed_pct=90,
    high_grow_pct=75,
    low_seed_pct=10,
    low_grow_pct=25,
    min_cluster_size=10,
    min_core_size=3,
)
```

You can also pass `box_by_frame = {frame: box}` or an array aligned with `frames`,
e.g. shape `(len(frames), 6)`.
