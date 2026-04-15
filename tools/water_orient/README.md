# water_orient

一个用于 **TIP4P/Ice 过冷水中取向变化—氢键拓扑—\(\zeta\) 序参量耦合分析** 的轻量 Python 模块包。

## 设计目标

这个包专门对应下面这条物理问题链：

1. 水分子的取向变化会不会单独导致氢键网络重连？
2. 这种重连会不会在**没有显著平移重排**的情况下改变 \(\zeta\)？
3. 如果会，\(\zeta\) 的快速翻转中有多少是：
   - 取向/氢键拓扑主导
   - 平移主导
   - 两者混合
   - 仅仅是阈值/热涨落噪声

## 依赖

- Python >= 3.10
- numpy
- scipy（推荐；用于更快的邻居搜索）

## 坐标输入约定

所有核心函数都直接接收 `numpy.ndarray`，便于在 ipynb 中调用。

对单帧：

- `O.shape == (n_mol, 3)`
- `H1.shape == (n_mol, 3)`
- `H2.shape == (n_mol, 3)`
- `box.shape == (3,)`，只支持正交盒

对多帧：

- `O_series.shape == (n_frames, n_mol, 3)`
- `H1_series.shape == (n_frames, n_mol, 3)`
- `H2_series.shape == (n_frames, n_mol, 3)`

## 快速开始

```python
import sys
sys.path.append('/mnt/data')

from water_orient import (
    body_frames, dipole_vectors, tetrahedral_q,
    detect_hbonds, compute_zeta, coarse_grain_zeta,
    identify_donor_arm_jumps, classify_zeta_change_cause,
)
```

### 单帧分析

```python
hb = detect_hbonds(O, H1, H2, box, r_oo_cut=3.5, angle_cut_deg=30.0)
zeta = compute_zeta(O, hb, box)
zeta_cg = coarse_grain_zeta(zeta, O, box, neighbor_cut=3.5)
q = tetrahedral_q(O, box)
frames = body_frames(O, H1, H2, box)
```

### 多帧分析：构建 donor-arm acceptor 序列

```python
hbond_series = [detect_hbonds(O_series[t], H1_series[t], H2_series[t], box) for t in range(n_frames)]
arm_series = np.stack([hb.arm_acceptors for hb in hbond_series], axis=0)
jumps = identify_donor_arm_jumps(arm_series, min_dwell=3)
```

## 模块说明

### `geometry.py`
- `minimum_image`
- `wrap_positions`
- `pairwise_distances_pbc`
- `neighbor_list`

### `orientation.py`
- `dipole_vectors`：分子偶极/角平分线方向
- `body_frames`：完整分子体坐标系 `e1,e2,e3`
- `angular_displacement`：向量角位移
- `angular_displacement_from_frames`：完整三维刚体转角
- `rotational_correlation`：`P1`/`P2` 旋转关联

### `local_order.py`
- `tetrahedral_q`：四面体序参数
- `pair_orientation_metrics`：局域偶极对齐与偶极-键向夹角统计

### `hbonds.py`
- `detect_hbonds`：基于 `O_d--O_a` 距离和 `H_d-O_d...O_a` 线性度识别氢键
- `HBondResult.arm_acceptors`：每个 donor 分子的两个 OH 臂当前指向哪个 acceptor
- `arm_acceptor_switch_mask`：两帧之间 donor-arm 受体是否切换

### `zeta.py`
- `compute_zeta`：基于 HB 连通性和排序邻居计算 \(\zeta\)
- `compute_zeta_series`
- `coarse_grain_zeta`
- `hysteretic_states`：带滞回的 structured/disordered 判态

### `jumps.py`
- `identify_donor_arm_jumps`：识别 donor-arm jump 事件
- `jumps_to_frame_mask`

### `events.py`
- `lead_lag_average`：事件触发平均
- `cumulative_disorder_exposure`
- `conditional_event_probability`
- `classify_zeta_change_cause`

## 推荐整体研究流程

### 阶段 A：取向量的定义

同时保留三种取向相关量，而不是只看偶极转角：

1. **完整刚体取向变化**：`angular_displacement_from_frames`
2. **偶极变化**：`angular_displacement(dipole_t0, dipole_t1)`
3. **局域角序变化**：`tetrahedral_q`

原因是：
- 偶极转角捕捉的是单分子朝向变化；
- `q` 捕捉的是邻居构型的角向四面体有序；
- donor-arm jump 则直接对应氢键拓扑切换。

### 阶段 B：识别 \(\zeta\) 变化的三种来源

对每个时间步或事件窗口，分子级别同时计算：

- `delta_zeta`
- `rotation_deg`
- `arm_switch`
- `radial_shell_rms`

然后用 `classify_zeta_change_cause` 得到粗分类：

- `orientational`
- `translational`
- `mixed`
- `unresolved`
- `stable`

### 阶段 C：检验“非平移不可逆变化”

重点挑出下面这类分子：

- `|delta_zeta|` 大
- `radial_shell_rms` 小
- `rotation_deg` 大 或 `arm_switch == True`

这就是最直接的候选：

> 在没有显著壳层径向重排时，仅由分子取向变化/氢键切换导致 \(\zeta\) 发生不可逆变化。

接下来再加一个**持续时间条件**：
- 变化后新 \(\zeta\) 状态持续至少 `t_hold`
- donor-arm 新受体持续至少 `min_dwell`

这样就能过滤掉 libration/recrossing 噪声。

### 阶段 D：事件触发分析

建议做两组 lead-lag 图：

1. 以 `zeta-flip` 为中心：
   - `rotation_deg(t0 + tau)`
   - `q(t0 + tau)`
   - `HB jump rate(t0 + tau)`

2. 以 `HB jump` 为中心：
   - `zeta(t0 + tau)`
   - `zeta_cg(t0 + tau)`
   - `q(t0 + tau)`

如果先看到 OH 跳跃/HB 切换，再看到 \(\zeta\) 改变，那说明 \(\zeta\) 变化更像**取向主导的结果**；
如果先看到 `q` 或 `zeta_cg` 松动，再发生 jump，则更像结构前驱。

## 你最应该优先回答的三个问题

### 1. raw \(\zeta\) 翻转里有多少是“取向驱动而非平移驱动”？

统计：

```python
frac_orient = np.mean(result['category'] == 'orientational')
frac_mixed = np.mean(result['category'] == 'mixed')
```

### 2. 这些取向驱动的 \(\zeta\) 改变是不是不可逆的？

对候选事件要求：
- donor-arm switch 持续超过 `min_dwell`
- `zeta` 新状态持续超过 `t_hold`

### 3. 是否还存在别的“非平移”来源？

有，至少要检查下面几项：

- **氢键判据敏感性**：距离/角阈值改变导致的表观切换
- **双受体竞争/bifurcated H-bond**：局域拓扑不唯一
- **局域密度/配位数波动**：并不一定表现为大平移
- **剪切诱导的仿射几何畸变**：需要先减去仿射流场
- **热 libration**：大角抖动但不伴随持续拓扑变化

## 建议的图表列表

1. `P(zeta, q)` 联合分布
2. `P(delta_zeta, rotation_deg)` 联合分布
3. `P(HB_jump | zeta_state)`
4. `lead_lag: <zeta(t0+tau)> around HB jump`
5. `lead_lag: <rotation_deg(t0+tau)> around zeta flip`
6. `category` 饼图 / 柱状图：stable/orientational/translational/mixed/unresolved
7. 分子切片图：只高亮 `orientational` 事件分子

## 当前版本的边界

1. 只支持正交盒；
2. 没有直接读 LAMMPS dump，需要你先把 O/H 坐标整理成 NumPy 数组；
3. `classify_zeta_change_cause` 是**物理启发式分类**，不是严格因果证明；
4. 如果你的体系在稳态剪切下具有显著仿射位移，建议先在外部完成去仿射处理，再把去仿射后的坐标送入这个包。
