# 取向研究整体协议

## 研究主目标

在稳态剪切条件下，判定以下命题是否成立：

> 部分 \(\zeta\) 的改变并不是由明显的平移壳层重排直接导致，而是由水分子的取向变化、OH donor-arm 跳跃和氢键网络拓扑重连驱动。

## 核心变量

### A. 单分子取向变化

- 偶极/角平分线方向：`dipole_vectors`
- 完整体坐标旋转角：`body_frames` + `angular_displacement_from_frames`
- 两根 OH 的方向变化：可以直接对 `H1-O`、`H2-O` 调 `angular_displacement`

### B. 局域角向结构

- 四面体序参数：`tetrahedral_q`
- 邻居偶极排列：`pair_orientation_metrics`

### C. 氢键网络拓扑

- donor-arm acceptor：`HBondResult.arm_acceptors`
- donor-arm jump：`identify_donor_arm_jumps`
- 邻接矩阵：`HBondResult.adjacency`

### D. \(\zeta\) 与粗粒化 \(\zeta_{cg}\)

- 原始：`compute_zeta`
- 粗粒化：`coarse_grain_zeta`
- 滞回判态：`hysteretic_states`

## 推荐分析顺序

### Step 1. 先验证取向与 \(\zeta\) 是否相关

单帧/双时刻做：

- `corr(zeta, q)`
- `corr(zeta, mean_mu_mu)`
- `corr(zeta(t), delta_theta_body(t, t+Δt))`
- `P(HB_jump | zeta low/high)`

### Step 2. 区分“瞬时取向摆动”和“持续取向重排”

必须加两个条件：

1. donor-arm 新受体持续 `min_dwell`
2. \(\zeta\) 新状态持续 `t_hold`

这样可以排除 libration 和阈值 recrossing。

### Step 3. 定位“非平移 \(\zeta\) 改变”

用 `classify_zeta_change_cause` 先筛出：

- `orientational`
- `mixed`

然后再逐个事件检查：

- 是否伴随 donor-arm jump
- 是否伴随大体坐标旋转
- 是否只伴随很小的壳层径向变化

### Step 4. 事件触发分析

#### 4a. 以 donor-arm jump 为中心

看：
- `zeta(t0+tau)`
- `zeta_cg(t0+tau)`
- `q(t0+tau)`

#### 4b. 以 zeta flip 为中心

看：
- `rotation_deg(t0+tau)`
- `HB_jump_rate(t0+tau)`
- `q(t0+tau)`

## 如何检查“其他非平移来源”

除了取向变化外，还要逐项排查：

### 1. 氢键判据敏感性

固定轨迹，改变：
- `r_oo_cut`
- `angle_cut_deg`

比较 \(\zeta\) flip 数量和 `orientational` 分类占比是否稳定。

### 2. bifurcated / competing hydrogen bonds

如果同一 OH 臂在相邻帧间对两个 acceptor 都接近阈值，\(\zeta\) 可能会因为“边界竞争”发生切换。

### 3. 局域密度变化

哪怕没有大位移，第一壳层/第二壳层配位数轻微变化也可能导致 \(\zeta\) 变化。
可在外部增加：
- 第一壳层配位数
- 第二壳层配位数
- 最近邻序统计

### 4. 剪切的仿射畸变

如果坐标没有先去仿射，可能会把流场导致的几何拉伸误判成结构变化。

### 5. 数值配准和 PBC 展开问题

错误的 unwrap / wrap 会伪造大转角或邻居交换。

## 最后要形成的物理论证

最终你希望把结论组织成三层：

1. **事实层**：部分 \(\zeta\) 改变发生时，没有显著壳层径向重排；
2. **机制层**：这些改变与 donor-arm jump / 完整体转角 / H-bond 重连同步；
3. **筛选层**：去掉热摆动、阈值敏感性和仿射畸变后，这类事件依然存在。

如果这三层都成立，就能比较有力地说：

> 在你的体系里，\(\zeta\) 不是纯粹的平移序参数，它还携带取向控制的氢键拓扑信息；而部分看似“非平移”的不可逆 \(\zeta\) 改变，正是由这种拓扑重连驱动的。
