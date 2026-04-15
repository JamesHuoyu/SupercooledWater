"""
测试窗口采样功能
"""
import numpy as np
import time
from new_chi_ultrafast import compute_S4_vs_q

# 模拟用户的数据规模（缩小版本用于快速测试）
T, N = 500, 1000  # 缩小规模用于测试
Lx, Ly, Lz = 50, 50, 50
q_values = np.array([0.5])
time_step = 0.2
t_chi = 59.6
t = int(t_chi / time_step)  # = 298

print("="*60)
print("窗口采样测试")
print("="*60)

# 生成测试轨迹
np.random.seed(42)
coords = np.random.rand(T, N, 3)
coords[:, :, 0] *= Lx
coords[:, :, 1] *= Ly
coords[:, :, 2] *= Lz

print(f"轨迹形状: {coords.shape}")
print(f"时间延迟 t: {t}")
print(f"总窗口数: {T - t}")

# 测试不同的采样比例
test_configs = [
    ("全部窗口", None),
    ("10% 窗口", int((T - t) * 0.1)),
    ("5% 窗口", int((T - t) * 0.05)),
    ("1% 窗口", int((T - t) * 0.01)),
]

print(f"\n测试不同窗口采样数量:")
for label, n_samples in test_configs:
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    start = time.time()
    results = compute_S4_vs_q(
        q_values, 0.1, coords, t, Lx, Ly, Lz,
        a=1.0, n_windows_samples=n_samples
    )
    elapsed = time.time() - start

    print(f"结果: {results[0]:.6f}")
    print(f"计算时间: {elapsed:.2f} 秒")

# 预估用户实际数据的计算时间
print(f"\n{'='*60}")
print("用户实际数据预估 (T=2501, N=4096)")
print(f"{'='*60}")

actual_windows = 2501 - 298  # 2203
for sample_ratio in [1.0, 0.1, 0.05, 0.01]:
    n_samples = int(actual_windows * sample_ratio)
    # 根据测试数据估算
    # 测试数据：N=1000，实际数据：N=4096 (4倍)
    # 双重求和复杂度 O(n_active^2)，假设慢粒子比例相同
    # 时间 ∝ N^2
    scale_factor = (4096 / 1000) ** 2  # ≈ 16.7
    # 另外考虑 q 向量数相同

    print(f"\n采样 {sample_ratio*100:.0f}% 窗口 (n={n_samples}):")
    print(f"  预估时间: ~{scale_factor * 2:.0f} × 基准时间")
    print(f"  (假设慢粒子比例相同，N 从 1000 增加到 4096)")
