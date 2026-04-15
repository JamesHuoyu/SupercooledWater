import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy import interpolate


def calculate_tau_q(time_ps, Q, threshold=1 / np.e):
    """计算tau_q，即Q衰减到1/e的时间"""
    interp_func = interpolate.interp1d(Q, time_ps, kind="linear", fill_value="extrapolate")
    try:
        tau_q = interp_func(threshold)
        return float(tau_q)
    except ValueError:
        # 如果插值失败，返回NaN
        return np.nan


def plot_Q(filepath: str, key: str, start_time: bool = True):
    df = pd.read_hdf(filepath, key=key)
    if start_time:
        x = df["time_ps"][1:]
        y = df["Q"][1:]
    else:
        x = df["time_ps"]
        y = df["Q"]
    plt.plot(x, y, "-", markersize=4, label=key)
    tau_Q_ps = calculate_tau_q(x, y)
    print(f"{key}: tau_Q = {tau_Q_ps:.5f} ps")
    plt.plot([tau_Q_ps], [1 / np.e], "ro")


# 示例使用
plt.figure(figsize=(7, 5))
plot_Q("/home/debian/water/TIP4P/Ice/test/Q_results.h5", "1e-6")
plot_Q("/home/debian/water/TIP4P/Ice/test/Q_results.h5", "5e-6")
# plot_Q("/home/debian/water/TIP4P/Ice/test/Q_results.h5", "1e-5")
plot_Q("/home/debian/water/TIP4P/Ice/test/Q_results.h5", "5e-5")
plot_Q("/home/debian/water/TIP4P/Ice/test/Q_results.h5", "1e-4")
plot_Q("/home/debian/water/TIP4P/Ice/test/Q_results.h5", "equili")
plt.xlabel("Time (ps)")
plt.ylabel(r"$Q$")
plt.title(r"$Q$ vs Time")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("/home/debian/water/TIP4P/2005/2020/rst/new_Q_results.png", dpi=300)
plt.show()
