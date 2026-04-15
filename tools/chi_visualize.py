import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def cal_max_chi(time_ps, chi4):
    """计算最大χ4值及其对应时间"""
    max_index = np.argmax(chi4)
    max_chi = chi4[max_index]
    tau_chi_ps = time_ps[max_index]
    print(f"Max χ4 = {max_chi:.5f} at τ_chi = {tau_chi_ps:.5f} ps")
    return max_chi, tau_chi_ps


def plot_chi(
    filepath: str,
    key: str,
    start_time: bool = True,
    smooth: bool = True,
    window: int = 11,
    poly: int = 3,
):
    df = pd.read_hdf(filepath, key=key)
    if start_time:
        x = df["time_ps"][1:]
        y = df["chi4"][1:]
    else:
        x = df["time_ps"]
        y = df["chi4"]
    if smooth:
        y_smooth = savgol_filter(y, window_length=window, polyorder=poly)
        plt.plot(x, y, "-", markersize=4, label=key)
        max_chi, tau_chi_ps = cal_max_chi(x, y_smooth)
    else:
        plt.plot(x, y, "-", markersize=4, label=key)
        max_chi, tau_chi_ps = cal_max_chi(x, y)
    plt.plot([tau_chi_ps], [max_chi], "ro")
    # 做以下平滑处理


# 示例使用
plt.figure(figsize=(7, 5))
# plot_chi("/home/debian/water/TIP4P/Ice/test/h5/chi4_results.h5", "1e-6")
plot_chi("/home/debian/water/TIP4P/Ice/test/h5/chi4_results.h5", "5e-6")
# plot_chi("/home/debian/water/TIP4P/Ice/test/test_chi4_results.h5", "1e-5")
# plot_chi("/home/debian/water/TIP4P/Ice/test/test_chi4_results.h5", "5e-5")
# plot_chi("/home/debian/water/TIP4P/Ice/test/test_chi4_results.h5", "1e-4")
plot_chi("/home/debian/water/TIP4P/Ice/test/h5/chi4_results.h5", "equili")
plt.xlabel("Time (ps)")
plt.ylabel(r"$\chi_4$")
plt.title(r"$\chi_4$ vs Time")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("/home/debian/water/TIP4P/2005/2020/rst/4096/chi4_shear_rates.png", dpi=300)
plt.show()
