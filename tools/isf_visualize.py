import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate


def calculate_tau_alpha(time_ps, isf, threshold=1 / np.e):
    """计算tau_alpha，即ISF衰减到1/e的时间"""

    interp_func = interpolate.interp1d(isf, time_ps, kind="linear", fill_value="extrapolate")
    try:
        tau_alpha = interp_func(threshold)
        return float(tau_alpha)
    except ValueError:
        # 如果插值失败，返回NaN
        return np.nan


def KWW_fit_func(t, A, tau, beta):
    return A * np.exp(-((t / tau) ** beta))


def plot_isf(filepath: str, key: str, start_time: bool = True):
    df = pd.read_hdf(filepath, key=key)
    if start_time:
        x = df["time_ps"][1:]
        y = df["ISF"][1:]
    else:
        x = df["time_ps"]
        y = df["ISF"]
    plt.plot(x, y, "-", markersize=4, label=key)
    tau_alpha = calculate_tau_alpha(x, y, threshold=1 / np.e)
    tau_alpha_ps_start = calculate_tau_alpha(x, y, threshold=0.4)
    tau_alpha_ps_end = calculate_tau_alpha(x, y, threshold=0.4 / np.e)
    tau_alpha_ps = tau_alpha_ps_end - tau_alpha_ps_start
    # print(f"{key}: tau_alpha = {tau_alpha_ps:.5f} ps")
    print(f"{key}: tau_alpha = {tau_alpha} ps")
    plt.plot([tau_alpha_ps_start], [0.4], "ro")
    plt.plot([tau_alpha_ps_end], [0.4 / np.e], "go")
    plt.plot([tau_alpha], [1 / np.e])
    # # KWW拟合
    # try:
    #     popt, pcov = curve_fit(
    #         KWW_fit_func,
    #         x,
    #         y,
    #         p0=(1.0, tau_alpha_ps, 0.8),
    #         bounds=([0, 0, 0], [1.0, np.inf, 2.0]),
    #     )
    #     A_fit, tau_fit, beta_fit = popt
    #     print(f"  KWW fit: A={A_fit:.4f}, tau={tau_fit:.4f} ps, beta={beta_fit:.4f}")
    #     x_fit = np.linspace(min(x), max(x), 200)
    #     y_fit = KWW_fit_func(x_fit, *popt)
    #     plt.plot(x_fit, y_fit, "--", label=f"KWW fit ({key})")
    # except Exception as e:
    #     print(f"  KWW fit failed for {key}: {e}")


# 示例使用
plt.figure(figsize=(7, 5))
plot_isf("/home/debian/water/TIP4P/Ice/test/h5/isf_results.h5", "equili")
plot_isf("/home/debian/water/TIP4P/Ice/test/h5/isf_results.h5", "5e-6")
# plot_isf("/home/debian/water/TIP4P/Ice/test/isf_results.h5", "5e-6")
# # plot_isf("/home/debian/water/TIP4P/Ice/test/isf_results.h5", "1e-5")
# plot_isf("/home/debian/water/TIP4P/Ice/test/isf_results.h5", "5e-5")
# plot_isf("/home/debian/water/TIP4P/Ice/test/isf_results.h5", "1e-4")
# plot_isf("/home/debian/water/TIP4P/Ice/test/isf_results.h5", "equili")
# plot_isf("/home/debian/water/TIP4P/2005/2020/rst/4096/new_isf_results.h5", "1e-5")
# plot_isf("/home/debian/water/TIP4P/2005/2020/rst/4096/new_isf_results.h5", "2.5e-5")
# plot_isf("/home/debian/water/TIP4P/2005/2020/rst/4096/new_isf_results.h5", "7.5e-5")
# plot_isf("/home/debian/water/TIP4P/2005/2020/rst/4096/new_isf_results.h5", "2.5e-4")
# plot_isf("test_isf_results.h5", "2.5e-4-everystep")
# plot_isf("test_isf_results.h5", "2.5e-5-everystep")
# plot_isf("/home/debian/water/TIP4P/2005/Tanaka_2018/rst/equili_isf_results.h5", "H2O")
plt.xlabel("Time (ps)")
plt.xscale("log")
plt.ylabel(r"$F_s(k,t)$")
plt.title("Self-intermediate Scattering Function")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("/home/debian/water/TIP4P/Ice/test/5e-6/isf_shear_rates.png", dpi=300)
plt.show()
