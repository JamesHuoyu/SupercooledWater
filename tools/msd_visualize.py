import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def cal_diffusion_coefficient(time_ps, msd):
    """计算扩散系数D，假设MSD = 2nDt，其中特征长度n为1"""
    n_points = len(time_ps)
    start_index = int(n_points * 0.8)
    x_fit = time_ps[start_index:]
    y_fit = msd[start_index:]
    popt, _ = curve_fit(lambda t, D: 2 * 1 * D * t, x_fit, y_fit)
    D = popt[0]
    residuals = y_fit - (2 * 1 * D * x_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Fitted Diffusion Coefficient D = {D:.5e} Å²/ps, R² = {r_squared:.5f}")
    return D, r_squared, start_index


def plot_msd(filepath: str, key: str, color: str, linestyle: str, label: str = None):
    """绘制MSD曲线，支持自定义颜色和线型"""
    df = pd.read_hdf(filepath, key=key)
    x = df["time_ps"][0:]
    y = df["MSD_A2"][0:]
    print(f"MSD Analysis for {key}:")
    D, r2, fit_start = cal_diffusion_coefficient(x, y)

    # 使用指定的颜色和线型绘制
    plt.plot(x, y, color=color, linestyle=linestyle, linewidth=2, label=label)
    # 可选：标记拟合起始点
    # plt.axvline(x=x[fit_start], color=color, linestyle=":", alpha=0.5)


if __name__ == "__main__":
    print("Starting MSD visualization...")
    # 定义剪切率和对应的颜色
    shear_rates = {
        "246": "blue",
        "2.5e-4": "red",
        "7.5e-5": "green",
        "2.5e-5": "orange",
        "1e-5": "purple",
    }

    # 定义方向对应的线型
    direction_linestyles = {"z": "-", "x": "--"}  # 实线表示z方向  # 虚线表示x方向

    # 创建图形
    plt.figure(figsize=(10, 7))

    filepath = "/home/debian/water/TIP4P/Ice/test/msd_results.h5"
    # plot_msd(filepath, "1e-6", color="blue", linestyle="-", label="1e-6(z)")
    plot_msd(filepath, "equili", color="black", linestyle="-", label="equili")
    plot_msd(filepath, "1e-6-z", color="blue", linestyle="-", label="1e-6(z)")
    plot_msd(filepath, "1e-6-y", color="blue", linestyle="-.", label="1e-6(y)")
    plot_msd(filepath, "1e-6-x", color="blue", linestyle="--", label="1e-6(x)")
    plot_msd(filepath, "5e-6-z", color="yellow", linestyle="-", label="5e-6(z)")
    plot_msd(filepath, "5e-6-y", color="yellow", linestyle="-.", label="5e-6(y)")
    plot_msd(filepath, "5e-6-x", color="yellow", linestyle="--", label="5e-6(x)")
    plot_msd(filepath, "1e-5-z", color="red", linestyle="-", label="1e-5(z)")
    plot_msd(filepath, "1e-5-y", color="red", linestyle="-.", label="1e-5(y)")
    plot_msd(filepath, "1e-5-x", color="red", linestyle="--", label="1e-5(x)")
    plot_msd(filepath, "5e-5-z", color="green", linestyle="-", label="5e-5(z)")
    plot_msd(filepath, "5e-5-y", color="green", linestyle="-.", label="5e-5(y)")
    plot_msd(filepath, "5e-5-x", color="green", linestyle="--", label="5e-5(x)")
    plot_msd(filepath, "1e-4-z", color="orange", linestyle="-", label="1e-4(z)")
    plot_msd(filepath, "1e-4-y", color="orange", linestyle="-.", label="1e-4(y)")
    plot_msd(filepath, "1e-4-x", color="orange", linestyle="--", label="1e-4(x)")
    print("Finished plotting all MSD curves.")
    # # 绘制所有曲线
    # for shear_rate, color in shear_rates.items():
    #     # 处理特殊情况的文件路径
    #     if shear_rate == "246":
    #         filepath = "equili_msd_results.h5"
    #     else:
    #         filepath = "/home/debian/water/TIP4P/2005/2020/rst/4096/new_msd_results.h5"

    #     # 绘制z方向（无后缀）
    #     z_key = shear_rate
    #     z_label = f"γ={shear_rate} (z)" if shear_rate != "246" else f"Equilibrium (z)"
    #     plot_msd(filepath, z_key, color, direction_linestyles["z"], z_label)

    #     # 绘制x方向（带-x后缀）
    #     x_key = f"{shear_rate}-x"
    #     x_label = f"γ={shear_rate} (x)" if shear_rate != "246" else f"Equilibrium (x)"
    #     plot_msd(filepath, x_key, color, direction_linestyles["x"], x_label)

    # 设置图形属性
    plt.xlabel("Time (ps)", fontsize=12)
    plt.xscale("log")
    plt.ylabel("MSD (Å²)", fontsize=12)
    plt.yscale("log")
    plt.title("Mean Squared Displacement by Shear Rate and Direction", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)  # 图例放在右侧
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 可选：添加方向说明的注释
    # plt.figtext(
    #     0.02,
    #     0.02,
    #     "Solid lines: z-direction, Dashed lines: x-direction",
    #     fontsize=10,
    #     style="italic",
    #     alpha=0.7,
    # )
    # plt.savefig("/home/debian/water/TIP4P/2005/2020/rst/msd_shear_direction.png", dpi=300)
    plt.show()
