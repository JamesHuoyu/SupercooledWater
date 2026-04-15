import json
import numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =========================
# 基础工具
# =========================


def first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> Optional[float]:
    """
    返回 y 首次降到 threshold 以下时对应的 x（线性插值）。
    x: 1D array, 单调递增
    y: 1D array
    """
    for i in range(1, len(y)):
        if y[i] <= threshold < y[i - 1]:
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[i - 1], y[i]
            if y1 == y0:
                return float(x1)
            frac = (threshold - y0) / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
    return None


def moving_average_trailing(arr: np.ndarray, window: int) -> np.ndarray:
    """
    单边回看平均.
    输入: arr.shape = (n_frames, n_particles)
    输出: same shape, 前 window-1 帧填 nan
    """
    n_frames, n_particles = arr.shape
    out = np.full_like(arr, np.nan, dtype=np.float64)
    csum = np.cumsum(arr, axis=0, dtype=np.float64)
    out[window - 1] = csum[window - 1] / window
    for t in range(window, n_frames):
        out[t] = (csum[t] - csum[t - window]) / window
    return out


def zscore_ignore_nan(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < 1e-12:
        return x - mu
    return (x - mu) / sd


# =========================
# A. tau_noise
# =========================


def zeta_autocorrelation(zeta: np.ndarray, max_lag: int) -> np.ndarray:
    """
    计算 C_zeta(lag)
    zeta.shape = (n_frames, n_particles)
    """
    z = zeta.astype(np.float64)
    z = z - np.nanmean(z, axis=0, keepdims=True)
    denom = np.nanmean(z * z)
    corr = np.zeros(max_lag + 1, dtype=np.float64)

    for lag in range(max_lag + 1):
        a = z[: len(z) - lag]
        b = z[lag:]
        corr[lag] = np.nanmean(a * b) / denom if denom > 1e-15 else np.nan
    return corr


def sign_persistence(zeta: np.ndarray, max_lag: int, zero_eps: float = 0.0) -> np.ndarray:
    """
    计算 S_sign(lag)
    零附近可以忽略，避免阈值附近抖动过强。
    """
    signs = np.sign(zeta)
    if zero_eps > 0:
        mask_zero = np.abs(zeta) <= zero_eps
        signs = signs.astype(np.float64)
        signs[mask_zero] = np.nan

    pers = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        s0 = signs[: len(signs) - lag]
        s1 = signs[lag:]
        valid = ~np.isnan(s0) & ~np.isnan(s1)
        if np.any(valid):
            pers[lag] = np.mean(s0[valid] == s1[valid])
        else:
            pers[lag] = np.nan
    return pers


def dwell_times_from_sign(zeta: np.ndarray, zero_eps: float = 0.0) -> Dict[str, np.ndarray]:
    """
    对每个粒子提取 sign 序列的 dwell times.
    返回 T(+), D(-), N(0/near zero) 三类.
    """
    states = np.zeros_like(zeta, dtype=np.int8)
    states[zeta > zero_eps] = 1
    states[zeta < -zero_eps] = -1

    dwell_T, dwell_D, dwell_N = [], [], []

    n_frames, n_particles = states.shape
    for i in range(n_particles):
        seq = states[:, i]
        run_state = seq[0]
        run_len = 1
        for t in range(1, n_frames):
            if seq[t] == run_state:
                run_len += 1
            else:
                if run_state == 1:
                    dwell_T.append(run_len)
                elif run_state == -1:
                    dwell_D.append(run_len)
                else:
                    dwell_N.append(run_len)
                run_state = seq[t]
                run_len = 1
        # flush
        if run_state == 1:
            dwell_T.append(run_len)
        elif run_state == -1:
            dwell_D.append(run_len)
        else:
            dwell_N.append(run_len)

    return {
        "T": np.array(dwell_T, dtype=int),
        "D": np.array(dwell_D, dtype=int),
        "N": np.array(dwell_N, dtype=int),
    }


@dataclass
class NoiseTimescaleResult:
    lags: np.ndarray
    C_zeta: np.ndarray
    S_sign: np.ndarray
    dwell_T: np.ndarray
    dwell_D: np.ndarray
    dwell_N: np.ndarray
    tau_corr_e1: Optional[float]
    tau_sign_half: Optional[float]
    tau_dwell_T_mean: Optional[float]
    tau_dwell_D_mean: Optional[float]


def compute_tau_noise(
    zeta: np.ndarray,
    max_lag: int,
    zero_eps: float = 0.0,
) -> NoiseTimescaleResult:
    lags = np.arange(max_lag + 1)
    C = zeta_autocorrelation(zeta, max_lag=max_lag)
    S = sign_persistence(zeta, max_lag=max_lag, zero_eps=zero_eps)
    dwell = dwell_times_from_sign(zeta, zero_eps=zero_eps)

    tau_corr_e1 = first_crossing(lags, C, np.exp(-1))
    tau_sign_half = first_crossing(lags, S, 0.5)

    tau_dwell_T_mean = float(np.mean(dwell["T"])) if len(dwell["T"]) else None
    tau_dwell_D_mean = float(np.mean(dwell["D"])) if len(dwell["D"]) else None

    return NoiseTimescaleResult(
        lags=lags,
        C_zeta=C,
        S_sign=S,
        dwell_T=dwell["T"],
        dwell_D=dwell["D"],
        dwell_N=dwell["N"],
        tau_corr_e1=tau_corr_e1,
        tau_sign_half=tau_sign_half,
        tau_dwell_T_mean=tau_dwell_T_mean,
        tau_dwell_D_mean=tau_dwell_D_mean,
    )


# =========================
# B. tau_pred
# =========================


def spearman_ignore_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 5:
        return np.nan
    return float(spearmanr(x[mask], y[mask]).correlation)


@dataclass
class PredTimescaleResult:
    windows: np.ndarray  # Δ list
    t_targets: np.ndarray  # target times
    corr_map: np.ndarray  # shape = (n_windows, n_t_targets)
    tau_pred_peak: Dict[float, Optional[int]]
    tau_pred_95: Dict[float, Optional[int]]


def compute_tau_pred(
    zeta: np.ndarray,
    t0: int,
    propensities: np.ndarray,
    t_targets: np.ndarray,
    windows: List[int],
) -> PredTimescaleResult:
    """
    zeta.shape = (n_frames, n_particles)
    propensities.shape = (n_t_targets, n_particles)
    """
    windows = np.array(windows, dtype=int)
    t_targets = np.array(t_targets)

    corr_map = np.full((len(windows), len(t_targets)), np.nan, dtype=np.float64)

    for wi, w in enumerate(windows):
        z_tilde = moving_average_trailing(zeta, window=w)
        if t0 >= len(z_tilde):
            raise ValueError("t0 out of range")
        x = z_tilde[t0]  # (n_particles,)
        for ti in range(len(t_targets)):
            y = propensities[ti]
            corr_map[wi, ti] = spearman_ignore_nan(x, y)

    tau_pred_peak = {}
    tau_pred_95 = {}

    for ti, tt in enumerate(t_targets):
        col = corr_map[:, ti]
        if np.all(np.isnan(col)):
            tau_pred_peak[float(tt)] = None
            tau_pred_95[float(tt)] = None
            continue

        idx_peak = int(np.nanargmax(np.abs(col)))
        peak_val = np.abs(col[idx_peak])
        tau_pred_peak[float(tt)] = int(windows[idx_peak])

        target = 0.95 * peak_val
        cand = np.where(np.abs(col) >= target)[0]
        tau_pred_95[float(tt)] = int(windows[cand[0]]) if len(cand) else None

    return PredTimescaleResult(
        windows=windows,
        t_targets=t_targets,
        corr_map=corr_map,
        tau_pred_peak=tau_pred_peak,
        tau_pred_95=tau_pred_95,
    )


# =========================
# C. tau_patch
# =========================


def remove_affine_shear(
    positions: np.ndarray,
    times: np.ndarray,
    shear_rate: float,
    flow_dir: int = 0,
    grad_dir: int = 1,
    ref_time_index: int = 0,
) -> np.ndarray:
    """
    简单的去仿射剪切:
    x' = x - gamma(t) * y
    其中 gamma(t)= shear_rate * (t - t_ref)

    注意:
    1) 这里只是接口示意，具体请按你们 LAMMPS/Lees-Edwards/SLLOD 的定义修改
    2) 如果你们已有 nonaffine 坐标，这一步可以跳过
    """
    pos = positions.copy().astype(np.float64)
    t_ref = times[ref_time_index]

    for ti, t in enumerate(times):
        gamma = shear_rate * (t - t_ref)
        pos[ti, :, flow_dir] -= gamma * pos[ti, :, grad_dir]
    return pos


def deposit_field_to_grid(
    positions: np.ndarray,
    values: np.ndarray,
    box: np.ndarray,
    grid_shape: Tuple[int, int, int],
    gaussian_sigma_grid: float = 1.0,
) -> np.ndarray:
    """
    把粒子值沉积到 3D 网格，再做 Gaussian blur.
    positions.shape = (n_particles, 3)
    values.shape = (n_particles,)
    box = np.array([Lx, Ly, Lz])
    """
    nx, ny, nz = grid_shape
    field = np.zeros(grid_shape, dtype=np.float64)

    frac = positions / box[None, :]
    frac = frac % 1.0

    ix = np.floor(frac[:, 0] * nx).astype(int) % nx
    iy = np.floor(frac[:, 1] * ny).astype(int) % ny
    iz = np.floor(frac[:, 2] * nz).astype(int) % nz

    for a, b, c, v in zip(ix, iy, iz, values):
        field[a, b, c] += v

    field = gaussian_filter(field, sigma=gaussian_sigma_grid, mode="wrap")
    return field


@dataclass
class PatchTimescaleResult:
    lags: np.ndarray
    Q_patch: np.ndarray
    tau_patch_e1: Optional[float]


def compute_tau_patch(
    zeta: np.ndarray,
    positions: np.ndarray,
    times: np.ndarray,
    box: np.ndarray,
    delta_window: int,
    max_lag: int,
    grid_shape: Tuple[int, int, int] = (32, 32, 32),
    gaussian_sigma_grid: float = 1.0,
    shear_rate: Optional[float] = None,
) -> PatchTimescaleResult:
    """
    1) 先对 zeta 做 trailing average
    2) 用去仿射后的坐标构造网格场
    3) 计算 patch overlap
    """
    z_tilde = moving_average_trailing(zeta, window=delta_window)

    if shear_rate is not None:
        pos_na = remove_affine_shear(
            positions,
            times=times,
            shear_rate=shear_rate,
            flow_dir=0,
            grad_dir=1,
            ref_time_index=0,
        )
    else:
        pos_na = positions.copy()

    valid_start = delta_window - 1
    n_frames = len(times)

    fields = []
    for t in range(valid_start, n_frames):
        field = deposit_field_to_grid(
            positions=pos_na[t],
            values=z_tilde[t],
            box=box,
            grid_shape=grid_shape,
            gaussian_sigma_grid=gaussian_sigma_grid,
        )
        fields.append(field)
    fields = np.array(fields)  # shape=(n_valid_frames, nx, ny, nz)

    Q = np.zeros(max_lag + 1, dtype=np.float64)
    denom = np.mean(fields * fields)

    for lag in range(max_lag + 1):
        a = fields[: len(fields) - lag]
        b = fields[lag:]
        Q[lag] = np.mean(a * b) / denom if denom > 1e-15 else np.nan

    lags = np.arange(max_lag + 1)
    tau_patch_e1 = first_crossing(lags, Q, np.exp(-1))
    return PatchTimescaleResult(lags=lags, Q_patch=Q, tau_patch_e1=tau_patch_e1)


# =========================
# IO / 示例主程序
# =========================


def save_noise_result(res: NoiseTimescaleResult, out_json: str):
    payload = {
        "lags": res.lags.tolist(),
        "C_zeta": res.C_zeta.tolist(),
        "S_sign": res.S_sign.tolist(),
        "dwell_T": res.dwell_T.tolist(),
        "dwell_D": res.dwell_D.tolist(),
        "dwell_N": res.dwell_N.tolist(),
        "tau_corr_e1": res.tau_corr_e1,
        "tau_sign_half": res.tau_sign_half,
        "tau_dwell_T_mean": res.tau_dwell_T_mean,
        "tau_dwell_D_mean": res.tau_dwell_D_mean,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_pred_result(res: PredTimescaleResult, out_json: str):
    payload = {
        "windows": res.windows.tolist(),
        "t_targets": res.t_targets.tolist(),
        "corr_map": res.corr_map.tolist(),
        "tau_pred_peak": res.tau_pred_peak,
        "tau_pred_95": res.tau_pred_95,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_patch_result(res: PatchTimescaleResult, out_json: str):
    payload = {
        "lags": res.lags.tolist(),
        "Q_patch": res.Q_patch.tolist(),
        "tau_patch_e1": res.tau_patch_e1,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
