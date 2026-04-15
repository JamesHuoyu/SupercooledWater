from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SQRT_2PI = np.sqrt(2.0 * np.pi)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * SQRT_2PI)


@dataclass
class GaussianComponent:
    weight: float
    mu: float
    sigma: float


class TwoGaussianEM:
    """
    用 EM 对一维 zeta 数据做两高斯分解。

    说明：Nature Communications 2014 论文可确认水存在 two-state picture，
    但我这里无法从公开预览页逐字核对到你提到的“具体高斯分解公式”。
    因此这份脚本采用的是与 two-state picture 对应的标准两高斯混合实现：
        P(zeta) = w1 * N(mu1, sigma1) + w2 * N(mu2, sigma2)
    并按 mu1 < mu2 排序，通常可把低 mu 分量视作更无序/HDL-like，高 mu 分量视作更有序/LDL-like。
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-7, reg_sigma: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.reg_sigma = reg_sigma
        self.components: Tuple[GaussianComponent, GaussianComponent] | None = None
        self.log_likelihood_: float | None = None

    def _initialize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q25, q75 = np.percentile(x, [25, 75])
        mu = np.array([q25, q75], dtype=float)
        sigma0 = max(np.std(x), 1e-3)
        sigma = np.array([sigma0, sigma0], dtype=float)
        weight = np.array([0.5, 0.5], dtype=float)
        return weight, mu, sigma

    def fit(self, x: np.ndarray) -> "TwoGaussianEM":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 10:
            raise ValueError("数据点太少，无法稳健做两高斯分解")

        weight, mu, sigma = self._initialize(x)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            p1 = weight[0] * gaussian_pdf(x, mu[0], sigma[0])
            p2 = weight[1] * gaussian_pdf(x, mu[1], sigma[1])
            total = p1 + p2 + 1e-300

            r1 = p1 / total
            r2 = p2 / total
            responsibilities = np.vstack([r1, r2])

            nk = responsibilities.sum(axis=1) + 1e-12
            weight = nk / x.size
            mu = (responsibilities @ x) / nk
            sigma = np.sqrt((responsibilities @ ((x[None, :] - mu[:, None]) ** 2)) / nk)
            sigma = np.maximum(sigma, self.reg_sigma)

            ll = float(np.sum(np.log(total)))
            if abs(ll - prev_ll) < self.tol:
                prev_ll = ll
                break
            prev_ll = ll

        order = np.argsort(mu)
        weight = weight[order]
        mu = mu[order]
        sigma = sigma[order]

        self.components = (
            GaussianComponent(float(weight[0]), float(mu[0]), float(sigma[0])),
            GaussianComponent(float(weight[1]), float(mu[1]), float(sigma[1])),
        )
        self.log_likelihood_ = prev_ll
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise ValueError("请先 fit")
        x = np.asarray(x, dtype=float)
        c1, c2 = self.components
        p1 = c1.weight * gaussian_pdf(x, c1.mu, c1.sigma)
        p2 = c2.weight * gaussian_pdf(x, c2.mu, c2.sigma)
        total = p1 + p2 + 1e-300
        return np.column_stack([p1 / total, p2 / total])

    def pdf(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.components is None:
            raise ValueError("请先 fit")
        c1, c2 = self.components
        g1 = c1.weight * gaussian_pdf(x, c1.mu, c1.sigma)
        g2 = c2.weight * gaussian_pdf(x, c2.mu, c2.sigma)
        return g1, g2, g1 + g2


def load_values(csv_file: str | Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if column not in df.columns:
        raise ValueError(f"列 {column} 不在文件中，可用列: {list(df.columns)}")
    df = df[np.isfinite(df[column].to_numpy())].copy()
    return df


def compute_intersection(c1: GaussianComponent, c2: GaussianComponent) -> float | None:
    """返回两个加权高斯的交点，优先选择位于两均值之间的那个。"""
    a = 1.0 / (2.0 * c2.sigma**2) - 1.0 / (2.0 * c1.sigma**2)
    b = c1.mu / (c1.sigma**2) - c2.mu / (c2.sigma**2)
    c = (
        c2.mu**2 / (2.0 * c2.sigma**2)
        - c1.mu**2 / (2.0 * c1.sigma**2)
        + np.log((c2.weight / c2.sigma) / (c1.weight / c1.sigma))
    )
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        x = -c / b
        return float(x)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    roots = [(-b - np.sqrt(disc)) / (2.0 * a), (-b + np.sqrt(disc)) / (2.0 * a)]
    lo, hi = sorted([c1.mu, c2.mu])
    between = [r for r in roots if lo <= r <= hi]
    if between:
        return float(between[0])
    return float(sorted(roots, key=lambda r: abs(r - 0.5 * (lo + hi)))[0])


def bic_score(x: np.ndarray, log_likelihood: float, n_params: int) -> float:
    n = x.size
    return n_params * np.log(n) - 2.0 * log_likelihood


def fit_and_save(
    csv_file: str | Path,
    column: str,
    output_prefix: str | Path,
    bins: int = 250,
) -> None:
    df = load_values(csv_file, column)
    x = df[column].to_numpy(dtype=float)

    model = TwoGaussianEM().fit(x)
    probs = model.predict_proba(x)
    c1, c2 = model.components
    assert c1 is not None and c2 is not None
    z_cross = compute_intersection(c1, c2)

    df_out = df.copy()
    df_out["p_component_1"] = probs[:, 0]
    df_out["p_component_2"] = probs[:, 1]
    df_out["component_label"] = np.where(probs[:, 1] >= probs[:, 0], 2, 1)
    df_out["component_name"] = np.where(df_out["component_label"] == 1, "low-zeta", "high-zeta")

    prefix = Path(output_prefix)
    csv_out = prefix.with_suffix("")
    df_out.to_csv(str(csv_out) + "_classified.csv", index=False)

    summary = {
        "column": column,
        "n_points": int(x.size),
        "log_likelihood": float(model.log_likelihood_),
        "bic": float(bic_score(x, model.log_likelihood_, n_params=5)),
        "intersection": z_cross,
        "component_1": asdict(c1),
        "component_2": asdict(c2),
        "notes": {
            "ordering": "components are sorted by mean, mu1 < mu2",
            "interpretation": "component_1 is lower-zeta; component_2 is higher-zeta",
        },
    }
    with open(str(csv_out) + "_fit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    x_grid = np.linspace(x.min() - 0.2 * np.std(x), x.max() + 0.2 * np.std(x), 1000)
    g1, g2, gsum = model.pdf(x_grid)

    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=bins, density=True, alpha=0.45, label=f"{column} histogram")
    plt.plot(x_grid, g1, label="Gaussian 1 (low-zeta)")
    plt.plot(x_grid, g2, label="Gaussian 2 (high-zeta)")
    plt.plot(x_grid, gsum, linestyle="--", linewidth=2.0, label="sum")
    if z_cross is not None and np.isfinite(z_cross):
        plt.axvline(z_cross, linestyle=":", linewidth=1.5, label=f"intersection = {z_cross:.3f}")
    plt.xlabel(column)
    plt.ylabel("Probability density")
    plt.title(f"Two-Gaussian decomposition of {column}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(str(csv_out) + "_fit.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Gaussian decomposition for zeta or zeta_cg")
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--column", default="zeta")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--bins", type=int, default=250)
    args = parser.parse_args()

    fit_and_save(args.csv_file, args.column, args.output_prefix, bins=args.bins)


if __name__ == "__main__":
    main()
