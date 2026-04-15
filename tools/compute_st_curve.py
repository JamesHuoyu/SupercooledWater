"""
Compute S(q, t) curve for q -> 0 (intermediate scattering function)

This script calculates the time-dependent structure factor S(q, t) at small q,
which is related to the intermediate scattering function F(q, t).
For q -> 0, this is sensitive to long-wavelength density fluctuations.
"""

import numpy as np
import matplotlib.pyplot as plt
from tools.new_chi_ultrafast import compute_averaged, q_shell_vectors
import time
from multiprocessing import cpu_count


def compute_st_curve(
    trajectories,
    q_values,
    t_values,
    Lx,
    Ly,
    Lz,
    a=1.0,
    dq=0.1,
    parallel=True,
    n_workers=None,
    verbose=True,
):
    """
    Compute S(q, t) for a range of t values at given q values

    :param trajectories: shape (N_times_total, N_numbers, 3)
    :param q_values: list of q magnitudes to compute (e.g., [0.01, 0.1, 1.0])
    :param t_values: list/array of time steps to compute
    :param Lx, Ly, Lz: box dimensions
    :param a: cutoff distance for overlap
    :param dq: q shell half-width for averaging
    :param parallel: whether to use parallel computation
    :param n_workers: number of parallel workers
    :param verbose: print progress
    :return: dict {q: [S(t) values for each t]}
    """
    N_times_total = trajectories.shape[0]
    max_t = max(t_values)

    if max_t >= N_times_total:
        raise ValueError(f"Max t={max_t} exceeds trajectory length {N_times_total}")

    if n_workers is None:
        n_workers = cpu_count()

    results = {q: [] for q in q_values}

    total_computations = len(q_values) * len(t_values)

    if verbose:
        print(f"Computing S(q, t) curve:")
        print(f"  Trajectory shape: {trajectories.shape}")
        print(f"  q values: {q_values}")
        print(f"  t range: {min(t_values)} to {max(t_values)}")
        print(f"  Total computations: {total_computations}")
        print(f"  Using {n_workers} workers")
        print(f"  {'-'*60}")

    start_time = time.time()

    for i, q in enumerate(q_values):
        if verbose:
            print(f"\nq = {q:.4f}")

        q_results = []

        for j, t in enumerate(t_values):
            if verbose and j % max(1, len(t_values) // 10) == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / ((i * len(t_values) + j + 1) / total_computations) - elapsed
                progress = (i * len(t_values) + j + 1) / total_computations * 100
                print(
                    f"  t = {t:4d} ({progress:5.1f}%) - elapsed: {elapsed:.1f}s, est. remaining: {remaining:.1f}s",
                    end="\r",
                )

            try:
                S = compute_averaged(q, dq, trajectories, t, Lx, Ly, Lz, a)
                q_results.append(S)
            except Exception as e:
                print(f"\nError at q={q}, t={t}: {e}")
                q_results.append(np.nan)

        results[q] = np.array(q_results)

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n{'-'*60}")
        print(f"Completed in {elapsed:.2f} seconds")

    return results


def plot_st_curve(
    results, t_values, q_values, Lx, Ly, Lz, title="S(q, t) for q -> 0", save_path=None
):
    """
    Plot S(q, t) curves for different q values

    :param results: dict {q: [S(t) values]}
    :param t_values: array of time steps
    :param q_values: list of q values
    :param Lx, Ly, Lz: box dimensions
    :param title: plot title
    :param save_path: path to save figure
    """
    plt.figure(figsize=(10, 6))

    for q in q_values:
        S_t = results[q]
        plt.plot(t_values, S_t, "o-", label=f"q = {q:.3f}", markersize=4)

    plt.xlabel("Time step t", fontsize=12)
    plt.ylabel("S(q, t)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add box size annotation
    plt.text(
        0.02,
        0.98,
        f"Box: {Lx:.1f} × {Ly:.1f} × {Lz:.1f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def analyze_s0_limit(results, t_values, q_values):
    """
    Analyze the q -> 0 limit by extrapolating S(q, t) for each t

    Returns the extrapolated S(0, t) values
    """
    S0_values = []

    print("\n" + "=" * 60)
    print("EXTRAPOLATION TO q -> 0")
    print("=" * 60)
    print(f"{'t':<8} {'S(0, t)':<12} {'Fit R²':<10}")
    print("-" * 60)

    for i, t in enumerate(t_values):
        # Collect S values at different q
        S_vals = [results[q][i] for q in q_values]

        # Fit to S(q) = A + B*q² (for small q)
        # Or S(q) = S(0) + C*q for linear approximation
        q_arr = np.array(q_values)
        S_arr = np.array(S_vals)

        # Remove NaN values
        valid = ~np.isnan(S_arr)
        if np.sum(valid) < 2:
            print(f"{t:<8} {'N/A':<12} {'-':<10}")
            S0_values.append(np.nan)
            continue

        # Quadratic fit: S(q) = a + b*q + c*q²
        coeffs = np.polyfit(q_arr[valid], S_arr[valid], 2)
        S0 = coeffs[2]  # constant term

        # Calculate R²
        S_fit = np.polyval(coeffs, q_arr[valid])
        ss_res = np.sum((S_arr[valid] - S_fit) ** 2)
        ss_tot = np.sum((S_arr[valid] - np.mean(S_arr[valid])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"{t:<8} {S0:<12.6f} {r2:<10.4f}")
        S0_values.append(S0)

    print("=" * 60)

    return np.array(S0_values)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Parameters
    N_times = 100
    N_particles = 50
    Lx = Ly = Lz = 10.0
    a = 1.0

    # Generate test trajectories (Brownian motion-like)
    trajectories = np.random.rand(N_times, N_particles, 3) * Lx

    print("Test trajectories shape:", trajectories.shape)

    # Define time steps to compute
    t_max = 20
    t_values = np.arange(0, t_max + 1, 1)

    # Define small q values
    q_values = [0.01, 0.05, 0.1, 0.2, 0.5]

    # Compute S(q, t) curves
    results = compute_st_curve(
        trajectories=trajectories,
        q_values=q_values,
        t_values=t_values,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        a=a,
        dq=0.1,
    )

    # Plot results
    plot_st_curve(
        results=results,
        t_values=t_values,
        q_values=q_values,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        title="S(q, t) for Small q",
        save_path="sqt_curve.png",
    )

    # Analyze q -> 0 limit
    S0_t = analyze_s0_limit(results, t_values, q_values)

    # Plot S(0, t) separately
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, S0_t, "o-", label="S(0, t) (extrapolated)", markersize=6)
    plt.xlabel("Time step t", fontsize=12)
    plt.ylabel("S(0, t)", fontsize=12)
    plt.title("q -> 0 Limit: S(0, t)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("s0t_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nResults saved to:")
    print("  - sqt_curve.png (S(q, t) for different q)")
    print("  - s0t_curve.png (S(0, t) extrapolated)")
