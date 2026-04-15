"""
Stress-Structure Correlation Analysis Script

This script analyzes the correlation between shear stress and local structure (Q4, Q6)
in supercooled water under shear flow.
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from scipy import stats
from tqdm import tqdm

# Import existing modules for structure analysis
import sys
sys.path.append('/home/debian/water/tools')

# Try to import Q4 and Q6 calculation modules
try:
    from q import TetrahedralOrder as Q4Calculator
    Q4_AVAILABLE = True
except ImportError:
    print("Warning: q.py not found. Q4 calculation will be skipped.")
    Q4_AVAILABLE = False

try:
    from Q import BondOrientationalOrder as Q6Calculator
    Q6_AVAILABLE = True
except ImportError:
    print("Warning: Q.py not found. Q6 calculation will be skipped.")
    Q6_AVAILABLE = False

from stress_analysis import (
    load_universe_with_stress,
    get_positions_and_sxy,
    sxy_structure_correlation,
    map_sxy_to_grid,
    compute_sxy_time_evolution,
)


def apply_shear_correction(coords, shear_rate, time_step, ref_y=25.0):
    """
    Apply shear correction to coordinates to remove affine deformation.

    Args:
        coords: (n_frames, n_atoms, 3) array of coordinates
        shear_rate: Shear rate in 1/ps
        time_step: Time step in ps
        ref_y: Reference y position for shear calculation

    Returns:
        corrected_coords: Coordinates with shear deformation removed
    """
    T = coords.shape[0]
    y = coords[:, :, 1] - ref_y  # shape (T, N)
    gamma_dt = shear_rate * time_step
    shear_disp = gamma_dt * np.cumsum(y, axis=0)  # shape (T, N)
    corrected_coords = coords.copy()
    corrected_coords[:, :, 0] -= shear_disp
    return corrected_coords


def compute_q4(u, ref_y=25.0):
    """
    Compute tetrahedral order parameter Q4 for oxygen atoms.

    Args:
        u: MDAnalysis Universe
        ref_y: Reference y position for shear correction

    Returns:
        q4_values: (n_frames, n_atoms) array of Q4 values
    """
    if not Q4_AVAILABLE:
        print("Q4 calculation not available.")
        return None

    # Extract oxygen atoms
    o_atoms = u.select_atoms('type 1')
    n_frames = len(u.trajectory)
    q4_values = np.zeros((n_frames, len(o_atoms)))

    calculator = Q4Calculator(cutoff=3.5)

    for i, ts in enumerate(tqdm(u.trajectory, desc="Computing Q4")):
        for j, atom in enumerate(o_atoms):
            # Get neighbors
            neighbors = u.select_atoms(
                f'not index {atom.index} and around 3.5 type 1',
                ts=ts
            )
            q4_values[i, j] = calculator.compute_q4(atom, neighbors)

    return q4_values


def compute_q6(u, ref_y=25.0):
    """
    Compute bond orientational order parameter Q6 for oxygen atoms.

    Args:
        u: MDAnalysis Universe
        ref_y: Reference y position for shear correction

    Returns:
        q6_values: (n_frames, n_atoms) array of Q6 values
    """
    if not Q6_AVAILABLE:
        print("Q6 calculation not available.")
        return None

    # Extract oxygen atoms
    o_atoms = u.select_atoms('type 1')
    n_frames = len(u.trajectory)
    q6_values = np.zeros((n_frames, len(o_atoms)))

    calculator = Q6Calculator(cutoff=3.5)

    for i, ts in enumerate(tqdm(u.trajectory, desc="Computing Q6")):
        for j, atom in enumerate(o_atoms):
            q6_values[i, j] = calculator.compute_q6(atom, u, ts)

    return q6_values


def analyze_stress_structure_correlation(dump_file, shear_rate, time_step=0.025):
    """
    Main analysis function for stress-structure correlation.

    Args:
        dump_file: Path to LAMMPS dump file with stress data
        shear_rate: Shear rate in 1/ps
        time_step: Time step in ps

    Returns:
        Dictionary with analysis results
    """
    print(f"Loading dump file: {dump_file}")

    # Load universe with stress data
    u = load_universe_with_stress(dump_file)

    # Extract positions and sxy
    print("Extracting positions and sxy stress...")
    coords, sxy = get_positions_and_sxy(u)

    n_frames, n_atoms = sxy.shape
    print(f"Loaded {n_frames} frames, {n_atoms} atoms")

    # Apply shear correction to coordinates
    print("Applying shear correction to coordinates...")
    coords_corrected = apply_shear_correction(coords, shear_rate, time_step)

    # Compute structural parameters
    results = {
        'sxy': sxy,
        'coords_corrected': coords_corrected,
        'n_frames': n_frames,
        'n_atoms': n_atoms,
    }

    if Q4_AVAILABLE:
        print("\nComputing Q4 (tetrahedral order parameter)...")
        q4 = compute_q4(u)
        results['q4'] = q4

        # Compute correlation
        corr, p_val = sxy_structure_correlation(sxy, q4)
        results['sxy_q4_correlation'] = corr
        results['sxy_q4_pvalue'] = p_val
        print(f"Sxy-Q4 correlation: r = {corr:.3f}, p = {p_val:.2e}")

    if Q6_AVAILABLE:
        print("\nComputing Q6 (bond orientational order parameter)...")
        q6 = compute_q6(u)
        results['q6'] = q6

        # Compute correlation
        corr, p_val = sxy_structure_correlation(sxy, q6)
        results['sxy_q6_correlation'] = corr
        results['sxy_q6_pvalue'] = p_val
        print(f"Sxy-Q6 correlation: r = {corr:.3f}, p = {p_val:.2e}")

    # Time evolution analysis
    print("\nComputing sxy time evolution...")
    time, mean_sxy, std_sxy = compute_sxy_time_evolution(sxy, time_step)
    results['time'] = time
    results['mean_sxy'] = mean_sxy
    results['std_sxy'] = std_sxy

    return results


def plot_sxy_distribution_evolution(results, time_step=0.025):
    """
    Plot sxy distribution evolution over time.

    Args:
        results: Dictionary with analysis results
        time_step: Time step in ps
    """
    sxy = results['sxy']
    time = results['time']
    mean_sxy = results['mean_sxy']
    std_sxy = results['std_sxy']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Time evolution of mean and std
    axes[0].plot(time, mean_sxy, 'b-', label='Mean')
    axes[0].fill_between(time, mean_sxy - std_sxy, mean_sxy + std_sxy,
                               alpha=0.3, label='± std')
    axes[0].set_xlabel('Time (ps)')
    axes[0].set_ylabel('Shear Stress sxy (bar)')
    axes[0].set_title('Shear Stress Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribution evolution
    # Show distribution at different time points
    time_points = [0, len(time)//4, len(time)//2, 3*len(time)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))

    for t_idx, color in zip(time_points, colors):
        sxy_dist = sxy[t_idx, :].flatten()
        axes[1].hist(sxy_dist, bins=50, alpha=0.5, density=True,
                     color=color, label=f't={time[t_idx]:.1f}ps')

    axes[1].set_xlabel('Shear Stress sxy (bar)')
    axes[1].set_ylabel('Probability Density')
    axes[1].set_title('Stress Distribution at Different Times')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sxy_structure_scatter(results):
    """
    Plot scatter plots of sxy vs structural parameters.

    Args:
        results: Dictionary with analysis results
    """
    sxy = results['sxy'].flatten()
    coords_corrected = results['coords_corrected']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    idx = 0
    if 'q4' in results and results['q4'] is not None:
        q4 = results['q4'].flatten()
        corr = results.get('sxy_q4_correlation', np.nan)
        p_val = results.get('sxy_q4_pvalue', np.nan)

        axes[idx].scatter(q4, sxy, alpha=0.5, s=5)
        axes[idx].set_xlabel('Q4 (Tetrahedral Order)')
        axes[idx].set_ylabel('Shear Stress sxy (bar)')
        axes[idx].set_title(f'Stress vs Q4 (r={corr:.3f}, p={p_val:.2e})')
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    if 'q6' in results and results['q6'] is not None:
        q6 = results['q6'].flatten()
        corr = results.get('sxy_q6_correlation', np.nan)
        p_val = results.get('sxy_q6_pvalue', np.nan)

        axes[idx].scatter(q6, sxy, alpha=0.5, s=5)
        axes[idx].set_xlabel('Q6 (Bond Orientational Order)')
        axes[idx].set_ylabel('Shear Stress sxy (bar)')
        axes[idx].set_title(f'Stress vs Q6 (r={corr:.3f}, p={p_val:.2e})')
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    plt.tight_layout()
    return fig


def plot_spatial_stress_map(results, box):
    """
    Plot spatial distribution of stress at specific timesteps.

    Args:
        results: Dictionary with analysis results
        box: Box dimensions [Lx, Ly, Lz]
    """
    sxy = results['sxy']
    coords_corrected = results['coords_corrected']
    n_frames = sxy.shape[0]

    # Select frames to visualize (early, middle, late)
    frame_indices = [0, n_frames//4, n_frames//2, -1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i // 2, i % 2]

        # Map sxy to grid
        H, edges = map_sxy_to_grid(
            sxy[frame_idx:frame_idx+1],
            coords_corrected[frame_idx:frame_idx+1],
            box,
            grid_size=50
        )

        im = ax.imshow(H.T, origin='lower', cmap='RdBu_r',
                     extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]],
                     aspect='auto')
        plt.colorbar(im, ax=ax, label='Shear Stress sxy (bar)')
        ax.set_xlabel('X Position (Å)')
        ax.set_ylabel('Y Position (Å)')
        ax.set_title(f'Spatial Stress Map (t={frame_idx})')

    plt.tight_layout()
    return fig


def save_results(results, output_file):
    """
    Save analysis results to file.

    Args:
        results: Dictionary with analysis results
        output_file: Output file path
    """
    np.savez(output_file, **results)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Stress-Structure Correlation Analysis')
    parser.add_argument('--dump', type=str, required=True,
                        help='LAMMPS dump file with stress data')
    parser.add_argument('--shear_rate', type=float, required=True,
                        help='Shear rate in 1/ps')
    parser.add_argument('--time_step', type=float, default=0.025,
                        help='Time step in ps (default: 0.025)')
    parser.add_argument('--output', type=str, default='stress_structure_results.npz',
                        help='Output file (default: stress_structure_results.npz)')

    args = parser.parse_args()

    print("=" * 60)
    print("Stress-Structure Correlation Analysis")
    print("=" * 60)

    # Run analysis
    results = analyze_stress_structure_correlation(
        args.dump,
        args.shear_rate,
        args.time_step
    )

    # Save results
    save_results(results, args.output)

    # Generate plots
    print("\nGenerating plots...")
    fig1 = plot_sxy_distribution_evolution(results, args.time_step)
    plt.savefig('sxy_distribution_evolution.png', dpi=300, bbox_inches='tight')
    print("Saved: sxy_distribution_evolution.png")

    if 'q4' in results or 'q6' in results:
        fig2 = plot_sxy_structure_scatter(results)
        plt.savefig('sxy_structure_scatter.png', dpi=300, bbox_inches='tight')
        print("Saved: sxy_structure_scatter.png")

    # Get box dimensions
    u = load_universe_with_stress(args.dump)
    box = u.dimensions
    fig3 = plot_spatial_stress_map(results, box)
    plt.savefig('sxy_spatial_maps.png', dpi=300, bbox_inches='tight')
    print("Saved: sxy_spatial_maps.png")

    print("\nAnalysis complete!")
