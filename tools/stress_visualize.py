"""
Stress Visualization Module

This module provides functions for visualizing stress-related data from molecular dynamics simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_stress_heatmap(sxy_grid, edges, time=None, title='Stress Heatmap',
                       cmap='RdBu_r', vmin=None, vmax=None):
    """
    Create 2D heatmap of stress vs (position, time).

    Args:
        sxy_grid: 2D grid of stress values
        edges: Tuple of (x_edges, y_edges) grid edges
        time: Optional time value for title
        title: Plot title
        cmap: Colormap name
        vmin, vmax: Color scale limits

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(sxy_grid.T, origin='lower', cmap=cmap,
                     extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]],
                     aspect='auto', vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, label='Shear Stress sxy (bar)')
    ax.set_xlabel('X Position (Å)')
    ax.set_ylabel('Y Position (Å)')
    ax.set_title(title)

    return fig, ax


def plot_stress_evolution(time, mean_sxy, std_sxy, title='Stress Evolution'):
    """
    Plot stress vs time for different regions.

    Args:
        time: Array of time values
        mean_sxy: Array of mean stress values
        std_sxy: Array of std stress values
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, mean_sxy, 'b-', label='Mean sxy', linewidth=2)
    ax.fill_between(time, mean_sxy - std_sxy, mean_sxy + std_sxy,
                   alpha=0.3, label='± 1 std')

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Shear Stress sxy (bar)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_stress_structure_scatter(sxy, structural_param, struct_name='Q4',
                                corr_coef=None, p_value=None,
                                title='Stress vs Structure'):
    """
    Scatter plots of stress vs structural parameters.

    Args:
        sxy: Array of stress values
        structural_param: Array of structural parameter values
        struct_name: Name of structural parameter (Q4, Q6, etc.)
        corr_coef: Correlation coefficient (optional)
        p_value: P-value (optional)
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Remove NaN values
    mask = ~np.isnan(sxy) & ~np.isnan(structural_param)
    sxy_clean = sxy[mask]
    struct_clean = structural_param[mask]

    # Create scatter plot with density coloring
    # Use hexbin to show density
    hb = ax.hexbin(struct_clean, sxy_clean, gridsize=50, cmap='Blues',
                   mincnt=1)
    cb = plt.colorbar(hb, ax=ax, label='Count')

    ax.set_xlabel(f'{struct_name} (Structural Parameter)')
    ax.set_ylabel('Shear Stress sxy (bar)')
    title_text = title
    if corr_coef is not None and not np.isnan(corr_coef):
        title_text += f' (r={corr_coef:.3f}'
        if p_value is not None and not np.isnan(p_value):
            title_text += f', p={p_value:.2e})'
        title_text += ')'
    ax.set_title(title_text)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_spatial_stress_map(sxy_frame, positions, box, frame_idx=0,
                            grid_size=50, title='Spatial Stress Map'):
    """
    Spatial distribution of stress at a specific timestep.

    Args:
        sxy_frame: (n_atoms,) array of stress values
        positions: (n_atoms, 3) array of coordinates
        box: Box dimensions [Lx, Ly, Lz]
        frame_idx: Frame index for title
        grid_size: Number of grid points
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    Lx, Ly, Lz = box[:3]

    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=grid_size,
        range=[[0, Lx], [0, Ly]],
        weights=sxy_frame
    )

    # Count atoms in each bin for normalization
    H_count, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=grid_size,
        range=[[0, Lx], [0, Ly]]
    )

    # Average sxy in each bin
    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = H / H_count
        H_avg[H_count == 0] = np.nan

    im = ax.imshow(H_avg.T, origin='lower', cmap='RdBu_r',
                 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                 aspect='auto', vmin=-100, vmax=100)

    cbar = plt.colorbar(im, ax=ax, label='Shear Stress sxy (bar)')
    ax.set_xlabel('X Position (Å)')
    ax.set_ylabel('Y Position (Å)')
    ax.set_title(f'{title} (Frame {frame_idx})')

    return fig, ax


def plot_spatial_correlation(r, g_sxy, title='Spatial Stress Correlation'):
    """
    Plot spatial correlation functions of stress.

    Args:
        r: Array of distance values
        g_sxy: Array of correlation values
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(r, g_sxy, 'b-', linewidth=2)
    ax.set_xlabel('Distance r (Å)')
    ax.set_ylabel('Spatial Correlation g_sxy(r)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Highlight zero correlation line
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    return fig, ax


def plot_stress_distribution(sxy_values, bins=50, title='Stress Distribution'):
    """
    Plot histogram of stress distribution.

    Args:
        sxy_values: Array of stress values
        bins: Number of bins for histogram
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Remove NaN values
    sxy_clean = sxy_values[~np.isnan(sxy_values)]

    n, bins, patches = ax.hist(sxy_clean, bins=bins, density=True,
                                        alpha=0.7, color='steelblue', edgecolor='black')

    # Add statistics
    mean_val = np.mean(sxy_clean)
    std_val = np.std(sxy_clean)
    median_val = np.median(sxy_clean)

    ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

    ax.set_xlabel('Shear Stress sxy (bar)')
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_time_evolution_heatmap(sxy_array, coords, box, time_step=0.025,
                                grid_size=30, title='Stress Evolution Heatmap'):
    """
    Create time evolution heatmap of stress.

    Args:
        sxy_array: (n_frames, n_atoms) array of stress values
        coords: (n_frames, n_atoms, 3) array of coordinates
        box: Box dimensions [Lx, Ly, Lz]
        time_step: Time step in ps
        grid_size: Grid size for spatial binning
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    n_frames = sxy_array.shape[0]
    Lx, Ly, Lz = box[:3]

    # Downsample frames for visualization
    frames_to_plot = min(50, n_frames)
    frame_indices = np.linspace(0, n_frames - 1, frames_to_plot, dtype=int)

    # Create spatial grid for each selected frame
    grid_data = []

    for frame_idx in frame_indices:
        sxy_frame = sxy_array[frame_idx]
        pos_frame = coords[frame_idx]

        # Create 2D histogram
        H, _, _ = np.histogram2d(
            pos_frame[:, 0], pos_frame[:, 1],
            bins=grid_size,
            range=[[0, Lx], [0, Ly]],
            weights=sxy_frame
        )
        grid_data.append(H.flatten())

    grid_data = np.array(grid_data)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(grid_data, cmap='RdBu_r', aspect='auto',
                     interpolation='nearest', vmin=-50, vmax=50)

    cbar = plt.colorbar(im, ax=ax, label='Shear Stress sxy (bar)')

    ax.set_xlabel('Spatial Grid Index (flattened)')
    ax.set_ylabel('Frame Index (downsampled)')
    ax.set_title(title)

    # Add time axis labels
    time_ticks = np.linspace(0, len(frame_indices) - 1, 5, dtype=int)
    actual_times = time_ticks * time_step * (n_frames // frames_to_plot)
    ax.set_yticks(time_ticks)
    ax.set_yticklabels([f'{t:.1f}ps' for t in actual_times])

    return fig, ax


def plot_stress_autocorrelation(time_lags, autocorr, title='Stress Autocorrelation'):
    """
    Plot stress autocorrelation function.

    Args:
        time_lags: Array of time lag values (in ps)
        autocorr: Array of autocorrelation values
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Remove NaN values
    valid_mask = ~np.isnan(autocorr)
    lags_clean = time_lags[valid_mask]
    autocorr_clean = autocorr[valid_mask]

    ax.plot(lags_clean, autocorr_clean, 'b-', linewidth=2, marker='o', markersize=5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Lag (ps)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_stress_heterogeneity(time, sxy_std, window_size=100,
                           title='Stress Heterogeneity'):
    """
    Plot stress heterogeneity (std) over time with moving average.

    Args:
        time: Array of time values
        sxy_std: Array of std stress values
        window_size: Size of moving average window
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, sxy_std, 'b-', linewidth=1, alpha=0.6, label='Instantaneous')

    # Compute moving average
    if len(sxy_std) > window_size:
        moving_avg = np.convolve(sxy_std, np.ones(window_size)/window_size, mode='valid')
        ma_time = time[window_size-1:]
        ax.plot(ma_time, moving_avg, 'r-', linewidth=2, label=f'{window_size}-frame MA')

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Stress Std (bar)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def create_colormap_diverging(name='stress_diverging', min_val=-100, max_val=100):
    """
    Create a custom diverging colormap for stress visualization.

    Args:
        name: Name for colormap
        min_val: Minimum value (color for negative stress)
        max_val: Maximum value (color for positive stress)

    Returns:
        cmap: Custom colormap
    """
    # Use RdBu (red-white-blue) diverging colormap
    return plt.cm.RdBu_r


if __name__ == '__main__':
    print("Stress Visualization Module")
    print("Available functions:")
    print("  - plot_stress_heatmap()")
    print("  - plot_stress_evolution()")
    print("  - plot_stress_structure_scatter()")
    print("  - plot_spatial_stress_map()")
    print("  - plot_spatial_correlation()")
    print("  - plot_stress_distribution()")
    print("  - plot_time_evolution_heatmap()")
    print("  - plot_stress_autocorrelation()")
    print("  - plot_stress_heterogeneity()")
