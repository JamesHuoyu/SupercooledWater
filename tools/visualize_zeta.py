#!/usr/bin/env python3
"""
Visualize zeta spatial distribution and time evolution with grid mapping.

Usage:
    python visualize_zeta.py --traj traj.lammpstrj --zeta zeta.csv --out_dir output
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


class ZetaVisualizer:
    def __init__(self, traj_file, zeta_file, out_dir="output"):
        self.traj_file = traj_file
        self.zeta_file = zeta_file
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Load zeta data
        self.zeta_df = pd.read_csv(zeta_file)

        # Load trajectory
        self.u = mda.Universe(traj_file, format="LAMMPSDUMP")
        self.O_atoms = self.u.select_atoms("type 1")
        self.total_frames = len(self.u.trajectory)

        # Get available frames in zeta data
        self.available_frames = sorted(self.zeta_df["frame"].unique())
        print(f"Total frames in trajectory: {self.total_frames}")
        print(f"Frames in zeta data: {len(self.available_frames)}")

    def plot_spatial_distribution(self, frame_idx=None, save_as="zeta_spatial.png"):
        """
        Plot 3D spatial distribution of zeta values for a specific frame.
        """
        if frame_idx is None:
            # Use middle frame or first available frame
            frame_idx = self.available_frames[len(self.available_frames) // 2]

        if frame_idx not in self.available_frames:
            print(f"Frame {frame_idx} not in zeta data")
            return

        print(f"Plotting spatial distribution for frame {frame_idx}...")

        # Get zeta values for this frame
        frame_zeta = self.zeta_df[self.zeta_df["frame"] == frame_idx]
        o_indices = frame_zeta["O_idx"].values
        zeta_values = frame_zeta["zeta"].values

        # Go to the frame in trajectory
        self.u.trajectory[frame_idx]

        # Get O atom positions
        positions = self.O_atoms.positions

        # Filter positions for atoms in zeta data
        mask = np.isin(self.O_atoms.indices, o_indices)
        plot_positions = positions[mask]

        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Normalize zeta values for coloring
        norm = Normalize(vmin=np.min(zeta_values), vmax=np.max(zeta_values))
        cmap = cm.RdBu_r  # Red-Blue colormap, reversed
        colors = cmap(norm(zeta_values))

        scatter = ax.scatter(
            plot_positions[:, 0],
            plot_positions[:, 1],
            plot_positions[:, 2],
            c=zeta_values,
            cmap=cmap,
            s=20,
            alpha=0.7,
            edgecolors="none",
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label("Zeta (Å)", fontsize=12)

        # Labels
        ax.set_xlabel("X (Å)", fontsize=10)
        ax.set_ylabel("Y (Å)", fontsize=10)
        ax.set_zlabel("Z (Å)", fontsize=10)
        ax.set_title(f"Spatial Distribution of Zeta - Frame {frame_idx}", fontsize=14, pad=20)

        # Set equal aspect ratio (approximate)
        try:
            box = self.u.dimensions
            ax.set_xlim(0, box[0])
            ax.set_ylim(0, box[1])
            ax.set_zlim(0, box[2])
        except:
            pass

        plt.tight_layout()
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_zeta_evolution(self, save_as="zeta_evolution.png"):
        """
        Plot zeta evolution over time for all atoms.
        """
        print("Plotting zeta evolution over time...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Zeta Evolution Over Time", fontsize=16)

        # 1. Average zeta vs time
        ax1 = axes[0, 0]
        avg_zeta_per_frame = self.zeta_df.groupby("frame")["zeta"].mean().reset_index()
        ax1.plot(avg_zeta_per_frame["frame"], avg_zeta_per_frame["zeta"], "b-", lw=2)
        ax1.set_xlabel("Frame", fontsize=11)
        ax1.set_ylabel("Average Zeta (Å)", fontsize=11)
        ax1.set_title("Average Zeta per Frame", fontsize=12)
        ax1.grid(alpha=0.3)

        # 2. Zeta distribution over frames (boxplot or violin plot style)
        ax2 = axes[0, 1]
        # Sample frames to avoid overcrowding
        n_frames = len(self.available_frames)
        sample_indices = np.linspace(0, n_frames - 1, min(n_frames, 20), dtype=int)

        box_data = []
        box_labels = []
        for idx in sample_indices:
            frame = self.available_frames[idx]
            frame_zeta = self.zeta_df[self.zeta_df["frame"] == frame]["zeta"].values
            box_data.append(frame_zeta)
            box_labels.append(str(frame))

        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        ax2.set_xlabel("Frame", fontsize=11)
        ax2.set_ylabel("Zeta (Å)", fontsize=11)
        ax2.set_title("Zeta Distribution per Frame", fontsize=12)
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(alpha=0.3, axis="y")

        # 3. Histogram of all zeta values
        ax3 = axes[1, 0]
        all_zeta = self.zeta_df["zeta"].values
        ax3.hist(all_zeta, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
        ax3.axvline(
            np.mean(all_zeta),
            color="red",
            linestyle="--",
            lw=2,
            label=f"Mean: {np.mean(all_zeta):.3f}",
        )
        ax3.axvline(
            np.median(all_zeta),
            color="green",
            linestyle="--",
            lw=2,
            label=f"Median: {np.median(all_zeta):.3f}",
        )
        ax3.set_xlabel("Zeta (Å)", fontsize=11)
        ax3.set_ylabel("Probability Density", fontsize=11)
        ax3.set_title("Overall Zeta Distribution", fontsize=12)
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Standard deviation over time
        ax4 = axes[1, 1]
        std_zeta_per_frame = self.zeta_df.groupby("frame")["zeta"].std().reset_index()
        ax4.plot(std_zeta_per_frame["frame"], std_zeta_per_frame["zeta"], "r-", lw=2)
        ax4.set_xlabel("Frame", fontsize=11)
        ax4.set_ylabel("Zeta Standard Deviation (Å)", fontsize=11)
        ax4.set_title("Zeta Standard Deviation per Frame", fontsize=12)
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_spatial_slices(self, frame_idx=None, axis="z", n_slices=5, save_as="zeta_slices.png"):
        """
        Plot zeta distribution on 2D slices along specified axis.
        """
        if frame_idx is None:
            frame_idx = self.available_frames[len(self.available_frames) // 2]

        if frame_idx not in self.available_frames:
            print(f"Frame {frame_idx} not in zeta data")
            return

        print(f"Plotting spatial slices for frame {frame_idx}...")

        # Get zeta values for this frame
        frame_zeta = self.zeta_df[self.zeta_df["frame"] == frame_idx]
        o_indices = frame_zeta["O_idx"].values
        zeta_values = frame_zeta["zeta"].values

        # Go to the frame in trajectory
        self.u.trajectory[frame_idx]

        # Get O atom positions
        positions = self.O_atoms.positions

        # Filter positions for atoms in zeta data
        mask = np.isin(self.O_atoms.indices, o_indices)
        plot_positions = positions[mask]

        # Determine slice ranges
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        min_pos = np.min(plot_positions[:, axis_idx])
        max_pos = np.max(plot_positions[:, axis_idx])
        slice_boundaries = np.linspace(min_pos, max_pos, n_slices + 1)

        # Create figure
        fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
        if n_slices == 1:
            axes = [axes]

        # Normalize zeta values
        norm = Normalize(vmin=np.min(zeta_values), vmax=np.max(zeta_values))

        for i, (ax, (lower, upper)) in enumerate(
            zip(axes, zip(slice_boundaries[:-1], slice_boundaries[1:]))
        ):
            # Select atoms in this slice
            slice_mask = (plot_positions[:, axis_idx] >= lower) & (
                plot_positions[:, axis_idx] < upper
            )
            slice_positions = plot_positions[slice_mask]
            slice_zeta = zeta_values[slice_mask]

            # Plot other two dimensions
            other_axes = [0, 1, 2]
            other_axes.remove(axis_idx)
            x_idx, y_idx = other_axes

            scatter = ax.scatter(
                slice_positions[:, x_idx],
                slice_positions[:, y_idx],
                c=slice_zeta,
                cmap="RdBu_r",
                norm=norm,
                s=15,
                alpha=0.7,
                edgecolors="none",
            )

            ax.set_xlabel(f"{'XYZ'[x_idx]} (Å)", fontsize=9)
            ax.set_ylabel(f"{'XYZ'[y_idx]} (Å)", fontsize=9)
            ax.set_title(
                f"{axis.upper()}: {lower:.1f}-{upper:.1f} Å\n(n={len(slice_zeta)})", fontsize=10
            )
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)

        # Add colorbar to last subplot
        cbar = plt.colorbar(scatter, ax=axes[-1], fraction=0.046, pad=0.04)
        cbar.set_label("Zeta (Å)", fontsize=10)

        plt.suptitle(
            f"Zeta Distribution on {axis.upper()}-axis Slices - Frame {frame_idx}", fontsize=14
        )
        plt.tight_layout()
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_atom_zeta_timeseries(self, n_atoms=20, save_as="zeta_timeseries.png"):
        """
        Plot zeta evolution for individual atoms.
        """
        print(f"Plotting zeta timeseries for {n_atoms} atoms...")

        # Select atoms with most variation
        atom_stats = (
            self.zeta_df.groupby("O_idx")["zeta"]
            .agg(["mean", "std"])
            .sort_values("std", ascending=False)
        )
        top_atoms = atom_stats.head(n_atoms).index.tolist()

        fig, ax = plt.subplots(figsize=(15, 6))

        colors = cm.tab20(np.linspace(0, 1, n_atoms))

        for atom_id, color in zip(top_atoms, colors):
            atom_data = self.zeta_df[self.zeta_df["O_idx"] == atom_id]
            ax.plot(
                atom_data["frame"],
                atom_data["zeta"],
                marker="o",
                markersize=3,
                linestyle="-",
                alpha=0.7,
                linewidth=1,
                color=color,
                label=f"Atom {atom_id}",
            )

        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Zeta (Å)", fontsize=12)
        ax.set_title(f"Zeta Evolution for Top {n_atoms} Most Variable Atoms", fontsize=14)
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
            ncol=2,
        )
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def map_to_grid(self, frame_idx, grid_resolution=(50, 50, 50), method="linear"):
        """
        Map zeta values to a 3D grid using interpolation.

        Parameters:
        -----------
        frame_idx : int
            Frame index
        grid_resolution : tuple of 3 ints
            Grid resolution (nx, ny, nz)
        method : str
            Interpolation method: 'linear', 'nearest', or 'cubic'

        Returns:
        --------
        grid_zeta : 3D numpy array
            Zeta values on the grid
        grid_coords : tuple
            (X, Y, Z) meshgrid coordinates
        """
        print(f"Mapping zeta to {grid_resolution} grid for frame {frame_idx}...")

        # Get zeta values for this frame
        frame_zeta = self.zeta_df[self.zeta_df["frame"] == frame_idx]
        o_indices = frame_zeta["O_idx"].values
        zeta_values = frame_zeta["zeta"].values

        # Go to the frame in trajectory
        self.u.trajectory[frame_idx]

        # Get O atom positions
        positions = self.O_atoms.positions

        # Filter positions for atoms in zeta data
        mask = np.isin(self.O_atoms.indices, o_indices)
        plot_positions = positions[mask]

        # Create regular grid
        nx, ny, nz = grid_resolution

        # Get box dimensions from trajectory(using apply_PBC)
        box = self.u.dimensions
        plot_positions = mda.lib.distances.apply_PBC(plot_positions, box)
        # x_min, x_max = 0, box[0]
        # y_min, y_max = 0, box[1]
        # z_min, z_max = 0, box[2]
        x_min, x_max = np.min(plot_positions[:, 0]), np.max(plot_positions[:, 0])
        y_min, y_max = np.min(plot_positions[:, 1]), np.max(plot_positions[:, 1])
        z_min, z_max = np.min(plot_positions[:, 2]), np.max(plot_positions[:, 2])

        # Create grid points
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        zi = np.linspace(z_min, z_max, nz)
        Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing="ij")

        # Flatten grid points for interpolation
        grid_points = np.column_stack([Xi.ravel(), Yi.ravel(), Zi.ravel()])

        # Interpolate zeta values onto grid
        print("Interpolating zeta values to grid...")
        grid_zeta_flat = griddata(plot_positions, zeta_values, grid_points, method=method)

        # Reshape back to 3D grid
        grid_zeta = grid_zeta_flat.reshape(nx, ny, nz)

        # Handle NaN values (fill with nearest neighbor)
        # if np.any(np.isnan(grid_zeta)):
        #     print("Handling NaN values with nearest neighbor interpolation...")
        #     tree = cKDTree(plot_positions)
        #     for i in range(nx):
        #         for j in range(ny):
        #             for k in range(nz):
        #                 if np.isnan(grid_zeta[i, j, k]):
        #                     dist, idx = tree.query([Xi[i, j, k], Yi[i, j, k], Zi[i, j, k]], k=1)
        #                     grid_zeta[i, j, k] = zeta_values[idx]

        print("Grid mapping complete!")
        return grid_zeta, (Xi, Yi, Zi)

    def plot_grid_slices(
        self,
        frame_idx=None,
        grid_resolution=(50, 50, 50),
        axis="z",
        n_slices=3,
        method="linear",
        save_as="zeta_grid_slices.png",
    ):
        """
        Plot zeta distribution on grid slices.
        """
        if frame_idx is None:
            frame_idx = self.available_frames[len(self.available_frames) // 2]

        print(f"Plotting grid slices for frame {frame_idx}...")

        # Map to grid
        grid_zeta, (Xi, Yi, Zi) = self.map_to_grid(frame_idx, grid_resolution, method)

        # Get slice indices
        nx, ny, nz = grid_resolution
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        slice_indices = np.linspace(0, grid_resolution[axis_idx] - 1, n_slices, dtype=int)

        # Create figure
        fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
        if n_slices == 1:
            axes = [axes]

        # Get coordinates based on axis
        if axis.lower() == "x":
            coords = [xi for xi in np.linspace(0, self.u.dimensions[0], nx)]
            coord_name = "X"
            x_grid, y_grid = Yi[:, 0, :], Zi[:, 0, :]
        elif axis.lower() == "y":
            coords = [yi for yi in np.linspace(0, self.u.dimensions[1], ny)]
            coord_name = "Y"
            x_grid, y_grid = Xi[0, :, :], Zi[0, :, :]
        else:  # z
            coords = [zi for zi in np.linspace(0, self.u.dimensions[2], nz)]
            coord_name = "Z"
            x_grid, y_grid = Xi[:, :, 0], Yi[:, :, 0]

        # Determine color limits based on full grid
        v_min = np.nanmin(grid_zeta)
        v_max = np.nanmax(grid_zeta)

        for ax, slice_idx in zip(axes, slice_indices):
            # Extract slice
            if axis.lower() == "x":
                slice_data = grid_zeta[slice_idx, :, :]
                slice_coord = coords[slice_idx]
                x_plot, y_plot = Yi[slice_idx, :, :], Zi[slice_idx, :, :]
            elif axis.lower() == "y":
                slice_data = grid_zeta[:, slice_idx, :]
                slice_coord = coords[slice_idx]
                x_plot, y_plot = Xi[:, slice_idx, :], Zi[:, slice_idx, :]
            else:  # z
                slice_data = grid_zeta[:, :, slice_idx]
                slice_coord = coords[slice_idx]
                x_plot, y_plot = Xi[:, :, slice_idx], Yi[:, :, slice_idx]

            # Plot with imshow
            im = ax.imshow(
                slice_data.T,  # Transpose to match orientation
                origin="lower",
                extent=[x_plot.min(), x_plot.max(), y_plot.min(), y_plot.max()],
                cmap="RdBu_r",
                vmin=v_min,
                vmax=v_max,
                aspect="auto",
                interpolation="bilinear",
            )

            ax.set_xlabel(f"{'Y' if axis.lower() == 'x' else 'X'} (Å)", fontsize=9)
            ax.set_ylabel(f"{'Z' if axis.lower() in ['x', 'y'] else 'Y'} (Å)", fontsize=9)
            ax.set_title(f"{coord_name} = {slice_coord:.1f} Å", fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        cbar.set_label("Zeta (Å)", fontsize=10)

        plt.suptitle(
            f"Zeta Grid Slices ({coord_name}-axis) - Frame {frame_idx}\nGrid: {grid_resolution}",
            fontsize=14,
        )
        plt.tight_layout()
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_grid_3d_isosurface(
        self,
        frame_idx=None,
        grid_resolution=(30, 30, 30),
        method="linear",
        n_levels=3,
        save_as="zeta_grid_isosurface.png",
    ):
        """
        Plot 3D isosurface of zeta distribution on grid.
        """
        if frame_idx is None:
            frame_idx = self.available_frames[len(self.available_frames) // 2]

        print(f"Plotting 3D isosurface for frame {frame_idx}...")

        # Map to grid
        grid_zeta, (Xi, Yi, Zi) = self.map_to_grid(frame_idx, grid_resolution, method)

        # Determine isosurface levels based on percentiles
        v_min, v_max = np.nanmin(grid_zeta), np.nanmax(grid_zeta)
        levels = np.linspace(v_min, v_max, n_levels + 2)[1:-1]  # Exclude extremes

        # Create figure
        fig = plt.figure(figsize=(14, 6))

        # Plot 1: 3D scatter plot of high zeta regions
        ax1 = fig.add_subplot(121, projection="3d")

        # Find points with zeta above a threshold (e.g., 75th percentile)
        threshold = np.nanpercentile(grid_zeta, 75)
        mask = grid_zeta > threshold

        x_high, y_high, z_high = Xi[mask], Yi[mask], Zi[mask]
        values_high = grid_zeta[mask]

        if len(x_high) > 0:
            # Downsample for visualization if too many points
            max_points = 5000
            if len(x_high) > max_points:
                indices = np.random.choice(len(x_high), max_points, replace=False)
                x_high = x_high[indices]
                y_high = y_high[indices]
                z_high = z_high[indices]
                values_high = values_high[indices]

            norm = Normalize(vmin=np.nanmin(grid_zeta), vmax=np.nanmax(grid_zeta))
            scatter = ax1.scatter(
                x_high, y_high, z_high, c=values_high, cmap="RdBu_r", norm=norm, s=10, alpha=0.5
            )

            cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.7, pad=0.1)
            cbar1.set_label("Zeta (Å)", fontsize=9)

        ax1.set_xlabel("X (Å)", fontsize=9)
        ax1.set_ylabel("Y (Å)", fontsize=9)
        ax1.set_zlabel("Z (Å)", fontsize=9)
        ax1.set_title(f"3D Distribution\n(Zeta > {threshold:.3f} Å)", fontsize=11)

        # Plot 2: Multiple 2D slices at different Z positions
        ax2 = fig.add_subplot(122)
        nz = grid_resolution[2]
        slice_indices = np.linspace(0, nz - 1, 4, dtype=int)
        z_coords = [zi for zi in np.linspace(0, self.u.dimensions[2], nz)]

        # Stack slices vertically
        slices = []
        for i, slice_idx in enumerate(slice_indices):
            slice_data = grid_zeta[:, :, slice_idx]
            slices.append(slice_data)
            if i > 0:
                # Add blank row between slices
                slices.append(np.full((1, grid_resolution[0]), np.nan))

        stacked_slices = np.vstack(slices)

        im = ax2.imshow(
            stacked_slices.T,
            origin="lower",
            cmap="RdBu_r",
            aspect="auto",
            interpolation="bilinear",
            extent=[0, self.u.dimensions[0], 0, self.u.dimensions[1] * 4],
        )

        ax2.set_xlabel("X (Å)", fontsize=9)
        ax2.set_ylabel("Y (Å)", fontsize=9)
        ax2.set_title("Z-Slices", fontsize=11)

        # Add Z position labels
        for i, slice_idx in enumerate(slice_indices):
            y_pos = (i * (grid_resolution[1] + 1) + grid_resolution[1] / 2) / (
                4 * (grid_resolution[1] + 1)
            )
            y_label = y_pos * self.u.dimensions[1] * 4
            ax2.text(
                -2,
                y_label,
                f"Z={z_coords[slice_idx]:.1f}Å",
                ha="right",
                va="center",
                fontsize=8,
                rotation=90,
            )

        cbar2 = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Zeta (Å)", fontsize=9)

        plt.suptitle(
            f"Zeta 3D Grid Visualization - Frame {frame_idx}\nGrid: {grid_resolution}", fontsize=14
        )
        plt.tight_layout()
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_grid_heatmap(
        self,
        frame_idx=None,
        grid_resolution=(50, 50),
        projection_axis="z",
        method="linear",
        save_as="zeta_grid_heatmap.png",
    ):
        """
        Project zeta onto a 2D grid (average or sum along one axis).
        """
        if frame_idx is None:
            frame_idx = self.available_frames[len(self.available_frames) // 2]

        print(f"Plotting 2D grid heatmap for frame {frame_idx}...")

        # Use higher resolution for the projected axes
        if projection_axis.lower() == "z":
            grid_res = (grid_resolution[0], grid_resolution[1], 20)
        elif projection_axis.lower() == "y":
            grid_res = (grid_resolution[0], 20, grid_resolution[1])
        else:  # x
            grid_res = (20, grid_resolution[0], grid_resolution[1])

        # Map to 3D grid
        grid_zeta, (Xi, Yi, Zi) = self.map_to_grid(frame_idx, grid_res, method)

        # Project onto 2D by averaging along one axis
        if projection_axis.lower() == "z":
            # Average along Z
            grid_2d = np.nanmean(grid_zeta, axis=2)
            extent = [0, self.u.dimensions[0], 0, self.u.dimensions[1]]
            xlabel, ylabel = "X (Å)", "Y (Å)"
            title = "XY Projection (averaged over Z)"
        elif projection_axis.lower() == "y":
            # Average along Y
            grid_2d = np.nanmean(grid_zeta, axis=1)
            extent = [0, self.u.dimensions[0], 0, self.u.dimensions[2]]
            xlabel, ylabel = "X (Å)", "Z (Å)"
            title = "XZ Projection (averaged over Y)"
        else:  # x
            # Average along X
            grid_2d = np.nanmean(grid_zeta, axis=0)
            extent = [0, self.u.dimensions[1], 0, self.u.dimensions[2]]
            xlabel, ylabel = "Y (Å)", "Z (Å)"
            title = "YZ Projection (averaged over X)"

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            grid_2d.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="auto",
            interpolation="bilinear",
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title}\nFrame {frame_idx}", fontsize=14)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Average Zeta (Å)", fontsize=12)

        # Add statistics text
        stats_text = f"Min: {np.nanmin(grid_2d):.3f} Å\nMax: {np.nanmax(grid_2d):.3f} Å\nMean: {np.nanmean(grid_2d):.3f} Å"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_grid_evolution(
        self,
        grid_resolution=(30, 30, 30),
        n_frames=10,
        projection_axis="z",
        method="linear",
        save_as="zeta_grid_evolution.png",
    ):
        """
        Plot zeta grid evolution over time as 2D heatmaps.
        """
        # Select frames evenly distributed
        n_total = len(self.available_frames)
        frame_indices = self.available_frames[
            np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)
        ]

        print(f"Plotting grid evolution for {len(frame_indices)} frames...")

        # Calculate number of rows and columns
        n_cols = min(5, len(frame_indices))
        n_rows = (len(frame_indices) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        # Global color limits
        all_grids = []
        for frame_idx in frame_indices:
            # Use appropriate grid resolution based on projection
            if projection_axis.lower() == "z":
                grid_res = (grid_resolution[0], grid_resolution[1], 10)
            elif projection_axis.lower() == "y":
                grid_res = (grid_resolution[0], 10, grid_resolution[1])
            else:  # x
                grid_res = (10, grid_resolution[0], grid_resolution[1])

            grid_zeta, _ = self.map_to_grid(frame_idx, grid_res, method)

            if projection_axis.lower() == "z":
                grid_2d = np.nanmean(grid_zeta, axis=2)
            elif projection_axis.lower() == "y":
                grid_2d = np.nanmean(grid_zeta, axis=1)
            else:  # x
                grid_2d = np.nanmean(grid_zeta, axis=0)

            all_grids.append(grid_2d)

        v_min = min(np.nanmin(g) for g in all_grids)
        v_max = max(np.nanmax(g) for g in all_grids)

        # Plot each frame
        extent_dict = {
            "z": [0, self.u.dimensions[0], 0, self.u.dimensions[1]],
            "y": [0, self.u.dimensions[0], 0, self.u.dimensions[2]],
            "x": [0, self.u.dimensions[1], 0, self.u.dimensions[2]],
        }
        xlabel_dict = {"z": "X (Å)", "y": "X (Å)", "x": "Y (Å)"}
        ylabel_dict = {"z": "Y (Å)", "y": "Z (Å)", "x": "Z (Å)"}

        for i, (frame_idx, grid_2d) in enumerate(zip(frame_indices, all_grids)):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            im = ax.imshow(
                grid_2d.T,
                origin="lower",
                extent=extent_dict[projection_axis.lower()],
                cmap="RdBu_r",
                aspect="auto",
                interpolation="bilinear",
                vmin=v_min,
                vmax=v_max,
            )

            ax.set_xlabel(xlabel_dict[projection_axis.lower()], fontsize=8)
            ax.set_ylabel(ylabel_dict[projection_axis.lower()], fontsize=8)
            ax.set_title(f"Frame {frame_idx}", fontsize=10)

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Average Zeta (Å)", fontsize=10)

        plt.suptitle(
            f"Zeta Grid Evolution ({projection_axis.upper()}-projection)\nGrid: {grid_resolution}",
            fontsize=14,
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])
        output_path = os.path.join(self.out_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def generate_all_plots(self, use_grid=False, grid_resolution=(50, 50, 50)):
        """
        Generate all visualization plots.

        Parameters:
        -----------
        use_grid : bool
            Whether to use grid-based visualization (default: False for scatter plots)
        grid_resolution : tuple
            Grid resolution (nx, ny, nz) for grid-based plots
        """
        print("\n" + "=" * 60)
        print("Generating Zeta Visualization Plots")
        print("=" * 60 + "\n")

        if use_grid:
            # Grid-based visualizations
            print("Using GRID-based visualizations\n")
            # 1. Grid slices
            self.plot_grid_slices(grid_resolution=grid_resolution, save_as="zeta_grid_slices_z.png")
            # # 2. Grid heatmap (XY projection)
            # self.plot_grid_heatmap(grid_resolution=grid_resolution[:2], projection_axis="z",
            #                        save_as="zeta_grid_heatmap_xy.png")
            # # 3. Grid heatmap (XZ projection)
            # self.plot_grid_heatmap(grid_resolution=(grid_resolution[0], grid_resolution[2]),
            #                        projection_axis="y", save_as="zeta_grid_heatmap_xz.png")
            # # 4. Grid heatmap (YZ projection)
            # self.plot_grid_heatmap(grid_resolution=(grid_resolution[1], grid_resolution[2]),
            #                        projection_axis="x", save_as="zeta_grid_heatmap_yz.png")
            # # 5. 3D isosurface
            # self.plot_grid_3d_isosurface(grid_resolution=(min(30, grid_resolution[0]),
            #                                              min(30, grid_resolution[1]),
            #                                              min(30, grid_resolution[2])),
            #                              save_as="zeta_grid_isosurface.png")
            # # 6. Grid evolution
            # self.plot_grid_evolution(grid_resolution=grid_resolution, n_frames=8, projection_axis="z",
            #                         save_as="zeta_grid_evolution.png")
        else:
            # Scatter-based visualizations
            print("Using SCATTER-based visualizations\n")
            # 1. 3D spatial distribution
            self.plot_spatial_distribution(save_as="zeta_spatial_3d.png")

            # 2. Time evolution analysis
            self.plot_zeta_evolution(save_as="zeta_evolution.png")

            # 3. Spatial slices (along z-axis)
            self.plot_spatial_slices(axis="z", n_slices=5, save_as="zeta_slices_z.png")

            # 4. Individual atom timeseries
            self.plot_atom_zeta_timeseries(n_atoms=30, save_as="zeta_timeseries.png")

        print("\n" + "=" * 60)
        print("All plots generated successfully!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize zeta spatial distribution and evolution with grid mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scatter-based visualization (default)
  python visualize_zeta.py --traj traj.lammpstrj --zeta zeta.csv

  # Grid-based visualization
  python visualize_zeta.py --traj traj.lammpstrj --zeta zeta.csv --use_grid --grid_res 50 50 50

  # Specific frame with grid visualization
  python visualize_zeta.py --traj traj.lammpstrj --zeta zeta.csv --use_grid --frame 100 --grid_res 40 40 40

  # Only evolution plots
  python visualize_zeta.py --traj traj.lammpstrj --zeta zeta.csv --evolution_only
        """,
    )
    parser.add_argument("--traj", required=True, help="LAMMPS trajectory file (.lammpstrj)")
    parser.add_argument("--zeta", required=True, help="Zeta CSV file")
    parser.add_argument("--out_dir", default="output", help="Output directory")
    parser.add_argument(
        "--frame", type=int, help="Specific frame for spatial plots (default: middle frame)"
    )

    # Grid visualization options
    parser.add_argument(
        "--use_grid",
        action="store_true",
        help="Use grid-based interpolation instead of scatter plots",
    )
    parser.add_argument(
        "--grid_res",
        nargs=3,
        type=int,
        default=[50, 50, 50],
        metavar=("NX", "NY", "NZ"),
        help="Grid resolution for interpolation (default: 50 50 50)",
    )
    parser.add_argument(
        "--method",
        choices=["linear", "nearest", "cubic"],
        default="linear",
        help="Interpolation method for grid (default: linear)",
    )

    # Plot type options
    parser.add_argument(
        "--evolution_only", action="store_true", help="Only generate time evolution plots"
    )
    parser.add_argument(
        "--spatial_only", action="store_true", help="Only generate spatial distribution plots"
    )

    args = parser.parse_args()

    viz = ZetaVisualizer(args.traj, args.zeta, args.out_dir)

    grid_res = tuple(args.grid_res)

    if args.evolution_only:
        # Only evolution plots
        viz.plot_zeta_evolution(save_as="zeta_evolution.png")
        viz.plot_atom_zeta_timeseries(n_atoms=30, save_as="zeta_timeseries.png")
        if args.use_grid:
            viz.plot_grid_evolution(
                grid_resolution=grid_res,
                n_frames=8,
                projection_axis="z",
                method=args.method,
                save_as="zeta_grid_evolution.png",
            )
    elif args.spatial_only:
        # Only spatial plots
        if args.use_grid:
            viz.plot_grid_slices(
                grid_resolution=grid_res,
                axis="z",
                n_slices=5,
                method=args.method,
                save_as="zeta_grid_slices_z.png",
            )
            viz.plot_grid_heatmap(
                grid_resolution=grid_res[:2],
                projection_axis="z",
                method=args.method,
                save_as="zeta_grid_heatmap_xy.png",
            )
            viz.plot_grid_heatmap(
                grid_resolution=(grid_res[0], grid_res[2]),
                projection_axis="y",
                method=args.method,
                save_as="zeta_grid_heatmap_xz.png",
            )
            viz.plot_grid_heatmap(
                grid_resolution=(grid_res[1], grid_res[2]),
                projection_axis="x",
                method=args.method,
                save_as="zeta_grid_heatmap_yz.png",
            )
            viz.plot_grid_3d_isosurface(
                grid_resolution=(min(30, grid_res[0]), min(30, grid_res[1]), min(30, grid_res[2])),
                method=args.method,
                save_as="zeta_grid_isosurface.png",
            )
        else:
            viz.plot_spatial_distribution(save_as="zeta_spatial_3d.png")
            viz.plot_spatial_slices(axis="z", n_slices=5, save_as="zeta_slices_z.png")
    else:
        # Generate all plots
        viz.generate_all_plots(use_grid=args.use_grid, grid_resolution=grid_res)

    # If specific frame requested, also plot it
    if args.frame is not None:
        if args.use_grid:
            viz.plot_grid_slices(
                frame_idx=args.frame,
                grid_resolution=grid_res,
                axis="z",
                n_slices=5,
                method=args.method,
                save_as=f"zeta_grid_slices_frame_{args.frame}.png",
            )
            viz.plot_grid_heatmap(
                frame_idx=args.frame,
                grid_resolution=grid_res[:2],
                projection_axis="z",
                method=args.method,
                save_as=f"zeta_grid_heatmap_frame_{args.frame}.png",
            )
        else:
            viz.plot_spatial_distribution(
                frame_idx=args.frame, save_as=f"zeta_spatial_frame_{args.frame}.png"
            )
            viz.plot_spatial_slices(
                frame_idx=args.frame,
                axis="z",
                n_slices=5,
                save_as=f"zeta_slices_frame_{args.frame}.png",
            )


if __name__ == "__main__":
    main()
