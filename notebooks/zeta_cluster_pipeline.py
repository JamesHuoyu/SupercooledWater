"""
Zeta Cluster Analysis – Full Pipeline
=======================================

Covers
------
  0.  Run the full stack (HBA → ZOP → ZCA)
  1.  Spatial slice: snapshot coloured by ζ, cluster boundaries overlaid
  2.  Spatial slice animation: PNG sequence showing particle motion
  3.  Cluster-size time series (total molecules in tet / distorted clusters)
  4.  Cluster lifetime distribution
  5.  Cluster size distribution P(N)
  6.  Cluster radius of gyration vs size  ⟨Rg⟩(N)
  7.  Centre-of-mass trajectories of the largest clusters
  8.  CoM mean-squared displacement (cluster-level MSD)
  9.  Cluster ζ-distribution over time (heatmap)
  10. Numerical summary table + CSV export
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
import MDAnalysis as mda

from water.tools.custom_hbond_analysis import HydrogenBondAnalysis as HBA
from water.tools.zeta_order_parameter  import ZetaOrderParameter  as ZOP
from water.tools.zeta_cluster_analysis import ZetaClusterAnalysis  as ZCA

os.makedirs("cluster_plots", exist_ok=True)

# ============================================================================
# 0.  Run the full stack
# ============================================================================

u = mda.Universe(
    "/root/water/TIP4P/Ice/test/tip4p-ice-225K.data",
    "/root/water/TIP4P/Ice/225/dump_225_test.lammpstrj",
    format="LAMMPSDUMP",
    dt=0.2,
)

hba = HBA(
    universe=u,
    donors_sel="type 1",
    hydrogens_sel="type 2",
    acceptors_sel="type 1",
    d_a_cutoff=3.5,
    h_d_a_angle_cutoff=30.0,
    update_selections=False,
)
hba.run(verbose=True)

zop = ZOP(hba=hba, central_sel="type 1", shell_cutoff=6.0)
zop.run(verbose=True)

zca = ZCA(
    zop=zop,
    eps=3.5,            # Å – first O-O shell for ice ≈ 2.75 Å, use 3.5 for some slack
    min_samples=4,      # minimum cluster size (central molecule + 3 neighbours)
    zeta_threshold=0.0,
    min_track_overlap=0.3,
    min_track_lifetime=2,
    cluster_classes=(1, -1),
)
zca.run(verbose=True)

# Convenience aliases
central_indices = zca.results.central_indices   # OW atom indices
times_ps        = zca.results.times             # shape (n_frames,)
n_frames        = len(zca.frames)
dt              = u.trajectory.dt               # ps per frame

# ============================================================================
# Helper: thin spatial slab selector
# ============================================================================

def get_slab_mask(positions, box_lengths, axis=2, centre=None, width=5.0):
    """Return boolean mask for atoms within a slab of given width (Å).

    Parameters
    ----------
    positions   : np.ndarray, shape (N, 3)
    box_lengths : array-like, shape (3,) – box lengths [lx, ly, lz]
    axis        : 0=x, 1=y, 2=z
    centre      : float or None – slab midpoint; None → box centre
    width       : float – full slab thickness in Å
    """
    L   = box_lengths[axis]
    mid = L / 2.0 if centre is None else centre
    z   = positions[:, axis]
    # Minimum-image along the slab axis
    dz  = z - mid
    dz -= L * np.round(dz / L)
    return np.abs(dz) <= width / 2.0

# ============================================================================
# 1.  Spatial slice snapshot coloured by ζ, cluster convex-hull overlaid
# ============================================================================

def plot_spatial_slice(frame_idx, axis=2, width=5.0, ax=None, show=True):
    """
    Parameters
    ----------
    frame_idx : index into zca.frames (not the trajectory frame number)
    axis      : 0/1/2 → x/y/z as slab normal
    width     : slab thickness (Å)
    """
    frame  = zca.frames[frame_idx]
    u.trajectory[frame]
    box    = u.trajectory.ts.dimensions
    box_L  = box[:3]

    # Positions and ζ for all central (OW) atoms
    positions = zca._central_ag.positions.copy()   # (N, 3)
    zop_row   = zop._frame_to_row[frame]
    zeta_vals = zop.results.zeta[zop_row].copy()

    # Structural labels at this frame
    labels    = zca.results.frame_labels[frame_idx]
    cluster_ids = zca.results.frame_cluster_ids[frame_idx]

    # Slab mask
    slab = get_slab_mask(positions, box_L, axis=axis, width=width)
    idx_in_slab   = np.where(slab)[0]

    if len(idx_in_slab) == 0:
        print(f"No atoms in slab at frame {frame}.")
        return

    ax_x, ax_y  = [i for i in range(3) if i != axis]
    pos_slab    = positions[idx_in_slab]
    zeta_slab   = zeta_vals[idx_in_slab]
    gid_slab    = cluster_ids[idx_in_slab]
    label_slab  = labels[idx_in_slab]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
        created_fig = True
    else:
        created_fig = False

    # ---- Scatter coloured by ζ -------------------------------------------
    norm = mcolors.TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.5)
    sc   = ax.scatter(
        pos_slab[:, ax_x], pos_slab[:, ax_y],
        c=zeta_slab, cmap="RdBu", norm=norm,
        s=14, linewidths=0, alpha=0.85, zorder=2,
    )

    # ---- Convex hull outlines for each tracked cluster -------------------
    unique_gids = np.unique(gid_slab[gid_slab >= 0])
    hull_colors = {1: "navy", -1: "darkred"}

    for gid in unique_gids:
        gid_mask  = gid_slab == gid
        pts       = pos_slab[gid_mask][:, [ax_x, ax_y]]
        if len(pts) < 3:
            continue
        cls       = label_slab[gid_mask][0]
        try:
            hull  = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            ax.plot(pts[verts, 0], pts[verts, 1],
                    color=hull_colors.get(cls, "grey"),
                    lw=1.2, alpha=0.7, zorder=3)
            # Annotate with global cluster ID
            cx, cy = pts.mean(axis=0)
            ax.text(cx, cy, str(gid), fontsize=6,
                    ha="center", va="center",
                    color=hull_colors.get(cls, "grey"), zorder=4)
        except Exception:
            pass

    plt.colorbar(sc, ax=ax, label="ζ (Å)", shrink=0.8)

    axis_labels = ["x", "y", "z"]
    ax.set_xlabel(f"{axis_labels[ax_x]} (Å)", fontsize=11)
    ax.set_ylabel(f"{axis_labels[ax_y]} (Å)", fontsize=11)
    ax.set_title(
        f"Spatial slice  |  frame {frame}  ({frame * dt:.1f} ps)  "
        f"|  slab {axis_labels[axis]} ± {width/2:.1f} Å",
        fontsize=10,
    )
    legend_handles = [
        Patch(edgecolor="navy",    facecolor="none", lw=1.5, label="Tetrahedral cluster (ζ>0)"),
        Patch(edgecolor="darkred", facecolor="none", lw=1.5, label="Distorted cluster  (ζ≤0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=6, label="ζ > 0  (blue)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=6, label="ζ ≤ 0  (red)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_aspect("equal")

    if created_fig and show:
        plt.tight_layout()
        plt.savefig(f"cluster_plots/spatial_slice_frame{frame_idx:04d}.png",
                    dpi=150)
        plt.close()


# Plot snapshot at the first, middle, and last frames
for fi in [0, n_frames // 2, n_frames - 1]:
    plot_spatial_slice(fi, axis=2, width=5.0, show=True)
    print(f"Slice written: cluster_plots/spatial_slice_frame{fi:04d}.png")

# ============================================================================
# 2.  Spatial slice animation – PNG sequence (every 10 frames)
# ============================================================================

print("\nWriting animation frames …")
for fi in range(0, n_frames, max(1, n_frames // 50)):
    plot_spatial_slice(fi, axis=2, width=5.0, show=True)
print("Animation frames written to cluster_plots/")

# ============================================================================
# 3.  Cluster-size time series (total molecules in tet / distorted clusters)
# ============================================================================

times_tet, count_tet = zca.get_cluster_size_timeseries(label=+1)
times_dis, count_dis = zca.get_cluster_size_timeseries(label=-1)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(times_tet, count_tet, color="steelblue", lw=1.5,
        label="Tetrahedral (ζ>0)")
ax.plot(times_dis, count_dis, color="tomato",    lw=1.5,
        label="Distorted  (ζ≤0)")
ax.set(xlabel="Time (ps)", ylabel="Molecules in clusters",
       title="Total cluster population vs. time")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("cluster_plots/cluster_population_timeseries.png", dpi=150)
plt.close()

# ============================================================================
# 4.  Cluster lifetime distribution
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, cls, name, color in zip(
    axes, [+1, -1], ["Tetrahedral", "Distorted"], ["steelblue", "tomato"]
):
    tracks = [t for t in zca.results.tracks.values() if t.label == cls]
    if not tracks:
        ax.text(0.5, 0.5, "No tracks", ha="center", transform=ax.transAxes)
        continue
    lifetimes_ps = np.array([t.lifetime for t in tracks]) * dt
    ax.hist(lifetimes_ps, bins=30, color=color, edgecolor="white", density=True)
    ax.axvline(lifetimes_ps.mean(), color="black", lw=1.2, ls="--",
               label=f"Mean = {lifetimes_ps.mean():.1f} ps")
    ax.set(xlabel="Lifetime (ps)", ylabel="Probability density",
           title=f"{name} cluster lifetimes")
    ax.legend(fontsize=9)
    print(f"\n[4] {name} clusters:  "
          f"n={len(tracks)},  "
          f"mean lifetime={lifetimes_ps.mean():.2f} ps,  "
          f"max={lifetimes_ps.max():.2f} ps")

plt.tight_layout()
plt.savefig("cluster_plots/cluster_lifetime_distribution.png", dpi=150)
plt.close()

# ============================================================================
# 5.  Cluster size distribution P(N)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, cls, name, color in zip(
    axes, [+1, -1], ["Tetrahedral", "Distorted"], ["steelblue", "tomato"]
):
    tracks = [t for t in zca.results.tracks.values() if t.label == cls]
    if not tracks:
        continue
    all_sizes = np.concatenate([t.size for t in tracks])
    ax.hist(all_sizes, bins=np.arange(1, all_sizes.max() + 2) - 0.5,
            color=color, edgecolor="white", density=True)
    ax.set(xlabel="Cluster size N (molecules)", ylabel="P(N)",
           title=f"P(N) – {name}")
    # Log-scale x when range spans orders of magnitude
    if all_sizes.max() / max(all_sizes.min(), 1) > 20:
        ax.set_xscale("log")
    print(f"[5] {name}  mean N={all_sizes.mean():.1f},  "
          f"max N={all_sizes.max()}")

plt.tight_layout()
plt.savefig("cluster_plots/cluster_size_distribution.png", dpi=150)
plt.close()

# ============================================================================
# 6.  Radius of gyration vs cluster size  ⟨Rg⟩(N)
# ============================================================================

fig, ax = plt.subplots(figsize=(6, 5))
for cls, name, color, marker in [
    (+1, "Tetrahedral", "steelblue", "o"),
    (-1, "Distorted",   "tomato",    "s"),
]:
    tracks = [t for t in zca.results.tracks.values() if t.label == cls]
    if not tracks:
        continue
    sizes = np.concatenate([t.size for t in tracks])
    rgs   = np.concatenate([t.rg   for t in tracks])
    valid = ~np.isnan(rgs)
    ax.scatter(sizes[valid], rgs[valid], s=6, alpha=0.3,
               color=color, label=name, marker=marker)
    # Bin-mean
    size_bins = np.arange(1, sizes.max() + 2)
    bin_mean_rg = []
    bin_centres = []
    for n in size_bins:
        mask = (sizes[valid] == n)
        if mask.sum() > 0:
            bin_mean_rg.append(rgs[valid][mask].mean())
            bin_centres.append(n)
    if bin_centres:
        ax.plot(bin_centres, bin_mean_rg, color=color, lw=2.0)

ax.set(xlabel="Cluster size N",
       ylabel="Radius of gyration Rg (Å)",
       title="⟨Rg⟩ vs cluster size")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("cluster_plots/rg_vs_size.png", dpi=150)
plt.close()

# ============================================================================
# 7.  CoM trajectories of the N_SHOW largest clusters
# ============================================================================

N_SHOW = 6   # How many tracks to show per class

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axis_pairs = [(0, 1), (0, 1)]   # x-y projection

for ax, cls, name in zip(axes, [+1, -1], ["Tetrahedral", "Distorted"]):
    tracks = sorted(
        [t for t in zca.results.tracks.values() if t.label == cls],
        key=lambda t: t.mean_size,
        reverse=True,
    )[:N_SHOW]

    cmap = cm.get_cmap("tab10", len(tracks))
    for i, track in enumerate(tracks):
        ax_x, ax_y = axis_pairs[0]
        com = track.com
        t_ps = track.frames * dt
        sc = ax.scatter(com[:, ax_x], com[:, ax_y],
                        c=t_ps, cmap="plasma", s=15, alpha=0.8,
                        label=f"ID={track.global_id} (N̄={track.mean_size:.0f})")
        # Arrow showing direction
        if len(com) >= 2:
            ax.annotate("",
                xy=(com[-1, ax_x], com[-1, ax_y]),
                xytext=(com[-2, ax_x], com[-2, ax_y]),
                arrowprops=dict(arrowstyle="->", color=cmap(i), lw=1.2),
            )

    ax.set(xlabel="x (Å)", ylabel="y (Å)",
           title=f"CoM trajectories – {name} clusters (top {N_SHOW} by size)")
    ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
plt.savefig("cluster_plots/com_trajectories.png", dpi=150)
plt.close()

# ============================================================================
# 8.  Cluster-level MSD: ⟨|ΔCoM(τ)|²⟩ vs lag time
# ============================================================================

def compute_cluster_msd(tracks, max_tau_frames):
    """Ensemble-average MSD of cluster CoM over lag times 1…max_tau_frames."""
    tau_vals = np.arange(1, max_tau_frames + 1)
    msd      = np.zeros(len(tau_vals))
    counts   = np.zeros(len(tau_vals), dtype=int)

    # Build a lookup: global_id → {frame: CoM}
    for track in tracks:
        if len(track.frames) < 2:
            continue
        frame_com = {int(f): track.com[i] for i, f in enumerate(track.frames)}
        frame_arr = sorted(frame_com.keys())
        for i0, f0 in enumerate(frame_arr):
            for i1 in range(i0 + 1, len(frame_arr)):
                f1  = frame_arr[i1]
                tau = f1 - f0
                if tau > max_tau_frames:
                    break
                tau_idx = tau - 1
                dr = frame_com[f1] - frame_com[f0]
                msd[tau_idx]    += (dr ** 2).sum()
                counts[tau_idx] += 1

    valid = counts > 0
    msd[valid] /= counts[valid]
    msd[~valid] = np.nan
    return tau_vals * dt, msd


max_tau = min(50, n_frames // 4)
fig, ax = plt.subplots(figsize=(7, 4))

for cls, name, color in [(+1, "Tetrahedral", "steelblue"),
                          (-1, "Distorted",   "tomato")]:
    tracks = [t for t in zca.results.tracks.values() if t.label == cls]
    if not tracks:
        continue
    tau_ps, msd = compute_cluster_msd(tracks, max_tau)
    valid = ~np.isnan(msd)
    ax.plot(tau_ps[valid], msd[valid], color=color, lw=2.0, label=name)

ax.set(xlabel="Lag time τ (ps)",
       ylabel=r"$\langle |\Delta \mathbf{r}_{CoM}(\tau)|^2 \rangle$ (Å²)",
       title="Cluster centre-of-mass MSD")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("cluster_plots/cluster_com_msd.png", dpi=150)
plt.close()

# ============================================================================
# 9.  Cluster ζ distribution over time – heatmap per class
# ============================================================================

zeta_bins  = np.linspace(-1.5, 1.5, 61)
zeta_ctrs  = 0.5 * (zeta_bins[:-1] + zeta_bins[1:])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, cls, name in zip(axes, [+1, -1], ["Tetrahedral clusters", "Distorted clusters"]):
    tracks = [t for t in zca.results.tracks.values() if t.label == cls]
    if not tracks:
        ax.text(0.5, 0.5, "No tracks", ha="center", transform=ax.transAxes)
        continue

    # Build 2-D array: rows = time, columns = ζ bins
    heatmap = np.zeros((n_frames, len(zeta_ctrs)))
    frame_to_fi = {int(f): i for i, f in enumerate(zca.frames)}
    for track in tracks:
        for fi_t, (frame, zeta_mean, zeta_std, sz) in enumerate(
            zip(track.frames, track.mean_zeta, track.std_zeta, track.size)
        ):
            fi = frame_to_fi.get(int(frame))
            if fi is None or np.isnan(zeta_mean):
                continue
            # Use sz as weight
            bin_idx = np.searchsorted(zeta_bins, zeta_mean) - 1
            bin_idx = np.clip(bin_idx, 0, len(zeta_ctrs) - 1)
            heatmap[fi, bin_idx] += sz

    # Normalise each frame row to a probability
    row_sums = heatmap.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    heatmap /= row_sums

    pcm = ax.pcolormesh(
        times_ps, zeta_ctrs, heatmap.T,
        cmap="hot_r",
        norm=mcolors.LogNorm(
            vmin=max(heatmap[heatmap > 0].min() if (heatmap > 0).any() else 1e-4, 1e-4),
            vmax=heatmap.max() if heatmap.max() > 0 else 1,
        ),
    )
    plt.colorbar(pcm, ax=ax, label="P(ζ, t)")
    ax.axhline(0, color="white", lw=0.8, ls="--")
    ax.set(xlabel="Time (ps)", ylabel="ζ (Å)", title=name)

plt.tight_layout()
plt.savefig("cluster_plots/cluster_zeta_heatmap.png", dpi=150)
plt.close()

# ============================================================================
# 10. Numerical summary table + CSV export
# ============================================================================

summary = zca.cluster_summary()

print("\n" + "=" * 64)
print("CLUSTER ANALYSIS SUMMARY")
print("=" * 64)
for cls_name, stats in summary.items():
    if stats is None:
        print(f"\n  {cls_name.upper()}: no clusters found.")
        continue
    print(f"\n  {cls_name.upper()}")
    print(f"    Total tracked clusters : {stats['n_tracks']}")
    print(f"    Mean lifetime          : {stats['mean_lifetime_frames'] * dt:.2f} ± "
          f"{stats['std_lifetime_frames'] * dt:.2f} ps")
    print(f"    Mean cluster size      : {stats['mean_size']:.1f} ± "
          f"{stats['std_size']:.1f} molecules")
    print(f"    Mean Rg                : {stats['mean_rg']:.2f} ± "
          f"{stats['std_rg']:.2f} Å")
    print(f"    Mean CoM displacement  : {stats['mean_displacement']:.2f} ± "
          f"{stats['std_displacement']:.2f} Å")

print("=" * 64)

# Per-track CSV
rows = []
for gid, track in sorted(zca.results.tracks.items()):
    rows.append({
        "global_id":          gid,
        "label":              track.label,
        "label_name":         "tetrahedral" if track.label == 1 else "distorted",
        "lifetime_frames":    track.lifetime,
        "lifetime_ps":        track.lifetime * dt,
        "birth_frame":        track.birth_frame,
        "death_frame":        track.death_frame,
        "mean_size":          track.mean_size,
        "mean_rg_A":          track.mean_rg,
        "total_com_disp_A":   track.total_com_displacement,
        "mean_zeta_mean":     float(np.nanmean(track.mean_zeta)),
        "mean_zeta_std":      float(np.nanmean(track.std_zeta)),
    })

if rows:
    import csv
    fieldnames = list(rows[0].keys())
    with open("cluster_plots/cluster_track_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-track CSV: cluster_plots/cluster_track_summary.csv  "
          f"({len(rows)} rows)")

print("\nAll outputs in cluster_plots/:")
for fname in [
    "spatial_slice_frame*.png  (snapshots + animation frames)",
    "cluster_population_timeseries.png",
    "cluster_lifetime_distribution.png",
    "cluster_size_distribution.png",
    "rg_vs_size.png",
    "com_trajectories.png",
    "cluster_com_msd.png",
    "cluster_zeta_heatmap.png",
    "cluster_track_summary.csv",
]:
    print(f"  {fname}")
