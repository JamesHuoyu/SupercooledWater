"""
Propensity Field Cluster Pipeline
===================================

Full workflow:

  1.  Compute dynamic propensity P_i via isoconfigurational ensemble
  2.  Run PropensityFieldClusters on the reference frame
  3.  Build density-field ρ_P, find ρ_th, label connected regions
  4.  Track clusters across subsequent ZOP frames
  5.  Characterise each cluster by initial ζ_cg
  6.  All diagnostic plots (density field, slices, tracks, ζ–propensity
      correlation, cluster-level statistics)
  7.  Robustness checks: vary σ, ρ_thr_mode, active-fraction threshold
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import MDAnalysis as mda
from scipy.optimize import linear_sum_assignment

from water.tools.custom_hbond_analysis   import HydrogenBondAnalysis as HBA
from water.tools.zeta_order_parameter    import ZetaOrderParameter   as ZOP
from water.tools.propensity_field_clusters import (
    PropensityFieldClusters  as PFC,
    PropensityFieldTracker   as PFT,
    finalise_track_arrays,
)

os.makedirs("pfc_plots", exist_ok=True)

# ============================================================================
# 0.  Load universe and run HBA + ZOP
# ============================================================================

TOPOLOGY   = "/root/water/TIP4P/Ice/test/tip4p-ice-225K.data"
TRAJECTORY = "/root/water/TIP4P/Ice/225/dump_225_test.lammpstrj"
DT_PS      = 0.2
TEMPERATURE = 225.0

u = mda.Universe(TOPOLOGY, TRAJECTORY, format="LAMMPSDUMP", dt=DT_PS)

hba = HBA(universe=u,
          donors_sel="type 1", hydrogens_sel="type 2",
          acceptors_sel="type 1",
          d_a_cutoff=3.5, h_d_a_angle_cutoff=30.0,
          update_selections=False)
hba.run(verbose=True)

zop = ZOP(hba=hba, central_sel="type 1", shell_cutoff=6.0)
zop.run(verbose=True)

ow_ag       = zop._central_ag
ow_indices  = zop.results.central_indices   # (N_OW,)
n_ow        = len(ow_indices)
times_ps    = zop.results.times
n_frames    = len(zop.frames)

print(f"\nZOP done: {n_ow} OW atoms, {n_frames} frames")

# ============================================================================
# 1.  Dynamic propensity
#     Two routes:
#
#     Route A (isoconfigurational ensemble – the rigorous way):
#       Run M trajectories from the same initial configuration with
#       randomised velocities.  Compute per-atom MSD at lag τ*.
#       P_i = ⟨|Δr_i(τ*)|²⟩_iso
#
#     Route B (single-trajectory approximation used here when M=1):
#       Use the MSD from the single available trajectory, sampled at τ*.
#       This is an approximation but still captures mobile/immobile
#       heterogeneity within the trajectory window.
#
#     Set USE_ISO_ENSEMBLE = True and supply iso_positions if you have
#     an isoconfigurational ensemble; otherwise Route B is used.
# ============================================================================

USE_ISO_ENSEMBLE = False   # set True + fill iso_positions if available

TAU_STAR_PS = 50.0         # lag time in ps at which propensity is evaluated
                           # paper suggests τ* near the plateau / cage breakout

TAU_STAR_FRAMES = max(1, int(TAU_STAR_PS / DT_PS))

# ---- Reference frame (frame 0 = initial configuration) -------------------
REF_FRAME_IDX = 0   # index into zop.frames
ref_frame     = zop.frames[REF_FRAME_IDX]

u.trajectory[ref_frame]
pos_ref = ow_ag.positions.copy()   # (N_OW, 3)
box_ref = u.trajectory.ts.dimensions[:3].copy()

if USE_ISO_ENSEMBLE:
    # iso_positions: shape (M, N_OW, 3) – M replicas at τ*
    # Replace the line below with your actual ensemble data.
    raise NotImplementedError(
        "Set iso_positions to your isoconfigurational ensemble array."
    )
    # propensity = np.mean(
    #     np.sum((iso_positions - pos_ref[None]) ** 2, axis=2),
    #     axis=0,
    # )
else:
    # Route B: single-trajectory MSD at τ_star
    fi_ref  = REF_FRAME_IDX
    fi_tau  = min(fi_ref + TAU_STAR_FRAMES, n_frames - 1)

    # Read positions at t=0 and t=τ*
    u.trajectory[zop.frames[fi_ref]]
    p0 = ow_ag.positions.copy()
    u.trajectory[zop.frames[fi_tau]]
    p_tau = ow_ag.positions.copy()

    # Minimum-image displacement
    dr    = p_tau - p0
    dr   -= box_ref * np.round(dr / box_ref)
    propensity = (dr ** 2).sum(axis=1)   # (N_OW,) in Å²

print(f"\nPropensity (τ*={TAU_STAR_PS} ps): "
      f"mean={propensity.mean():.3f} Å², "
      f"max={propensity.max():.3f} Å², "
      f"top-15% threshold = "
      f"{np.nanpercentile(propensity, 85.0):.3f} Å²")

# ============================================================================
# 2.  Retrieve ζ_cg at the reference frame
# ============================================================================

zop_row_ref = zop._frame_to_row[ref_frame]
zeta_ref    = zop.results.zeta[zop_row_ref].copy()   # (N_OW,) at t=0

print(f"ζ_cg (ref frame): mean={np.nanmean(zeta_ref):.4f} Å")

# ============================================================================
# 3.  Build density-field clusters at the reference frame
# ============================================================================

pfc_ref = PFC(
    positions            = pos_ref,
    propensity           = propensity,
    zeta_cg              = zeta_ref,
    box                  = box_ref,
    sigma                = 3.0,            # Å – first-shell scale
    grid_spacing         = 1.0,            # Å – coarse grid
    propensity_top_frac  = 0.15,           # top 15 % active
    rho_threshold_mode   = "iterative",
    min_cluster_voxels   = 4,
    refine_grid          = False,          # set True for sharper boundaries
)
pfc_ref.run()

print(f"\nReference frame clusters:")
print(f"  ρ_th = {pfc_ref.results.rho_threshold:.5f}")
print(f"  N_clusters = {pfc_ref.results.n_clusters}")
print(f"  N_active molecules = {pfc_ref.results.n_active}")

# ============================================================================
# 4.  Plot 1 – Density field with ρ_th contour (Fig. from document §A.B)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pfc_ref.plot_density_slice(axis=2, ax=axes[0])
pfc_ref.plot_slice(axis=2, ax=axes[1],
                   show_boundaries=True, show_propensity=False,
                   title="Clusters coloured by ζ_cg at t=0")
plt.tight_layout()
plt.savefig("pfc_plots/ref_frame_clusters.png", dpi=150)
plt.close()
print("[4] pfc_plots/ref_frame_clusters.png")

# ============================================================================
# 5.  Plot 2 – Propensity vs ζ_cg scatter coloured by cluster membership
# ============================================================================

fig, ax = plt.subplots(figsize=(6, 5))
bg_mask = pfc_ref.results.atom_cluster_id == 0
ax.scatter(propensity[bg_mask], zeta_ref[bg_mask],
           s=4, alpha=0.2, color="lightgrey", label="Background")

for stats in pfc_ref.results.cluster_stats[:10]:
    mi = stats.member_indices
    ax.scatter(propensity[mi], zeta_ref[mi],
               s=12, alpha=0.7,
               label=f"Cluster #{stats.cluster_id}  "
                     f"(N={stats.n_molecules}, ζ={stats.mean_zeta:+.2f})")

ax.axhline(0, color="black", lw=0.8, ls="--", label="ζ=0")
ax.set(xlabel="Propensity P_i (Å²)",
       ylabel="ζ_cg (Å)",
       title="Structure vs dynamics: ζ_cg vs propensity")
ax.legend(fontsize=7, loc="upper left", ncol=2)
plt.tight_layout()
plt.savefig("pfc_plots/propensity_vs_zeta.png", dpi=150)
plt.close()
print("[5] pfc_plots/propensity_vs_zeta.png")

# ============================================================================
# 6.  Cluster characterisation summary at reference frame
# ============================================================================

print("\n--- Cluster statistics at reference frame ---")
print(f"{'ID':>4}  {'N_mol':>6}  {'⟨ζ⟩':>8}  {'f_tet':>6}  "
      f"{'⟨P⟩':>10}  {'Rg':>6}  {'V_vox':>7}")
for s in pfc_ref.results.cluster_stats[:15]:
    print(f"{s.cluster_id:>4}  {s.n_molecules:>6}  "
          f"{s.mean_zeta:>+8.3f}  {s.frac_tetrahedral:>6.3f}  "
          f"{s.mean_propensity:>10.3f}  {s.rg:>6.2f}  "
          f"{s.volume_vox:>7}")

# ============================================================================
# 7.  Track clusters across all ZOP frames
# ============================================================================

tracker = PFT(min_overlap=0.3, min_lifetime=3)

print("\nTracking clusters across frames …")
for fi, frame in enumerate(zop.frames):
    # Positions at this frame
    u.trajectory[frame]
    pos_fi   = ow_ag.positions.copy()
    box_fi   = u.trajectory.ts.dimensions[:3].copy()

    # ζ_cg at this frame
    zop_row  = zop._frame_to_row[frame]
    zeta_fi  = zop.results.zeta[zop_row].copy()

    # Build density-field clusters (propensity is fixed to reference frame)
    pfc_fi = PFC(
        positions           = pos_fi,
        propensity          = propensity,   # fixed to t=0
        zeta_cg             = zeta_fi,      # evolves each frame
        box                 = box_fi,
        sigma               = 3.0,
        grid_spacing        = 1.0,
        propensity_top_frac = 0.15,
        rho_threshold_mode  = "iterative",
        min_cluster_voxels  = 4,
    )
    pfc_fi.run()
    tracker.update(pfc_fi, frame_time=frame * DT_PS)

    if fi % 20 == 0:
        print(f"  frame {frame:5d}  ({frame * DT_PS:7.1f} ps)  "
              f"N_clusters = {pfc_fi.results.n_clusters}")

raw_tracks = tracker.finalise()
tracks = {gid: finalise_track_arrays(t) for gid, t in raw_tracks.items()}

print(f"\nTracking complete: {len(tracks)} tracks "
      f"with lifetime ≥ {tracker.min_lifetime} frames")

# ============================================================================
# 8.  Plot 3 – Cluster size and ζ_cg time series for top tracked clusters
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

top_tracks = sorted(tracks.values(),
                    key=lambda t: t["n_molecules"].mean(),
                    reverse=True)[:8]

cmap_t = plt.get_cmap("tab10")
for i, tr in enumerate(top_tracks):
    t_ps = tr["times"]
    label = f"ID={tr['global_id']} (N̄={tr['n_molecules'].mean():.0f})"
    axes[0].plot(t_ps, tr["n_molecules"], lw=1.5,
                 color=cmap_t(i), label=label)
    axes[1].plot(t_ps, tr["mean_zeta"], lw=1.5,
                 color=cmap_t(i), label=label)

axes[0].set(ylabel="Cluster size N", title="Tracked cluster size vs time")
axes[0].legend(fontsize=7, ncol=2)
axes[1].axhline(0, color="black", lw=0.8, ls="--")
axes[1].set(xlabel="Time (ps)", ylabel="⟨ζ_cg⟩",
            title="Mean ζ_cg per tracked cluster vs time")
axes[1].legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("pfc_plots/tracked_cluster_timeseries.png", dpi=150)
plt.close()
print("[8] pfc_plots/tracked_cluster_timeseries.png")

# ============================================================================
# 9.  Plot 4 – ζ_cg(t=0) histogram: inside clusters vs background
# ============================================================================

cluster_mask = pfc_ref.results.atom_cluster_id > 0
bg_mask      = ~cluster_mask
zeta_valid   = ~np.isnan(zeta_ref)

bins = np.linspace(-1.5, 1.5, 61)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(zeta_ref[cluster_mask & zeta_valid], bins=bins,
        density=True, alpha=0.6, color="tomato",
        label=f"Inside clusters  (N={cluster_mask.sum()})")
ax.hist(zeta_ref[bg_mask & zeta_valid], bins=bins,
        density=True, alpha=0.5, color="steelblue",
        label=f"Background       (N={bg_mask.sum()})")
ax.axvline(0, color="black", lw=0.9, ls="--")
ax.set(xlabel="ζ_cg (Å)", ylabel="P(ζ)",
       title="Initial ζ_cg: cluster members vs background")
ax.legend(fontsize=9)

# KS-test to quantify separation
from scipy.stats import ks_2samp
ks_stat, ks_p = ks_2samp(
    zeta_ref[cluster_mask & zeta_valid],
    zeta_ref[bg_mask      & zeta_valid],
)
ax.text(0.98, 0.95, f"KS = {ks_stat:.3f}  p = {ks_p:.2e}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9)

plt.tight_layout()
plt.savefig("pfc_plots/zeta_inside_vs_background.png", dpi=150)
plt.close()
print(f"[9] KS test: D={ks_stat:.3f}, p={ks_p:.2e}")
print("    pfc_plots/zeta_inside_vs_background.png")

# ============================================================================
# 10. Plot 5 – Cluster CoM trajectories (top 6 tracks)
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 7))
cmap_com = plt.get_cmap("plasma")

for i, tr in enumerate(top_tracks[:6]):
    com   = tr["centroid"]        # (T, 3)
    t_ps  = np.array(tr["times"])
    sc = ax.scatter(com[:, 0], com[:, 1],
                    c=t_ps, cmap="plasma",
                    vmin=times_ps.min(), vmax=times_ps.max(),
                    s=15, alpha=0.8, zorder=2)
    if len(com) >= 2:
        ax.annotate("", xy=com[-1, :2], xytext=com[-2, :2],
                    arrowprops=dict(arrowstyle="->",
                                   color=cmap_t(i), lw=1.2))

plt.colorbar(sc, ax=ax, label="Time (ps)", shrink=0.8)
ax.set(xlabel="x (Å)", ylabel="y (Å)",
       title="Cluster CoM trajectories (top 6 by mean size)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("pfc_plots/cluster_com_trajectories.png", dpi=150)
plt.close()
print("[10] pfc_plots/cluster_com_trajectories.png")

# ============================================================================
# 11. Plot 6 – frac_tetrahedral per cluster over time
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 4))
for i, tr in enumerate(top_tracks[:8]):
    ax.plot(tr["times"], tr["frac_tet"], lw=1.5,
            color=cmap_t(i),
            label=f"ID={tr['global_id']}")
ax.axhline(0.5, color="black", lw=0.8, ls="--", label="f_tet = 0.5")
ax.set(xlabel="Time (ps)",
       ylabel="Fraction ζ > 0",
       title="Tetrahedral fraction per tracked cluster")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("pfc_plots/cluster_frac_tet_timeseries.png", dpi=150)
plt.close()
print("[11] pfc_plots/cluster_frac_tet_timeseries.png")

# ============================================================================
# 12. Plot 7 – Cluster lifetime distribution and size distribution
# ============================================================================

lifetimes_ps  = np.array([tr["lifetime"] * DT_PS for tr in tracks.values()])
mean_sizes    = np.array([tr["n_molecules"].mean() for tr in tracks.values()])
mean_zeta_all = np.array([np.nanmean(tr["mean_zeta"]) for tr in tracks.values()])
disps         = np.array([tr["com_displacement"] for tr in tracks.values()])

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].hist(lifetimes_ps, bins=30, color="steelblue", edgecolor="white",
                density=True)
axes[0, 0].axvline(lifetimes_ps.mean(), color="black", lw=1.2, ls="--",
                   label=f"Mean = {lifetimes_ps.mean():.1f} ps")
axes[0, 0].set(xlabel="Lifetime (ps)", ylabel="P", title="Track lifetime")
axes[0, 0].legend(fontsize=9)

axes[0, 1].hist(mean_sizes, bins=30, color="darkorchid", edgecolor="white",
                density=True)
axes[0, 1].axvline(mean_sizes.mean(), color="black", lw=1.2, ls="--",
                   label=f"Mean = {mean_sizes.mean():.1f}")
axes[0, 1].set(xlabel="Mean cluster size N", ylabel="P", title="Cluster size")
axes[0, 1].legend(fontsize=9)

axes[1, 0].hist(mean_zeta_all, bins=40, color="tomato", edgecolor="white",
                density=True)
axes[1, 0].axvline(0, color="black", lw=0.8, ls="--")
axes[1, 0].set(xlabel="Time-averaged ⟨ζ_cg⟩ per cluster",
               ylabel="P", title="Structural identity of tracks")

axes[1, 1].scatter(mean_sizes, mean_zeta_all, s=8, alpha=0.5,
                   c=lifetimes_ps, cmap="plasma")
axes[1, 1].axhline(0, color="black", lw=0.8, ls="--")
sc2 = axes[1, 1].scatter(mean_sizes, mean_zeta_all, s=8, alpha=0.5,
                          c=lifetimes_ps, cmap="plasma")
plt.colorbar(sc2, ax=axes[1, 1], label="Lifetime (ps)")
axes[1, 1].set(xlabel="Mean size N", ylabel="⟨ζ_cg⟩",
               title="Size vs structure, coloured by lifetime")

plt.tight_layout()
plt.savefig("pfc_plots/cluster_aggregate_stats.png", dpi=150)
plt.close()
print("[12] pfc_plots/cluster_aggregate_stats.png")

# ============================================================================
# 13. Robustness check: vary σ and active-fraction threshold
# ============================================================================

print("\nRobustness check (varying σ and active fraction) …")
robust_rows = []
for sigma_test in [2.0, 3.0, 4.0]:
    for frac_test in [0.10, 0.15, 0.20]:
        pfc_t = PFC(
            positions           = pos_ref,
            propensity          = propensity,
            zeta_cg             = zeta_ref,
            box                 = box_ref,
            sigma               = sigma_test,
            grid_spacing        = 1.0,
            propensity_top_frac = frac_test,
            rho_threshold_mode  = "iterative",
            min_cluster_voxels  = 4,
        )
        pfc_t.run()
        n_cl    = pfc_t.results.n_clusters
        rho_th  = pfc_t.results.rho_threshold
        stats_t = pfc_t.results.cluster_stats
        mean_z_in  = float(np.nanmean(
            [zeta_ref[s.member_indices] for s in stats_t
             for _ in [None] if len(s.member_indices) > 0]
        )) if stats_t else np.nan
        robust_rows.append((sigma_test, frac_test, n_cl,
                            float(rho_th), mean_z_in))
        print(f"  σ={sigma_test:.1f} Å  frac={frac_test:.2f}  "
              f"N_cl={n_cl:3d}  ρ_th={rho_th:.5f}")

# ============================================================================
# 14. Save numerical tracks to CSV
# ============================================================================

import csv
rows = [["global_id", "lifetime_frames", "lifetime_ps",
         "mean_size", "mean_zeta", "frac_tet",
         "com_displacement_A"]]
for tr in tracks.values():
    rows.append([
        tr["global_id"],
        tr["lifetime"],
        tr["lifetime"] * DT_PS,
        float(tr["n_molecules"].mean()),
        float(np.nanmean(tr["mean_zeta"])),
        float(np.nanmean(tr["frac_tet"])),
        tr["com_displacement"],
    ])

with open("pfc_plots/cluster_tracks_summary.csv", "w", newline="") as fh:
    csv.writer(fh).writerows(rows)
print(f"\n[14] pfc_plots/cluster_tracks_summary.csv  ({len(rows)-1} tracks)")

print("\nAll outputs:")
for f in sorted(os.listdir("pfc_plots")):
    print(f"  pfc_plots/{f}")
