"""
Reproduce Kikutsuji, Kim & Matubayasi J. Chem. Phys. 150, 204502 (2019)
==========================================================================

Run this after placing cage_jump_analysis.py in your water/tools/ directory.

What this script produces
--------------------------
  cagejump_plots/fig2a_P_tauC.png      – P(τ_C) distribution
  cagejump_plots/fig2b_tauC_tauHB.png  – ⟨τ_C⟩ vs τ_HB comparison
  cagejump_plots/fig3a_P_tauJ.png      – P(τ_J) distribution
  cagejump_plots/fig3b_P_rJ.png        – P(r_J) distribution
  cagejump_plots/fig4a_JMSD.png        – JMSD(Θ_J)
  cagejump_plots/fig4b_MSD.png         – MSD with plateau and 6Dt line
  cagejump_plots/fig5_D_vs_estimate.png – D_MSD vs ρ_J D_J
  cagejump_plots/hbond_acf.png         – c(t) with stretched-exp fit
  cagejump_plots/rhoJ_distribution.png – per-molecule ρ_J histogram
  cagejump_plots/zeta_vs_tauC.png      – ζ vs τ_C (requires ZOP)
  cagejump_summary.csv                 – all scalar results
  cagejump_*.npy                       – raw arrays
"""

import numpy as np
import MDAnalysis as mda

from tools.custom_hbond_analysis import HydrogenBondAnalysis as HBA
from tools.zeta_order_parameter  import ZetaOrderParameter  as ZOP
from tools.cage_jump_analysis    import CageJumpAnalysis     as CJA

# ============================================================================
# 0.  System parameters – edit these to match your simulation
# ============================================================================

TOPOLOGY   = "/root/shared-nvme/tip4p-ice-225K.data"
TRAJECTORY = "/root/shared-nvme/dump_225_test.lammpstrj"
TEMPERATURE = 225.0   # K
DT_PS       = 0.2    # ps per dump frame  (paper used 0.2 ps)

# LAMMPS type integers
OW_TYPE  = "type 1"
HW_TYPE  = "type 2"

# ============================================================================
# 1.  Load universe
# ============================================================================

u = mda.Universe(
    TOPOLOGY, TRAJECTORY,
    format="LAMMPSDUMP",
    dt=DT_PS,
)
print(f"Loaded: {u.atoms.n_atoms} atoms, {u.trajectory.n_frames} frames, "
      f"dt = {DT_PS} ps")

# ============================================================================
# 2.  H-bond analysis  (our criterion: R_OO ≤ 3.5 Å, β_HDA ≤ 30°)
#     Paper criterion: R_OO ≤ 0.34 nm = 3.4 Å, β_OHO ≤ 30°
#     The angle convention is the same (H-D-A at donor);
#     only the distance cutoff differs slightly.
# ============================================================================

hba = HBA(
    universe=u,
    donors_sel=OW_TYPE,
    hydrogens_sel=HW_TYPE,
    acceptors_sel=OW_TYPE,
    d_a_cutoff=3.4,           # Å – paper uses 3.4 Å; 3.5 is our default
    h_d_a_angle_cutoff=30.0,  # degrees
    update_selections=False,
)
hba.run(verbose=True)

print(f"\nHBA complete: {len(hba.results.hbonds)} H-bond observations "
      f"over {len(hba.frames)} frames")

# ============================================================================
# 3.  Zeta order parameter  (optional but enables ζ–τ_C correlation)
# ============================================================================

zop = ZOP(
    hba=hba,
    central_sel=OW_TYPE,
    shell_cutoff=6.0,
)
zop.run(verbose=True)

print(f"\nZOP complete: ⟨ζ⟩ = {np.nanmean(zop.results.zeta):.4f} Å")

# ============================================================================
# 4.  Cage-jump analysis
# ============================================================================

cja = CJA(
    hba=hba,
    zop=zop,
    oxygen_sel=OW_TYPE,
    n_hb_cage=4,          # tetrahedral cage definition (paper default)
    temperature=TEMPERATURE,
)
cja.run()

# ============================================================================
# 5.  Generate all paper figures + save numerical results
# ============================================================================

cja.plot_all("cagejump_plots")
cja.save_results("cagejump")

# ============================================================================
# 6.  Print the key numbers that appear in the paper's Figs. 2–5
# ============================================================================

r = cja.results

print("\n=== KEY RESULTS  (compare with Fig. 2b, 3a, 4b, 5) ===")
print(f"  ⟨τ_C⟩           = {r.mean_tau_C:.2f} ps")
print(f"  τ_HB            = {r.tau_HB:.2f} ps  (β = {r.beta_HB:.3f})")
print(f"  ⟨τ_J⟩           = {r.mean_tau_J:.2f} ps   "
      f"(paper: ~1 ps, weakly T-dependent)")
print(f"  ⟨r²_J⟩          = {r.mean_r2_J:.4f} Å²")
print(f"  ρ_J             = {r.rho_J:.4f}")
print(f"  ρ_J (⟨τ_J⟩/sum) = {r.rho_J_mean_field:.4f}")
print(f"  D_J             = {r.D_J:.4e} m²/s")
print(f"  D_estimate      = {r.D_estimate:.4e} m²/s   "
      f"[= ρ_J × ⟨r²_J⟩ / 6⟨τ_J⟩]")
print(f"  D_MSD           = {r.D_MSD:.4e} m²/s")
print(f"\n  Ratio D_estimate/D_MSD = {r.D_estimate/r.D_MSD:.3f}  "
      f"(paper: close to 1.0 above 190 K)")

# ============================================================================
# 7.  (Advanced) multi-temperature comparison
#     Uncomment and adapt if you have trajectories at multiple T values.
# ============================================================================

# SYSTEMS = [
#     (225, "/path/to/225K.lammpstrj"),
#     (240, "/path/to/240K.lammpstrj"),
#     (260, "/path/to/260K.lammpstrj"),
# ]
#
# records = []
# for T, traj in SYSTEMS:
#     u_t = mda.Universe(TOPOLOGY, traj, format="LAMMPSDUMP", dt=DT_PS)
#     hba_t = HBA(universe=u_t, donors_sel=OW_TYPE, hydrogens_sel=HW_TYPE,
#                 acceptors_sel=OW_TYPE, d_a_cutoff=3.5, h_d_a_angle_cutoff=30.0)
#     hba_t.run()
#     cja_t = CJA(hba=hba_t, oxygen_sel=OW_TYPE, n_hb_cage=4, temperature=T)
#     cja_t.run()
#     r_t = cja_t.results
#     records.append({
#         "T":           T,
#         "inv_T":       1000.0 / T,
#         "mean_tau_C":  r_t.mean_tau_C,
#         "tau_HB":      r_t.tau_HB,
#         "D_estimate":  r_t.D_estimate,
#         "D_MSD":       r_t.D_MSD,
#         "rho_J":       r_t.rho_J,
#         "mean_r2_J":   r_t.mean_r2_J,
#         "mean_tau_J":  r_t.mean_tau_J,
#     })
#
# # Reproduce Fig. 2b (Arrhenius plot) and Fig. 5 (D vs estimate)
# import matplotlib.pyplot as plt
# inv_T = [rec["inv_T"] for rec in records]
#
# fig, ax1 = plt.subplots(figsize=(6, 5))
# ax2 = ax1.twinx()
# ax1.semilogy(inv_T, [r["mean_tau_C"] for r in records], "s--",
#              color="steelblue", label=r"$\langle\tau_C\rangle$")
# ax1.semilogy(inv_T, [r["tau_HB"] for r in records], "o--",
#              color="tomato", label=r"$\tau_{HB}$")
# ax2.semilogy(inv_T, [1.0/r["D_MSD"] for r in records], "D-",
#              color="grey", label=r"$D^{-1}$")
# ax1.set(xlabel="1000/T (K⁻¹)", ylabel="Time (ps)")
# ax2.set(ylabel="D⁻¹ (ps m⁻² s)")
# ax1.legend(loc="upper left", fontsize=9)
# ax2.legend(loc="upper right", fontsize=9)
# plt.tight_layout()
# plt.savefig("cagejump_plots/fig2b_arrhenius.png", dpi=150)
# plt.close()
#
# D_est_list = [r["D_estimate"] for r in records]
# D_msd_list = [r["D_MSD"]     for r in records]
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.loglog(D_est_list, D_msd_list, "o", ms=8, color="navy")
# for rec in records:
#     ax.annotate(f"{rec['T']}K",
#                 (rec["D_estimate"], rec["D_MSD"]),
#                 textcoords="offset points", xytext=(5, 3), fontsize=8)
# lo = min(min(D_est_list), min(D_msd_list)) * 0.5
# hi = max(max(D_est_list), max(D_msd_list)) * 2.0
# ax.loglog([lo, hi], [lo, hi], "k-", lw=1.0)
# ax.set(xlabel=r"$\rho_J\langle r^2_J\rangle/\langle\tau_J\rangle$ (m²/s)",
#        ylabel=r"$D_{MSD}$ (m²/s)",
#        title="Fig. 5: Cage-jump model vs MSD")
# plt.tight_layout()
# plt.savefig("cagejump_plots/fig5_multi_T.png", dpi=150)
# plt.close()
