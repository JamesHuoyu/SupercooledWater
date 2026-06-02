import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import (
    wrap_positions_orthorhombic,
    unwrap_positions_relative,
    slab_mask_pbc,
    cell_lengths,
    cell_matrix_from_box,
    positions_to_unsheared_orthogonal,
)


def _unique_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h)
            l2.append(l)
            seen.add(l)
    if h2:
        ax.legend(h2, l2, loc="upper right", fontsize=8, frameon=True)


def _scatter_domain_overlay(
    ax,
    positions_wrapped,
    slab,
    result,
    ax_x,
    ax_y,
    halo_size=110,
    core_size=155,
):
    masks = result["masks"]

    overlays = [
        ("slow_halo", "slow halo", "deepskyblue", "none", halo_size, "o", 4),
        ("slow_core", "slow core", "navy", "deepskyblue", core_size, "o", 5),
        ("fast_halo", "fast halo", "magenta", "none", halo_size, "o", 4),
        ("fast_core", "fast core", "darkred", "magenta", core_size, "o", 5),
    ]

    for mask_name, label, edge, face, size, marker, zorder in overlays:
        idx = slab & masks[mask_name]
        if not np.any(idx):
            continue

        ax.scatter(
            positions_wrapped[idx, ax_x],
            positions_wrapped[idx, ax_y],
            s=size,
            marker=marker,
            facecolors=face,
            edgecolors=edge,
            linewidths=1.3,
            alpha=0.95,
            label=label,
            zorder=zorder,
        )


def plot_zeta_cg_slice(
    positions,
    zeta_cg,
    box,
    axis=2,
    slice_center=None,
    slice_width=6.0,
    point_size=48,
    alpha=0.8,
    cmap="coolwarm",
    ax=None,
):
    """
    Plot one zeta_cg spatial slab.

    axis:
        0 -> YZ slab normal to x
        1 -> XZ slab normal to y
        2 -> XY slab normal to z
    """
    pos = wrap_positions_orthorhombic(positions, box)
    zeta_cg = np.asarray(zeta_cg, dtype=float)
    lengths = cell_lengths(box)

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if slice_center is None:
        slice_center = 0.5 * lengths[axis]

    slab = slab_mask_pbc(pos, box, axis, slice_center, slice_width)
    ax_x, ax_y = [i for i in range(3) if i != axis]
    axis_names = ["X", "Y", "Z"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    else:
        fig = ax.figure

    z_s = zeta_cg[slab]
    z_finite = z_s[np.isfinite(z_s)]

    if z_finite.size:
        vmin, vmax = np.nanpercentile(z_finite, [2, 98])
        if vmin == vmax:
            vmin, vmax = np.nanmin(z_finite), np.nanmax(z_finite)
    else:
        vmin, vmax = None, None

    sc = ax.scatter(
        pos[slab, ax_x],
        pos[slab, ax_y],
        c=zeta_cg[slab],
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=alpha,
    )
    fig.colorbar(sc, ax=ax, label=r"$\zeta_{\rm cg}$")

    ax.set_xlabel(f"{axis_names[ax_x]} (Å)")
    ax.set_ylabel(f"{axis_names[ax_y]} (Å)")
    ax.set_xlim(0, lengths[ax_x])
    ax.set_ylim(0, lengths[ax_y])
    ax.set_aspect("equal")
    ax.set_title(
        f"$\\zeta_{{\\rm cg}}$ slice normal to {axis_names[axis]} | "
        f"{axis_names[axis]}={slice_center:.2f}±{slice_width/2:.2f} Å"
    )

    return fig, ax


def plot_domain_slice(
    result,
    axis=2,
    slice_center=None,
    slice_width=6.0,
    point_size=48,
    background_alpha=0.75,
    use_log_color_for_P=True,
):
    """
    Plot a 2D slab with zeta_cg and P3D colors, overlaying extracted domains.

    axis:
        0 -> YZ slab normal to x
        1 -> XZ slab normal to y
        2 -> XY slab normal to z
    """
    pos = result["positions_wrapped"]
    box = result["box"]
    lengths = cell_lengths(box)
    P3D = result["P3D"]
    zeta = result["zeta_cg"]

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if slice_center is None:
        slice_center = 0.5 * lengths[axis]

    slab = slab_mask_pbc(pos, box, axis, slice_center, slice_width)
    ax_x, ax_y = [i for i in range(3) if i != axis]
    axis_names = ["X", "Y", "Z"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    z_s = zeta[slab]
    z_finite = z_s[np.isfinite(z_s)]
    if z_finite.size:
        z_vmin, z_vmax = np.nanpercentile(z_finite, [2, 98])
        if z_vmin == z_vmax:
            z_vmin, z_vmax = np.nanmin(z_finite), np.nanmax(z_finite)
    else:
        z_vmin, z_vmax = None, None

    sc0 = axes[0].scatter(
        pos[slab, ax_x],
        pos[slab, ax_y],
        c=zeta[slab],
        s=point_size,
        cmap="coolwarm",
        vmin=z_vmin,
        vmax=z_vmax,
        edgecolors="none",
        alpha=background_alpha,
        zorder=1,
    )
    plt.colorbar(sc0, ax=axes[0], label=r"$\zeta_{\rm cg}$")
    _scatter_domain_overlay(axes[0], pos, slab, result, ax_x, ax_y)

    axes[0].set_xlabel(f"{axis_names[ax_x]} (Å)")
    axes[0].set_ylabel(f"{axis_names[ax_y]} (Å)")
    axes[0].set_xlim(0, lengths[ax_x])
    axes[0].set_ylim(0, lengths[ax_y])
    axes[0].set_aspect("equal")
    _unique_legend(axes[0])

    p_s = P3D[slab]
    p_finite = p_s[np.isfinite(p_s)]
    norm = None

    if p_finite.size:
        if use_log_color_for_P and np.nanmin(p_finite) > 0.0:
            p_vmin, p_vmax = np.nanpercentile(p_finite, [2, 98])
            p_vmin = max(p_vmin, np.nanmin(p_finite[p_finite > 0.0]))
            if p_vmax > p_vmin:
                norm = LogNorm(vmin=p_vmin, vmax=p_vmax)
        else:
            p_vmin, p_vmax = np.nanpercentile(p_finite, [2, 98])
            norm = Normalize(vmin=p_vmin, vmax=p_vmax)

    sc1 = axes[1].scatter(
        pos[slab, ax_x],
        pos[slab, ax_y],
        c=P3D[slab],
        s=point_size,
        cmap="coolwarm",
        norm=norm,
        edgecolors="none",
        alpha=background_alpha,
        zorder=1,
    )
    plt.colorbar(sc1, ax=axes[1], label=r"$P_i^{3D}$")
    _scatter_domain_overlay(axes[1], pos, slab, result, ax_x, ax_y)

    axes[1].set_xlabel(f"{axis_names[ax_x]} (Å)")
    axes[1].set_ylabel(f"{axis_names[ax_y]} (Å)")
    axes[1].set_xlim(0, lengths[ax_x])
    axes[1].set_ylim(0, lengths[ax_y])
    axes[1].set_aspect("equal")
    _unique_legend(axes[1])

    fig.suptitle(
        f"Slice normal to {axis_names[axis]} | "
        f"{axis_names[axis]} = {slice_center:.2f} ± {slice_width/2:.2f} Å",
        fontsize=14,
    )

    return fig, axes




def plot_zeta_domain_slice(
    result,
    axis=2,
    slice_center=None,
    slice_width=6.0,
    point_size=48,
    background_alpha=0.75,
    cmap="coolwarm",
    ax=None,
):
    """
    Plot a single zeta_cg slab and overlay extracted domains.

    This is the preferred slice visualization for direct zeta_cg-based
    clustering, because there may be no P3D field in the result.
    """
    pos = result["positions_wrapped"]
    box = result["box"]
    lengths = cell_lengths(box)
    zeta = result["zeta_cg"]

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if slice_center is None:
        slice_center = 0.5 * lengths[axis]

    slab = slab_mask_pbc(pos, box, axis, slice_center, slice_width)
    ax_x, ax_y = [i for i in range(3) if i != axis]
    axis_names = ["X", "Y", "Z"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    else:
        fig = ax.figure

    z_s = zeta[slab]
    z_finite = z_s[np.isfinite(z_s)]
    if z_finite.size:
        z_vmin, z_vmax = np.nanpercentile(z_finite, [2, 98])
        if z_vmin == z_vmax:
            z_vmin, z_vmax = np.nanmin(z_finite), np.nanmax(z_finite)
    else:
        z_vmin, z_vmax = None, None

    sc = ax.scatter(
        pos[slab, ax_x],
        pos[slab, ax_y],
        c=zeta[slab],
        s=point_size,
        cmap=cmap,
        vmin=z_vmin,
        vmax=z_vmax,
        edgecolors="none",
        alpha=background_alpha,
        zorder=1,
    )
    plt.colorbar(sc, ax=ax, label=r"$\zeta_{\rm cg}$")
    _scatter_domain_overlay(ax, pos, slab, result, ax_x, ax_y)

    ax.set_xlabel(f"{axis_names[ax_x]} (Å)")
    ax.set_ylabel(f"{axis_names[ax_y]} (Å)")
    ax.set_xlim(0, lengths[ax_x])
    ax.set_ylim(0, lengths[ax_y])
    ax.set_aspect("equal")
    ax.set_title(
        f"Direct $\\zeta_{{\\rm cg}}$ domains | slice normal to {axis_names[axis]} | "
        f"{axis_names[axis]}={slice_center:.2f}±{slice_width/2:.2f} Å"
    )
    _unique_legend(ax)

    return fig, ax

def plot_score_distributions(result, bins=50):
    """Plot local standardized P, local standardized zeta, slow score, and fast score."""
    Pz = result["local"]["Pz"]
    Zz = result["local"]["Zz"]
    slow = result["scores"]["slow"]
    fast = result["scores"]["fast"]
    valid = result["masks"]["valid"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    data = [
        (Pz[valid], "Z(local P)"),
        (Zz[valid], "Z(local zeta)"),
        (slow[valid], "slow score"),
        (fast[valid], "fast score"),
    ]

    for ax, (x, label) in zip(axes, data):
        ax.hist(x[np.isfinite(x)], bins=bins, density=True, edgecolor="k", alpha=0.65)
        ax.set_xlabel(label)
        ax.set_ylabel("density")

    return fig, axes


def _subsample_indices(indices, max_points=None, random_state=0):
    indices = np.asarray(indices, dtype=int)
    if max_points is None or len(indices) <= max_points:
        return indices
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def _add_box_edges_plotly(fig, box, line_color="black", line_width=4):
    H = cell_matrix_from_box(box)
    a, b, c = H
    corners = np.array([
        [0, 0, 0], a, a + b, b,
        c, a + c, a + b + c, b + c,
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        fig.add_trace(_plotly_go().Scatter3d(
            x=[corners[i, 0], corners[j, 0]],
            y=[corners[i, 1], corners[j, 1]],
            z=[corners[i, 2], corners[j, 2]],
            mode="lines",
            line=dict(color=line_color, width=line_width),
            showlegend=False,
            hoverinfo="skip",
        ))


def _plotly_go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for this function.") from exc
    return go


def _add_cluster_hull_plotly(fig, points, color="rgba(0,0,255,0.15)", name="cluster hull"):
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return
    try:
        hull = ConvexHull(points)
    except Exception:
        return

    simplices = hull.simplices
    fig.add_trace(_plotly_go().Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=simplices[:, 0],
        j=simplices[:, 1],
        k=simplices[:, 2],
        color=color,
        opacity=0.12,
        name=name,
        hoverinfo="skip",
        showlegend=True,
    ))


def plot_domains_3d_plotly(
    result,
    color_by="P3D",
    show_background=True,
    max_background=3000,
    show_slow=True,
    show_fast=True,
    show_hulls=True,
    show_centroids=True,
    hull_for="members",
    random_state=0,
    title="3D joint domains in simulation box",
):
    """Full 3D simulation-box visualization with Plotly."""
    go = _plotly_go()

    pos = np.asarray(result["positions_wrapped"], dtype=float)
    box = np.asarray(result["box"], dtype=float)
    P3D = np.asarray(result["P3D"], dtype=float)
    zeta = np.asarray(result["zeta_cg"], dtype=float)

    masks = result["masks"]
    clusters = result["clusters"]

    slow_all = np.asarray(masks["slow_all"], dtype=bool)
    fast_all = np.asarray(masks["fast_all"], dtype=bool)
    slow_core = np.asarray(masks["slow_core"], dtype=bool)
    slow_halo = np.asarray(masks["slow_halo"], dtype=bool)
    fast_core = np.asarray(masks["fast_core"], dtype=bool)
    fast_halo = np.asarray(masks["fast_halo"], dtype=bool)
    valid = np.asarray(masks["valid"], dtype=bool)

    fig = go.Figure()

    if show_background:
        bg_mask = valid & ~(slow_all | fast_all)
        bg_idx = np.where(bg_mask)[0]
        bg_idx = _subsample_indices(bg_idx, max_points=max_background, random_state=random_state)

        if len(bg_idx) > 0:
            marker_dict = dict(size=3, opacity=0.35)
            if color_by == "P3D":
                cvals = P3D[bg_idx]
                if np.all(cvals > 0):
                    cshow = np.log10(cvals)
                    ctitle = "log10(P3D)"
                else:
                    cshow = cvals
                    ctitle = "P3D"
                marker_dict.update(dict(
                    color=cshow,
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title=ctitle),
                ))
            elif color_by == "zeta":
                marker_dict.update(dict(
                    color=zeta[bg_idx],
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="zeta_cg"),
                ))
            else:
                marker_dict.update(dict(color="lightgray"))

            fig.add_trace(go.Scatter3d(
                x=pos[bg_idx, 0],
                y=pos[bg_idx, 1],
                z=pos[bg_idx, 2],
                mode="markers",
                marker=marker_dict,
                name="background",
            ))

    overlays = []
    if show_slow:
        overlays.extend([
            ("slow halo", slow_halo, dict(size=5, color="deepskyblue", opacity=0.75)),
            ("slow core", slow_core, dict(size=8, color="navy", opacity=1.0, line=dict(color="cyan", width=1))),
        ])
    if show_fast:
        overlays.extend([
            ("fast halo", fast_halo, dict(size=5, color="magenta", opacity=0.75)),
            ("fast core", fast_core, dict(size=8, color="darkred", opacity=1.0, line=dict(color="pink", width=1))),
        ])

    for name, mask, marker in overlays:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=pos[idx, 0],
            y=pos[idx, 1],
            z=pos[idx, 2],
            mode="markers",
            marker=marker,
            name=name,
        ))

    def _process_kind(kind_name, cluster_list, hull_color, centroid_symbol):
        for ncl, c in enumerate(cluster_list, start=1):
            idx = c["core_indices"] if hull_for == "core" else c["member_indices"]
            if len(idx) == 0:
                continue

            cen = np.asarray(c["centroid"], dtype=float)
            pts_local = unwrap_positions_relative(pos[idx], cen, box)

            if show_hulls and len(pts_local) >= 4:
                _add_cluster_hull_plotly(fig, pts_local, color=hull_color, name=f"{kind_name} hull #{ncl}")

            if show_centroids:
                fig.add_trace(go.Scatter3d(
                    x=[cen[0]], y=[cen[1]], z=[cen[2]],
                    mode="markers+text",
                    marker=dict(size=10, color=hull_color, symbol=centroid_symbol),
                    text=[f"{kind_name[0].upper()}{ncl}"],
                    textposition="top center",
                    name=f"{kind_name} centroid #{ncl}",
                ))

    if show_slow:
        _process_kind("slow", clusters["slow"], "rgba(0,0,180,0.65)", "diamond")
    if show_fast:
        _process_kind("fast", clusters["fast"], "rgba(180,0,80,0.65)", "diamond")

    _add_box_edges_plotly(fig, box)

    Lx, Ly, Lz = cell_lengths(box)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (Å)", range=[-2, Lx + 2]),
            yaxis=dict(title="Y (Å)", range=[-2, Ly + 2]),
            zaxis=dict(title="Z (Å)", range=[-2, Lz + 2]),
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def plot_single_cluster_3d_plotly(
    result,
    kind="slow",
    cluster_rank=1,
    center_mode="centroid",
    show_hull=True,
    title=None,
):
    """Cluster-centric Plotly 3D view, with the selected cluster locally unwrapped."""
    go = _plotly_go()

    pos = np.asarray(result["positions_wrapped"], dtype=float)
    box = np.asarray(result["box"], dtype=float)
    cluster_list = result["clusters"][kind]

    if len(cluster_list) == 0:
        raise ValueError(f"No {kind} clusters available.")
    if cluster_rank < 1 or cluster_rank > len(cluster_list):
        raise ValueError(f"cluster_rank must be between 1 and {len(cluster_list)}.")

    c = cluster_list[cluster_rank - 1]
    members = np.asarray(c["member_indices"], dtype=int)
    core = np.asarray(c["core_indices"], dtype=int)
    halo = np.asarray(c["halo_indices"], dtype=int)

    if center_mode == "core_mean" and len(core) > 0:
        ref = np.mean(pos[core], axis=0)
    else:
        ref = np.asarray(c["centroid"], dtype=float)

    pos_members = unwrap_positions_relative(pos[members], ref, box)
    pos_core = unwrap_positions_relative(pos[core], ref, box) if len(core) > 0 else np.empty((0, 3))
    pos_halo = unwrap_positions_relative(pos[halo], ref, box) if len(halo) > 0 else np.empty((0, 3))

    pos_members0 = pos_members - ref[None, :]
    pos_core0 = pos_core - ref[None, :]
    pos_halo0 = pos_halo - ref[None, :]

    fig = go.Figure()

    if len(pos_halo0) > 0:
        fig.add_trace(go.Scatter3d(
            x=pos_halo0[:, 0],
            y=pos_halo0[:, 1],
            z=pos_halo0[:, 2],
            mode="markers",
            marker=dict(size=5, color="deepskyblue" if kind == "slow" else "magenta", opacity=0.75),
            name=f"{kind} halo",
        ))

    if len(pos_core0) > 0:
        fig.add_trace(go.Scatter3d(
            x=pos_core0[:, 0],
            y=pos_core0[:, 1],
            z=pos_core0[:, 2],
            mode="markers",
            marker=dict(
                size=8,
                color="navy" if kind == "slow" else "darkred",
                opacity=1.0,
                line=dict(color="cyan" if kind == "slow" else "pink", width=1),
            ),
            name=f"{kind} core",
        ))

    if show_hull and len(pos_members0) >= 4:
        _add_cluster_hull_plotly(
            fig,
            pos_members0,
            color="rgba(0,0,180,0.65)" if kind == "slow" else "rgba(180,0,80,0.65)",
            name=f"{kind} hull",
        )

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=8, color="black", symbol="x"),
        text=["reference"],
        textposition="top center",
        name="reference",
    ))

    pad = 2.0
    xmin, ymin, zmin = np.min(pos_members0, axis=0) - pad
    xmax, ymax, zmax = np.max(pos_members0, axis=0) + pad

    fig.update_layout(
        title=title or f"{kind.capitalize()} cluster #{cluster_rank} (cluster-centric view)",
        scene=dict(
            xaxis=dict(title="ΔX (Å)", range=[xmin, xmax]),
            yaxis=dict(title="ΔY (Å)", range=[ymin, ymax]),
            zaxis=dict(title="ΔZ (Å)", range=[zmin, zmax]),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def _add_box_edges_matplotlib(ax, box, color="black", lw=1.0, alpha=0.8):
    H = cell_matrix_from_box(box)
    a, b, c = H
    corners = np.array([
        [0, 0, 0], a, a + b, b,
        c, a + c, a + b + c, b + c,
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color, lw=lw, alpha=alpha,
        )


def _add_convex_hull_matplotlib(ax, points, facecolor, edgecolor, alpha=0.12, lw=0.4):
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return
    try:
        hull = ConvexHull(points)
    except Exception:
        return

    faces = [points[simplex] for simplex in hull.simplices]
    poly = Poly3DCollection(faces, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha)
    ax.add_collection3d(poly)


def _set_axes_equal_3d(ax, box):
    H = cell_matrix_from_box(box)
    corners = np.array([
        [0, 0, 0], H[0], H[0] + H[1], H[1],
        H[2], H[0] + H[2], H[0] + H[1] + H[2], H[1] + H[2],
    ], dtype=float)
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0e-12)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(spans)


def plot_domains_3d_matplotlib(
    result,
    show_background=True,
    max_background=2000,
    show_hulls=True,
    hull_for="members",
    elev=22,
    azim=38,
    background_color_by=None,
    random_state=0,
    figsize=(7.0, 6.2),
):
    """Static 3D visualization suitable for publication-oriented export."""
    pos = np.asarray(result["positions_wrapped"], dtype=float)
    box = np.asarray(result["box"], dtype=float)
    P3D = np.asarray(result["P3D"], dtype=float)
    zeta = np.asarray(result["zeta_cg"], dtype=float)
    masks = result["masks"]
    clusters = result["clusters"]

    slow_all = np.asarray(masks["slow_all"], dtype=bool)
    fast_all = np.asarray(masks["fast_all"], dtype=bool)
    valid = np.asarray(masks["valid"], dtype=bool)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if show_background:
        bg = np.where(valid & ~(slow_all | fast_all))[0]
        bg = _subsample_indices(bg, max_points=max_background, random_state=random_state)

        if len(bg) > 0:
            if background_color_by == "P3D":
                cvals = np.log10(P3D[bg]) if np.all(P3D[bg] > 0) else P3D[bg]
                sc = ax.scatter(pos[bg, 0], pos[bg, 1], pos[bg, 2], c=cvals, s=3, alpha=0.25, cmap="viridis")
                fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, label="log10(P3D)" if np.all(P3D[bg] > 0) else "P3D")
            elif background_color_by == "zeta":
                sc = ax.scatter(pos[bg, 0], pos[bg, 1], pos[bg, 2], c=zeta[bg], s=3, alpha=0.25, cmap="coolwarm")
                fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, label=r"$\zeta_{\rm cg}$")
            else:
                ax.scatter(pos[bg, 0], pos[bg, 1], pos[bg, 2], c="lightgray", s=3, alpha=0.20)

    overlays = [
        ("slow_halo", "deepskyblue", 8, 0.70),
        ("slow_core", "navy", 16, 1.00),
        ("fast_halo", "magenta", 8, 0.70),
        ("fast_core", "darkred", 16, 1.00),
    ]
    for mask_name, color, size, alpha in overlays:
        idx = np.where(masks[mask_name])[0]
        if len(idx):
            ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2], c=color, s=size, alpha=alpha, label=mask_name)

    if show_hulls:
        for kind, face, edge in [("slow", "deepskyblue", "navy"), ("fast", "magenta", "darkred")]:
            for c in clusters[kind]:
                idx = c["core_indices"] if hull_for == "core" else c["member_indices"]
                if len(idx) < 4:
                    continue
                pts = unwrap_positions_relative(pos[idx], np.asarray(c["centroid"]), box)
                _add_convex_hull_matplotlib(ax, pts, facecolor=face, edgecolor=edge, alpha=0.10)

    _add_box_edges_matplotlib(ax, box)
    _set_axes_equal_3d(ax, box)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig, ax


def plot_track_boundary_evolution(
    results,
    track,
    axis=2,
    use_core=False,
    coordinate_system="unsheared",
    every=1,
    max_observations=8,
    draw_hull=True,
    point_size=20,
    alpha=0.65,
    ax=None,
):
    """
    Overlay the projected boundary/footprint of one tracked domain over time.

    Parameters
    ----------
    results : list[dict]
        Framewise outputs from extract_zeta_domains_over_frames(...) or
        extract_domains_over_frames(...).
    track : dict
        One track returned by track_clusters_over_time(...).
    axis : int
        Projection normal. axis=2 gives the XY footprint.
    use_core : bool
        If True, draw only core particles. Otherwise draw all member particles.
    coordinate_system : {'unsheared', 'cartesian'}
        'unsheared' maps triclinic coordinates to fractional*cell_length space,
        which removes the trivial affine skew of the simulation cell.
        'cartesian' draws locally unwrapped Cartesian coordinates.
    every : int
        Plot every Nth observation along the track.
    max_observations : int
        Maximum number of observations to overlay.
    draw_hull : bool
        Draw a 2D convex hull around the projected points when possible.

    Notes
    -----
    The hull is a visual footprint, not the actual non-convex cluster boundary.
    For a stricter non-convex boundary, replace this with an alpha-shape step.
    """
    if coordinate_system not in ("unsheared", "cartesian"):
        raise ValueError("coordinate_system must be 'unsheared' or 'cartesian'.")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    observations = list(track.get("observations", []))[::max(1, int(every))]
    if max_observations is not None and len(observations) > max_observations:
        pick = np.linspace(0, len(observations) - 1, max_observations).round().astype(int)
        observations = [observations[i] for i in pick]

    if len(observations) == 0:
        raise ValueError("track has no observations to plot.")

    ax_x, ax_y = [i for i in range(3) if i != axis]
    axis_names = ["X", "Y", "Z"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    else:
        fig = ax.figure

    cmap = plt.get_cmap("viridis")
    denom = max(len(observations) - 1, 1)

    for k, obs in enumerate(observations):
        step = int(obs["frame_step"])
        result = results[step]
        pos = np.asarray(result["positions_wrapped"], dtype=float)
        box = result["box"]
        idx_key = "core_indices" if use_core else "member_indices"
        idx = np.asarray(obs[idx_key], dtype=int)
        if idx.size == 0:
            continue

        ref = np.asarray(obs["centroid"], dtype=float)
        if coordinate_system == "unsheared":
            pts = positions_to_unsheared_orthogonal(pos[idx], box, reference=ref)
        else:
            pts = unwrap_positions_relative(pos[idx], ref, box)

        xy = pts[:, [ax_x, ax_y]]
        color = cmap(k / denom)
        label = f"f={obs['frame']}"
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=alpha, color=color, label=label)

        if draw_hull and xy.shape[0] >= 3:
            try:
                hull = ConvexHull(xy)
                ring = np.r_[hull.vertices, hull.vertices[0]]
                ax.plot(xy[ring, 0], xy[ring, 1], lw=1.5, color=color, alpha=0.95)
            except Exception:
                pass

    ax.set_xlabel(f"{axis_names[ax_x]} ({'unsheared Å' if coordinate_system == 'unsheared' else 'Å'})")
    ax.set_ylabel(f"{axis_names[ax_y]} ({'unsheared Å' if coordinate_system == 'unsheared' else 'Å'})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Boundary evolution of {track.get('track_id', 'track')} | "
        f"{'core' if use_core else 'members'} | projection normal to {axis_names[axis]}"
    )
    _unique_legend(ax)
    return fig, ax
