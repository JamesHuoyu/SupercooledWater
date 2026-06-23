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

# =============================================================================
# v6 visualization layer
# =============================================================================
# The functions below intentionally override earlier v5 definitions with the
# same names.  The physical clustering/tracking still uses MDAnalysis triclinic
# PBC in geometry.py.  These routines only decide how to display the coordinates.

from .geometry import (  # noqa: E402,F811
    positions_to_brick_coordinates,
    brick_to_cartesian_coordinates,
    unwrap_cluster_positions,
    box_to_mda_dimensions,
)


def _normalise_coordinate_system(name):
    name = str(name).lower()
    aliases = {
        "cell": "brick",
        "fractional": "brick",
        "unsheared": "brick",
        "orthogonal": "brick",
        "triclinic": "cartesian",
        "real": "cartesian",
        "lab": "cartesian",
    }
    return aliases.get(name, name)


def _orthogonal_slab_mask(display_positions, lengths, axis, slice_center, slice_width):
    display_positions = np.asarray(display_positions, dtype=float)
    L = float(lengths[int(axis)])
    d = display_positions[:, int(axis)] - float(slice_center)
    d = d - L * np.round(d / L)
    return np.abs(d) <= 0.5 * float(slice_width)


def _display_positions_for_frame(positions, box, coordinate_system="brick"):
    coord = _normalise_coordinate_system(coordinate_system)
    if coord == "brick":
        return positions_to_brick_coordinates(positions, box, wrap=True), cell_lengths(box), "brick"
    if coord == "cartesian":
        pos = wrap_positions_orthorhombic(positions, box)
        # Cartesian triclinic coordinates do not occupy [0,Lx]x[0,Ly] exactly;
        # limits are set from the true cell corners in the plotting functions.
        return pos, cell_lengths(box), "cartesian"
    raise ValueError("coordinate_system must be 'brick' or 'cartesian'.")


def _plot_limits_for_coordinate_system(box, coordinate_system, axis):
    coord = _normalise_coordinate_system(coordinate_system)
    ax_x, ax_y = [i for i in range(3) if i != axis]
    if coord == "brick":
        lengths = cell_lengths(box)
        return (0.0, lengths[ax_x]), (0.0, lengths[ax_y])

    H = cell_matrix_from_box(box)
    a, b, c = H
    corners = np.array([
        [0, 0, 0], a, a + b, b,
        c, a + c, a + b + c, b + c,
    ], dtype=float)
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    return (mins[ax_x], maxs[ax_x]), (mins[ax_y], maxs[ax_y])


def _axis_label(axis_id, coordinate_system):
    axis_names = ["X", "Y", "Z"]
    coord = _normalise_coordinate_system(coordinate_system)
    if coord == "brick":
        return f"{axis_names[axis_id]}_brick (Å)"
    return f"{axis_names[axis_id]} (Å)"


def _scatter_domain_overlay_v6(
    ax,
    display_positions,
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
        if mask_name not in masks:
            continue
        idx = slab & np.asarray(masks[mask_name], dtype=bool)
        if not np.any(idx):
            continue
        ax.scatter(
            display_positions[idx, ax_x],
            display_positions[idx, ax_y],
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
    coordinate_system="brick",
    ax=None,
):
    """
    Plot a zeta_cg slab in either a brick display cell or the real triclinic cell.

    Parameters
    ----------
    coordinate_system : {'brick', 'cartesian'}, default 'brick'
        ``'brick'`` cuts the triclinic cell along periodic boundaries and pastes
        it into an orthogonal box for readable 2D slices.  ``'cartesian'`` uses
        the real sheared coordinates; this is mainly a diagnostic view.
    """
    coord = _normalise_coordinate_system(coordinate_system)
    pos_disp, lengths, _ = _display_positions_for_frame(positions, box, coord)
    zeta_cg = np.asarray(zeta_cg, dtype=float)

    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if slice_center is None:
        slice_center = 0.5 * lengths[axis]

    if coord == "brick":
        slab = _orthogonal_slab_mask(pos_disp, lengths, axis, slice_center, slice_width)
    else:
        slab = slab_mask_pbc(pos_disp, box, axis, slice_center, slice_width, coordinate_system="cartesian")

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
        pos_disp[slab, ax_x],
        pos_disp[slab, ax_y],
        c=zeta_cg[slab],
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=alpha,
    )
    fig.colorbar(sc, ax=ax, label=r"$\zeta_{\rm cg}$")

    xlim, ylim = _plot_limits_for_coordinate_system(box, coord, axis)
    ax.set_xlabel(_axis_label(ax_x, coord))
    ax.set_ylabel(_axis_label(ax_y, coord))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"$\\zeta_{{\\rm cg}}$ {coord} slice normal to {axis_names[axis]} | "
        f"{axis_names[axis]}={slice_center:.2f}±{slice_width/2:.2f} Å"
    )
    return fig, ax


def plot_zeta_domain_slice(
    result,
    axis=2,
    slice_center=None,
    slice_width=6.0,
    point_size=48,
    background_alpha=0.75,
    cmap="coolwarm",
    coordinate_system="brick",
    ax=None,
):
    """
    Plot one zeta_cg slab and overlay direct zeta_cg domains.

    Use ``coordinate_system='brick'`` for publication-style slices where a
    tilted triclinic cell is periodically repasted into an orthogonal box.  Use
    ``'cartesian'`` only when the real sheared geometry is desired in 2D.
    """
    coord = _normalise_coordinate_system(coordinate_system)
    pos = np.asarray(result["positions_wrapped"], dtype=float)
    box = result["box"]
    zeta = np.asarray(result["zeta_cg"], dtype=float)
    pos_disp, lengths, _ = _display_positions_for_frame(pos, box, coord)

    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if slice_center is None:
        slice_center = 0.5 * lengths[axis]

    if coord == "brick":
        slab = _orthogonal_slab_mask(pos_disp, lengths, axis, slice_center, slice_width)
    else:
        slab = slab_mask_pbc(pos, box, axis, slice_center, slice_width, coordinate_system="cartesian")

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
        pos_disp[slab, ax_x],
        pos_disp[slab, ax_y],
        c=zeta[slab],
        s=point_size,
        cmap=cmap,
        vmin=z_vmin,
        vmax=z_vmax,
        edgecolors="none",
        alpha=background_alpha,
        zorder=1,
    )
    fig.colorbar(sc, ax=ax, label=r"$\zeta_{\rm cg}$")
    _scatter_domain_overlay_v6(ax, pos_disp, slab, result, ax_x, ax_y)

    xlim, ylim = _plot_limits_for_coordinate_system(box, coord, axis)
    ax.set_xlabel(_axis_label(ax_x, coord))
    ax.set_ylabel(_axis_label(ax_y, coord))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Direct $\\zeta_{{\\rm cg}}$ domains | {coord} slice normal to {axis_names[axis]} | "
        f"{axis_names[axis]}={slice_center:.2f}±{slice_width/2:.2f} Å"
    )
    _unique_legend(ax)
    return fig, ax


def plot_domain_slice(
    result,
    axis=2,
    slice_center=None,
    slice_width=6.0,
    point_size=48,
    background_alpha=0.75,
    use_log_color_for_P=True,
    coordinate_system="brick",
):
    """Plot zeta_cg and P3D slices with domain overlays in a chosen display cell."""
    coord = _normalise_coordinate_system(coordinate_system)
    pos = np.asarray(result["positions_wrapped"], dtype=float)
    box = result["box"]
    lengths = cell_lengths(box)
    P3D = np.asarray(result["P3D"], dtype=float)
    zeta = np.asarray(result["zeta_cg"], dtype=float)
    pos_disp, lengths, _ = _display_positions_for_frame(pos, box, coord)

    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if slice_center is None:
        slice_center = 0.5 * lengths[axis]

    if coord == "brick":
        slab = _orthogonal_slab_mask(pos_disp, lengths, axis, slice_center, slice_width)
    else:
        slab = slab_mask_pbc(pos, box, axis, slice_center, slice_width, coordinate_system="cartesian")

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
        pos_disp[slab, ax_x], pos_disp[slab, ax_y], c=zeta[slab],
        s=point_size, cmap="coolwarm", vmin=z_vmin, vmax=z_vmax,
        edgecolors="none", alpha=background_alpha, zorder=1,
    )
    plt.colorbar(sc0, ax=axes[0], label=r"$\zeta_{\rm cg}$")
    _scatter_domain_overlay_v6(axes[0], pos_disp, slab, result, ax_x, ax_y)

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
        pos_disp[slab, ax_x], pos_disp[slab, ax_y], c=P3D[slab],
        s=point_size, cmap="coolwarm", norm=norm,
        edgecolors="none", alpha=background_alpha, zorder=1,
    )
    plt.colorbar(sc1, ax=axes[1], label=r"$P_i^{3D}$")
    _scatter_domain_overlay_v6(axes[1], pos_disp, slab, result, ax_x, ax_y)

    xlim, ylim = _plot_limits_for_coordinate_system(box, coord, axis)
    for ax in axes:
        ax.set_xlabel(_axis_label(ax_x, coord))
        ax.set_ylabel(_axis_label(ax_y, coord))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        _unique_legend(ax)

    fig.suptitle(
        f"{coord} slice normal to {axis_names[axis]} | "
        f"{axis_names[axis]} = {slice_center:.2f} ± {slice_width/2:.2f} Å",
        fontsize=14,
    )
    return fig, axes


def _add_brick_box_edges_matplotlib(ax, box, color="black", lw=1.0, alpha=0.8):
    Lx, Ly, Lz = cell_lengths(box)
    corners = np.array([
        [0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
        [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz],
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color=color, lw=lw, alpha=alpha)


def _set_axes_equal_3d_for_points(ax, points, pad=0.0):
    points = np.asarray(points, dtype=float)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        return
    pts = points[finite]
    mins = pts.min(axis=0) - pad
    maxs = pts.max(axis=0) + pad
    spans = np.maximum(maxs - mins, 1.0e-12)
    center = 0.5 * (mins + maxs)
    half = 0.5 * np.max(spans)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect([1, 1, 1])


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
    coordinate_system="cartesian",
    draw_box=True,
):
    """
    Static 3D domain visualization.

    The default ``coordinate_system='cartesian'`` keeps the real triclinic/sheared
    geometry.  Use ``'brick'`` only when you want a de-skewed diagnostic view.
    """
    coord = _normalise_coordinate_system(coordinate_system)
    pos0 = np.asarray(result["positions_wrapped"], dtype=float)
    box = result["box"]
    P3D = np.asarray(result.get("P3D", np.full(len(pos0), np.nan)), dtype=float)
    zeta = np.asarray(result["zeta_cg"], dtype=float)
    masks = result["masks"]
    clusters = result["clusters"]

    if coord == "brick":
        pos = positions_to_brick_coordinates(pos0, box, wrap=True)
    elif coord == "cartesian":
        pos = pos0
    else:
        raise ValueError("coordinate_system must be 'cartesian' or 'brick'.")

    slow_all = np.asarray(masks.get("slow_all", np.zeros(len(pos), dtype=bool)), dtype=bool)
    fast_all = np.asarray(masks.get("fast_all", np.zeros(len(pos), dtype=bool)), dtype=bool)
    valid = np.asarray(masks.get("valid", np.ones(len(pos), dtype=bool)), dtype=bool)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if show_background:
        bg = np.where(valid & ~(slow_all | fast_all))[0]
        bg = _subsample_indices(bg, max_points=max_background, random_state=random_state)
        if len(bg) > 0:
            color_key = background_color_by
            if color_key in ("zeta_cg", "zeta", "Z", "zeta-cg"):
                sc = ax.scatter(pos[bg, 0], pos[bg, 1], pos[bg, 2], c=zeta[bg], s=3, alpha=0.25, cmap="coolwarm")
                fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, label=r"$\zeta_{\rm cg}$")
            elif color_key == "P3D" and np.any(np.isfinite(P3D[bg])):
                cvals = np.log10(P3D[bg]) if np.all(P3D[bg] > 0) else P3D[bg]
                sc = ax.scatter(pos[bg, 0], pos[bg, 1], pos[bg, 2], c=cvals, s=3, alpha=0.25, cmap="viridis")
                fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, label="log10(P3D)" if np.all(P3D[bg] > 0) else "P3D")
            else:
                ax.scatter(pos[bg, 0], pos[bg, 1], pos[bg, 2], c="lightgray", s=3, alpha=0.20)

    overlays = [
        ("slow_halo", "deepskyblue", 8, 0.70),
        ("slow_core", "navy", 16, 1.00),
        ("fast_halo", "magenta", 8, 0.70),
        ("fast_core", "darkred", 16, 1.00),
    ]
    for mask_name, color, size, alpha in overlays:
        if mask_name not in masks:
            continue
        idx = np.where(np.asarray(masks[mask_name], dtype=bool))[0]
        if len(idx):
            ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2], c=color, s=size, alpha=alpha, label=mask_name)

    if show_hulls:
        for kind, face, edge in [("slow", "deepskyblue", "navy"), ("fast", "magenta", "darkred")]:
            for c in clusters.get(kind, []):
                idx = c["core_indices"] if hull_for == "core" else c["member_indices"]
                if len(idx) < 4:
                    continue
                if coord == "brick":
                    pts = positions_to_brick_coordinates(pos0[idx], box, wrap=False, reference=np.asarray(c["centroid"]))
                else:
                    pts = unwrap_positions_relative(pos0[idx], np.asarray(c["centroid"]), box)
                _add_convex_hull_matplotlib(ax, pts, facecolor=face, edgecolor=edge, alpha=0.10)

    if draw_box:
        if coord == "brick":
            _add_brick_box_edges_matplotlib(ax, box)
            L = cell_lengths(box)
            ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_zlim(0, L[2])
            ax.set_box_aspect(L)
        else:
            _add_box_edges_matplotlib(ax, box)
            _set_axes_equal_3d(ax, box)
    else:
        _set_axes_equal_3d_for_points(ax, pos, pad=2.0)

    ax.view_init(elev=elev, azim=azim)
    suffix = "brick" if coord == "brick" else "triclinic Cartesian"
    ax.set_xlabel(f"X ({suffix}, Å)")
    ax.set_ylabel(f"Y ({suffix}, Å)")
    ax.set_zlabel(f"Z ({suffix}, Å)")
    _unique_legend(ax)
    fig.tight_layout()
    return fig, ax


def _indices_from_observation(obs, index_set=None, use_core=None):
    if index_set is None:
        if use_core is None:
            index_set = "members"
        else:
            index_set = "core" if use_core else "members"
    mapping = {
        "members": "member_indices",
        "member": "member_indices",
        "all": "member_indices",
        "core": "core_indices",
        "halo": "halo_indices",
    }
    key = mapping.get(str(index_set).lower())
    if key is None:
        raise ValueError("index_set must be 'members', 'core', or 'halo'.")
    return np.asarray(obs[key], dtype=int), key


def _strain_for_observation(obs, reference_obs, shear_rate=None, dt_per_frame=None, strain_getter=None):
    if strain_getter is not None:
        return float(strain_getter(obs)) - float(strain_getter(reference_obs))
    if shear_rate is None or dt_per_frame is None:
        raise ValueError(
            "coordinate_system='deaffined' requires either strain_getter(obs) or "
            "both shear_rate and dt_per_frame."
        )
    return float(shear_rate) * (float(obs["frame"]) - float(reference_obs["frame"])) * float(dt_per_frame)


def _track_points_for_observation(
    result,
    obs,
    idx,
    coordinate_system,
    align_centroid=True,
    reference_obs=None,
    shear_rate=None,
    dt_per_frame=None,
    strain_getter=None,
    shear_axis=0,
    gradient_axis=1,
):
    coord = _normalise_coordinate_system(coordinate_system)
    if coord == "deaffined":
        coord_base = "brick"
    else:
        coord_base = coord

    pos = np.asarray(result["positions_wrapped"], dtype=float)
    box = result["box"]
    centroid = np.asarray(obs["centroid"], dtype=float)

    if coord_base == "cartesian":
        pts = unwrap_positions_relative(pos[idx], centroid, box)
        center = centroid
    elif coord_base == "brick":
        pts = positions_to_brick_coordinates(pos[idx], box, wrap=False, reference=centroid)
        center = positions_to_brick_coordinates(centroid, box, wrap=False)
    else:
        raise ValueError("coordinate_system must be 'cartesian', 'brick', or 'deaffined'.")

    if align_centroid:
        pts = pts - center[None, :]

    if coord == "deaffined":
        if reference_obs is None:
            reference_obs = obs
        dgamma = _strain_for_observation(
            obs,
            reference_obs,
            shear_rate=shear_rate,
            dt_per_frame=dt_per_frame,
            strain_getter=strain_getter,
        )
        pts = pts.copy()
        pts[:, int(shear_axis)] -= dgamma * pts[:, int(gradient_axis)]
    return pts


def collect_track_boundary_points(
    results,
    track,
    index_set="members",
    coordinate_system="brick",
    align_centroid=True,
    every=1,
    max_observations=None,
    shear_rate=None,
    dt_per_frame=None,
    strain_getter=None,
    shear_axis=0,
    gradient_axis=1,
):
    """
    Return projected-ready point clouds for one tracked cluster.

    This is useful when you want to compute your own hull, alpha shape, or save
    the boundary points for further analysis.  It uses cluster-aware local
    unwrapping before applying the requested visualization transform.
    """
    observations = list(track.get("observations", []))[::max(1, int(every))]
    if max_observations is not None and len(observations) > max_observations:
        pick = np.linspace(0, len(observations) - 1, int(max_observations)).round().astype(int)
        observations = [observations[i] for i in pick]
    if not observations:
        return []

    reference_obs = observations[0]
    rows = []
    for obs in observations:
        step = int(obs["frame_step"])
        result = results[step]
        idx, idx_key = _indices_from_observation(obs, index_set=index_set)
        if idx.size == 0:
            continue
        pts = _track_points_for_observation(
            result=result,
            obs=obs,
            idx=idx,
            coordinate_system=coordinate_system,
            align_centroid=align_centroid,
            reference_obs=reference_obs,
            shear_rate=shear_rate,
            dt_per_frame=dt_per_frame,
            strain_getter=strain_getter,
            shear_axis=shear_axis,
            gradient_axis=gradient_axis,
        )
        rows.append({
            "track_id": track.get("track_id", None),
            "kind": track.get("kind", None),
            "frame": int(obs["frame"]),
            "frame_step": int(obs["frame_step"]),
            "index_set": idx_key,
            "indices": idx,
            "points": pts,
            "centroid": np.asarray(obs["centroid"], dtype=float),
        })
    return rows


def plot_track_boundary_evolution(
    results,
    track,
    axis=2,
    index_set="members",
    use_core=None,
    coordinate_system="brick",
    align_centroid=True,
    every=1,
    max_observations=8,
    draw_hull=True,
    draw_points=True,
    draw_centroid=True,
    point_size=20,
    alpha=0.65,
    mode="overlay",
    shear_rate=None,
    dt_per_frame=None,
    strain_getter=None,
    shear_axis=0,
    gradient_axis=1,
    ax=None,
):
    """
    Visualize how one tracked cluster boundary evolves over time.

    Parameters
    ----------
    coordinate_system : {'cartesian', 'brick', 'deaffined'}, default 'brick'
        - ``cartesian``: locally unwrapped true sheared Cartesian coordinates.
        - ``brick``: locally unwrapped, then de-skewed into an orthogonal display
          cell.  This is the recommended first diagnostic.
        - ``deaffined``: brick coordinates plus a simple shear correction
          ``x_shear -= Δγ * y_gradient``.  Provide either ``strain_getter`` or
          both ``shear_rate`` and ``dt_per_frame``.
    align_centroid : bool, default True
        If True, each observation is centered at its own cluster centroid before
        overlaying.  This isolates boundary-shape evolution from translation.
    mode : {'overlay', 'panel'}, default 'overlay'
        Overlay all selected observations on one axis, or create one panel per
        observation.
    index_set : {'members', 'core', 'halo'}
        Which part of the cluster defines the boundary.

    Notes
    -----
    The convex hull is only a visual footprint.  It is not a non-convex alpha
    shape.  For publication, use it as a qualitative continuity diagnostic.
    """
    coord = _normalise_coordinate_system(coordinate_system)
    if coord not in ("cartesian", "brick", "deaffined"):
        raise ValueError("coordinate_system must be 'cartesian', 'brick', or 'deaffined'.")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    # Backward compatibility with the v5 use_core argument.
    if use_core is not None and index_set == "members":
        index_set = "core" if use_core else "members"

    rows = collect_track_boundary_points(
        results=results,
        track=track,
        index_set=index_set,
        coordinate_system=coord,
        align_centroid=align_centroid,
        every=every,
        max_observations=max_observations,
        shear_rate=shear_rate,
        dt_per_frame=dt_per_frame,
        strain_getter=strain_getter,
        shear_axis=shear_axis,
        gradient_axis=gradient_axis,
    )
    if not rows:
        raise ValueError("track has no observations/particles to plot.")

    ax_x, ax_y = [i for i in range(3) if i != axis]
    axis_names = ["X", "Y", "Z"]
    cmap = plt.get_cmap("viridis")
    denom = max(len(rows) - 1, 1)

    if mode == "panel":
        n = len(rows)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        for extra_ax in axes[n:]:
            extra_ax.axis("off")
    elif mode == "overlay":
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
        else:
            fig = ax.figure
        axes = [ax]
    else:
        raise ValueError("mode must be 'overlay' or 'panel'.")

    all_xy = []
    for k, row in enumerate(rows):
        pts = row["points"]
        xy = pts[:, [ax_x, ax_y]]
        all_xy.append(xy)
        color = cmap(k / denom)
        label = f"f={row['frame']}"
        ax_here = axes[k] if mode == "panel" else axes[0]

        if draw_points:
            ax_here.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=alpha, color=color, label=label)
        if draw_hull and xy.shape[0] >= 3:
            try:
                hull = ConvexHull(xy)
                ring = np.r_[hull.vertices, hull.vertices[0]]
                ax_here.plot(xy[ring, 0], xy[ring, 1], lw=1.5, color=color, alpha=0.95, label=None if draw_points else label)
            except Exception:
                pass
        if draw_centroid and align_centroid:
            ax_here.scatter([0.0], [0.0], marker="x", s=35, color="black", alpha=0.75)
        if mode == "panel":
            ax_here.set_title(label)
            ax_here.set_aspect("equal", adjustable="box")

    if all_xy:
        xy_all = np.vstack(all_xy)
        finite = np.all(np.isfinite(xy_all), axis=1)
        if np.any(finite):
            mn = xy_all[finite].min(axis=0)
            mx = xy_all[finite].max(axis=0)
            pad = 0.08 * np.maximum(mx - mn, 1.0)
            xlim = (mn[0] - pad[0], mx[0] + pad[0])
            ylim = (mn[1] - pad[1], mx[1] + pad[1])
            for ax_here in axes[:len(rows) if mode == "panel" else 1]:
                ax_here.set_xlim(*xlim)
                ax_here.set_ylim(*ylim)

    unit = "Å"
    if coord == "brick":
        unit = "brick Å"
    elif coord == "deaffined":
        unit = "de-affined brick Å"

    target_axes = axes[:len(rows)] if mode == "panel" else [axes[0]]
    for ax_here in target_axes:
        ax_here.set_xlabel(f"{axis_names[ax_x]} ({unit})")
        ax_here.set_ylabel(f"{axis_names[ax_y]} ({unit})")
        ax_here.set_aspect("equal", adjustable="box")

    if mode == "overlay":
        axes[0].set_title(
            f"Boundary evolution of {track.get('track_id', 'track')} | "
            f"{index_set} | {coord} | projection normal to {axis_names[axis]}"
        )
        _unique_legend(axes[0])
    else:
        fig.suptitle(
            f"Boundary evolution of {track.get('track_id', 'track')} | "
            f"{index_set} | {coord} | projection normal to {axis_names[axis]}",
            fontsize=13,
        )
    return fig, axes[0] if mode == "overlay" else axes[:len(rows)]
