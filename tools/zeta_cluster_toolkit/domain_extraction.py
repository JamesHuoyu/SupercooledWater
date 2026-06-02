import numpy as np
from .geometry import (
    wrap_positions_orthorhombic,
    minimum_image,
    pbc_centroid,
    pbc_radius_of_gyration,
    query_ball_point_pbc,
    query_pairs_pbc,
    cell_matrix_from_box,
)


def transform_propensity(P3D, mode="auto"):
    """
    Transform dynamic propensity before scoring.

    mode="auto" uses log(P3D) only when all finite values are positive and
    the dynamic range is large; otherwise it keeps raw P3D.
    """
    P3D = np.asarray(P3D, dtype=float)
    finite = np.isfinite(P3D)

    if not np.any(finite):
        raise ValueError("P3D contains no finite values.")

    if mode == "auto":
        p_valid = P3D[finite]
        positive = p_valid[p_valid > 0.0]
        if positive.size == p_valid.size:
            ratio = np.nanmax(positive) / max(np.nanmin(positive), np.finfo(float).eps)
            mode = "log" if ratio > 5.0 else "raw"
        else:
            mode = "raw"

    if mode == "raw":
        return P3D.astype(float), "raw"

    if mode == "log":
        if np.nanmin(P3D[finite]) <= 0.0:
            raise ValueError("P3D has non-positive values; use mode='raw' or mode='auto'.")
        return np.log(P3D), "log"

    raise ValueError("mode must be 'auto', 'log', or 'raw'.")


def robust_zscore(x, method="robust"):
    """Return robust or standard z-score while preserving NaNs."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(x)

    if not np.any(finite):
        return out

    xf = x[finite]

    if method == "robust":
        center = np.nanmedian(xf)
        mad = np.nanmedian(np.abs(xf - center))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            scale = np.nanstd(xf)
    elif method == "standard":
        center = np.nanmean(xf)
        scale = np.nanstd(xf)
    else:
        raise ValueError("method must be 'robust' or 'standard'.")

    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = 1.0

    out[finite] = (xf - center) / scale
    return out


def local_kernel_average(positions, values, box, length=3.0, cutoff=6.0):
    """
    Particle-centered local exponential-kernel average:

        Fbar_i = sum_j F_j exp(-r_ij / length) / sum_j exp(-r_ij / length)

    The self term is included.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    cell_matrix_from_box(box)

    pos = wrap_positions_orthorhombic(positions, box)
    neighbors = query_ball_point_pbc(pos, box, cutoff)

    out = np.full(values.shape, np.nan, dtype=float)

    for i, js in enumerate(neighbors):
        js = np.asarray(js, dtype=int)
        vals = values[js]
        finite = np.isfinite(vals)

        if not np.any(finite):
            continue

        js = js[finite]
        vals = vals[finite]
        delta = minimum_image(pos[js] - pos[i][None, :], box)
        dist = np.linalg.norm(delta, axis=1)
        weights = np.exp(-dist / length)
        sw = np.sum(weights)

        if sw > 0.0:
            out[i] = np.sum(weights * vals) / sw

    return out


class _DisjointSet:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=int)
        self.size = np.ones(n, dtype=int)

    def find(self, x):
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def component_labels_from_seed_growth(
    positions,
    box,
    grow_mask,
    seed_mask,
    r_connect=3.6,
    min_cluster_size=10,
    min_core_size=3,
):
    """
    Build connected components in grow_mask and retain only components with
    at least min_core_size seed particles.
    """
    positions = np.asarray(positions, dtype=float)
    cell_matrix_from_box(box)
    grow_mask = np.asarray(grow_mask, dtype=bool)
    seed_mask = np.asarray(seed_mask, dtype=bool)

    n = len(positions)
    labels = np.zeros(n, dtype=int)

    if not np.any(grow_mask) or not np.any(seed_mask & grow_mask):
        return labels

    pos = wrap_positions_orthorhombic(positions, box)
    pairs = query_pairs_pbc(pos, box, r_connect, active_mask=grow_mask)
    dsu = _DisjointSet(n)

    if pairs.size > 0:
        for i, j in pairs:
            if grow_mask[i] and grow_mask[j]:
                dsu.union(i, j)

    groups = {}
    grow_indices = np.where(grow_mask)[0]
    for idx in grow_indices:
        root = dsu.find(idx)
        groups.setdefault(root, []).append(idx)

    components = []
    for members in groups.values():
        members = np.asarray(members, dtype=int)
        core_members = members[seed_mask[members]]

        if members.size < min_cluster_size:
            continue
        if core_members.size < min_core_size:
            continue

        components.append({
            "members": members,
            "core_members": core_members,
            "n": members.size,
            "n_core": core_members.size,
        })

    components.sort(key=lambda d: d["n"], reverse=True)

    for cid, comp in enumerate(components, start=1):
        labels[comp["members"]] = cid

    return labels


def build_cluster_stats(
    labels,
    core_mask,
    positions_wrapped,
    box,
    P3D,
    zeta_cg,
    P_local_for_score,
    zeta_local,
    score,
    kind,
):
    """Build per-cluster member/core/halo index lists and scalar statistics."""
    labels = np.asarray(labels, dtype=int)
    core_mask = np.asarray(core_mask, dtype=bool)
    clusters = []

    def _mean_or_nan(a):
        a = np.asarray(a, dtype=float)
        finite = np.isfinite(a)
        return float(np.mean(a[finite])) if np.any(finite) else float("nan")

    def _median_or_nan(a):
        a = np.asarray(a, dtype=float)
        finite = np.isfinite(a)
        return float(np.median(a[finite])) if np.any(finite) else float("nan")

    def _max_or_nan(a):
        a = np.asarray(a, dtype=float)
        finite = np.isfinite(a)
        return float(np.max(a[finite])) if np.any(finite) else float("nan")

    def _min_or_nan(a):
        a = np.asarray(a, dtype=float)
        finite = np.isfinite(a)
        return float(np.min(a[finite])) if np.any(finite) else float("nan")

    for cid in range(1, int(labels.max()) + 1):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            continue

        core_indices = members[core_mask[members]]
        halo_indices = np.setdiff1d(members, core_indices, assume_unique=False)
        pos_m = positions_wrapped[members]
        centroid = pbc_centroid(pos_m, box)
        rg = pbc_radius_of_gyration(pos_m, centroid, box)

        clusters.append({
            "kind": kind,
            "cluster_id": int(cid),
            "n_members": int(members.size),
            "n_core": int(core_indices.size),
            "n_halo": int(halo_indices.size),
            "member_indices": members,
            "core_indices": core_indices,
            "halo_indices": halo_indices,
            "centroid": centroid,
            "rg": rg,
            "mean_P3D": _mean_or_nan(P3D[members]),
            "median_P3D": _median_or_nan(P3D[members]),
            "mean_zeta_cg": _mean_or_nan(zeta_cg[members]),
            "median_zeta_cg": _median_or_nan(zeta_cg[members]),
            "mean_local_P_for_score": _mean_or_nan(P_local_for_score[members]),
            "mean_local_zeta": _mean_or_nan(zeta_local[members]),
            "mean_score": _mean_or_nan(score[members]),
            "max_score": _max_or_nan(score[members]),
            "min_score": _min_or_nan(score[members]),
        })

    clusters.sort(key=lambda d: d["n_members"], reverse=True)
    return clusters


def extract_joint_propensity_structure_domains(
    positions,
    P3D,
    zeta_cg,
    box,
    local_length=3.0,
    local_cutoff=6.0,
    r_connect=3.6,
    alpha=1.0,
    p_transform="auto",
    zscore_method="robust",
    mode="score",
    slow_seed_pct=90,
    slow_grow_pct=75,
    fast_seed_pct=90,
    fast_grow_pct=75,
    slow_p_seed_pct=15,
    slow_p_grow_pct=30,
    slow_z_seed_pct=80,
    slow_z_grow_pct=65,
    fast_p_seed_pct=85,
    fast_p_grow_pct=70,
    fast_z_seed_pct=20,
    fast_z_grow_pct=35,
    min_cluster_size=10,
    min_core_size=3,
    resolve_overlap=True,
):
    """
    Extract slow-structured and fast-disordered domains from positions, P3D,
    and zeta_cg.

    mode="score":
        slow_score = Z(local zeta) - alpha * Z(local P)
        fast_score = Z(local P) - alpha * Z(local zeta)

    mode="double":
        use explicit low/high P and high/low zeta constraints.
    """
    positions = np.asarray(positions, dtype=float)
    P3D = np.asarray(P3D, dtype=float)
    zeta_cg = np.asarray(zeta_cg, dtype=float)
    cell_matrix_from_box(box)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")

    n = positions.shape[0]
    if P3D.shape[0] != n or zeta_cg.shape[0] != n:
        raise ValueError("P3D and zeta_cg must have the same length as positions.")

    pos_wrapped = wrap_positions_orthorhombic(positions, box)

    P_for_score, p_transform_used = transform_propensity(P3D, mode=p_transform)

    P_local_for_score = local_kernel_average(
        pos_wrapped, P_for_score, box, length=local_length, cutoff=local_cutoff
    )
    zeta_local = local_kernel_average(
        pos_wrapped, zeta_cg, box, length=local_length, cutoff=local_cutoff
    )
    P_local_raw = local_kernel_average(
        pos_wrapped, P3D, box, length=local_length, cutoff=local_cutoff
    )

    Pz = robust_zscore(P_local_for_score, method=zscore_method)
    Zz = robust_zscore(zeta_local, method=zscore_method)

    valid = (
        np.all(np.isfinite(pos_wrapped), axis=1)
        & np.isfinite(P3D)
        & np.isfinite(zeta_cg)
        & np.isfinite(Pz)
        & np.isfinite(Zz)
    )
    if not np.any(valid):
        raise ValueError("No valid particles after filtering finite positions/P3D/zeta_cg.")

    slow_score = Zz - alpha * Pz
    fast_score = Pz - alpha * Zz

    thresholds = {
        "mode": mode,
        "p_transform_used": p_transform_used,
        "alpha": alpha,
        "local_length": local_length,
        "local_cutoff": local_cutoff,
        "r_connect": r_connect,
    }

    if mode == "score":
        slow_seed_thr = np.nanpercentile(slow_score[valid], slow_seed_pct)
        slow_grow_thr = np.nanpercentile(slow_score[valid], slow_grow_pct)
        fast_seed_thr = np.nanpercentile(fast_score[valid], fast_seed_pct)
        fast_grow_thr = np.nanpercentile(fast_score[valid], fast_grow_pct)

        slow_seed = valid & (slow_score >= slow_seed_thr)
        slow_grow = valid & (slow_score >= slow_grow_thr)
        fast_seed = valid & (fast_score >= fast_seed_thr)
        fast_grow = valid & (fast_score >= fast_grow_thr)

        thresholds.update({
            "slow_seed_pct": slow_seed_pct,
            "slow_grow_pct": slow_grow_pct,
            "fast_seed_pct": fast_seed_pct,
            "fast_grow_pct": fast_grow_pct,
            "slow_seed_thr": float(slow_seed_thr),
            "slow_grow_thr": float(slow_grow_thr),
            "fast_seed_thr": float(fast_seed_thr),
            "fast_grow_thr": float(fast_grow_thr),
        })

    elif mode == "double":
        p_slow_seed_thr = np.nanpercentile(Pz[valid], slow_p_seed_pct)
        p_slow_grow_thr = np.nanpercentile(Pz[valid], slow_p_grow_pct)
        z_slow_seed_thr = np.nanpercentile(Zz[valid], slow_z_seed_pct)
        z_slow_grow_thr = np.nanpercentile(Zz[valid], slow_z_grow_pct)

        p_fast_seed_thr = np.nanpercentile(Pz[valid], fast_p_seed_pct)
        p_fast_grow_thr = np.nanpercentile(Pz[valid], fast_p_grow_pct)
        z_fast_seed_thr = np.nanpercentile(Zz[valid], fast_z_seed_pct)
        z_fast_grow_thr = np.nanpercentile(Zz[valid], fast_z_grow_pct)

        slow_seed = valid & (Pz <= p_slow_seed_thr) & (Zz >= z_slow_seed_thr)
        slow_grow = valid & (Pz <= p_slow_grow_thr) & (Zz >= z_slow_grow_thr)
        fast_seed = valid & (Pz >= p_fast_seed_thr) & (Zz <= z_fast_seed_thr)
        fast_grow = valid & (Pz >= p_fast_grow_thr) & (Zz <= z_fast_grow_thr)

        thresholds.update({
            "slow_p_seed_pct": slow_p_seed_pct,
            "slow_p_grow_pct": slow_p_grow_pct,
            "slow_z_seed_pct": slow_z_seed_pct,
            "slow_z_grow_pct": slow_z_grow_pct,
            "fast_p_seed_pct": fast_p_seed_pct,
            "fast_p_grow_pct": fast_p_grow_pct,
            "fast_z_seed_pct": fast_z_seed_pct,
            "fast_z_grow_pct": fast_z_grow_pct,
            "p_slow_seed_thr": float(p_slow_seed_thr),
            "p_slow_grow_thr": float(p_slow_grow_thr),
            "z_slow_seed_thr": float(z_slow_seed_thr),
            "z_slow_grow_thr": float(z_slow_grow_thr),
            "p_fast_seed_thr": float(p_fast_seed_thr),
            "p_fast_grow_thr": float(p_fast_grow_thr),
            "z_fast_seed_thr": float(z_fast_seed_thr),
            "z_fast_grow_thr": float(z_fast_grow_thr),
        })

    else:
        raise ValueError("mode must be 'score' or 'double'.")

    slow_labels = component_labels_from_seed_growth(
        pos_wrapped, box, slow_grow, slow_seed, r_connect, min_cluster_size, min_core_size
    )
    fast_labels = component_labels_from_seed_growth(
        pos_wrapped, box, fast_grow, fast_seed, r_connect, min_cluster_size, min_core_size
    )

    overlap = (slow_labels > 0) & (fast_labels > 0)

    if resolve_overlap and np.any(overlap):
        prefer_slow = np.zeros(n, dtype=bool)
        prefer_slow[overlap] = slow_score[overlap] >= fast_score[overlap]

        slow_grow = slow_grow & (~overlap | prefer_slow)
        slow_seed = slow_seed & slow_grow
        fast_grow = fast_grow & (~overlap | ~prefer_slow)
        fast_seed = fast_seed & fast_grow

        slow_labels = component_labels_from_seed_growth(
            pos_wrapped, box, slow_grow, slow_seed, r_connect, min_cluster_size, min_core_size
        )
        fast_labels = component_labels_from_seed_growth(
            pos_wrapped, box, fast_grow, fast_seed, r_connect, min_cluster_size, min_core_size
        )

    slow_core = (slow_labels > 0) & slow_seed
    slow_halo = (slow_labels > 0) & (~slow_core)
    fast_core = (fast_labels > 0) & fast_seed
    fast_halo = (fast_labels > 0) & (~fast_core)

    slow_clusters = build_cluster_stats(
        slow_labels, slow_core, pos_wrapped, box, P3D, zeta_cg,
        P_local_for_score, zeta_local, slow_score, kind="slow_structured"
    )
    fast_clusters = build_cluster_stats(
        fast_labels, fast_core, pos_wrapped, box, P3D, zeta_cg,
        P_local_for_score, zeta_local, fast_score, kind="fast_disordered"
    )

    return {
        "positions_wrapped": pos_wrapped,
        "box": box,
        "P3D": P3D,
        "zeta_cg": zeta_cg,
        "local": {
            "P_raw": P_local_raw,
            "P_for_score": P_local_for_score,
            "zeta": zeta_local,
            "Pz": Pz,
            "Zz": Zz,
        },
        "scores": {"slow": slow_score, "fast": fast_score},
        "masks": {
            "valid": valid,
            "slow_seed_raw": slow_seed,
            "slow_grow_raw": slow_grow,
            "fast_seed_raw": fast_seed,
            "fast_grow_raw": fast_grow,
            "slow_core": slow_core,
            "slow_halo": slow_halo,
            "slow_all": slow_labels > 0,
            "fast_core": fast_core,
            "fast_halo": fast_halo,
            "fast_all": fast_labels > 0,
        },
        "labels": {"slow": slow_labels, "fast": fast_labels},
        "clusters": {"slow": slow_clusters, "fast": fast_clusters},
        "thresholds": thresholds,
    }




def extract_zeta_structure_domains(
    positions,
    zeta_cg,
    box,
    r_connect=3.6,
    high_seed_pct=90,
    high_grow_pct=75,
    low_seed_pct=10,
    low_grow_pct=25,
    min_cluster_size=10,
    min_core_size=3,
    resolve_overlap=True,
):
    """
    Extract structural domains directly from particle-level zeta_cg values.

    This is the recommended route when zeta_cg has already been computed as a
    local coarse-grained structural order parameter. No additional spatial
    convolution or local averaging is applied here.

    Definitions
    -----------
    high-zeta / slow-structured domain:
        core: zeta_cg >= percentile(zeta_cg, high_seed_pct)
        halo: connected grow region satisfying
              zeta_cg >= percentile(zeta_cg, high_grow_pct)

    low-zeta / fast-disordered domain:
        core: zeta_cg <= percentile(zeta_cg, low_seed_pct)
        halo: connected grow region satisfying
              zeta_cg <= percentile(zeta_cg, low_grow_pct)

    Notes
    -----
    high_seed_pct should normally be larger than high_grow_pct.
    low_seed_pct should normally be smaller than low_grow_pct.
    """
    positions = np.asarray(positions, dtype=float)
    zeta_cg = np.asarray(zeta_cg, dtype=float)
    cell_matrix_from_box(box)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")

    n = positions.shape[0]
    if zeta_cg.shape[0] != n:
        raise ValueError("zeta_cg must have the same length as positions.")

    if high_seed_pct < high_grow_pct:
        raise ValueError("For high-zeta domains, require high_seed_pct >= high_grow_pct.")
    if low_seed_pct > low_grow_pct:
        raise ValueError("For low-zeta domains, require low_seed_pct <= low_grow_pct.")

    pos_wrapped = wrap_positions_orthorhombic(positions, box)
    valid = np.all(np.isfinite(pos_wrapped), axis=1) & np.isfinite(zeta_cg)

    if not np.any(valid):
        raise ValueError("No valid particles after filtering finite positions/zeta_cg.")

    high_seed_thr = np.nanpercentile(zeta_cg[valid], high_seed_pct)
    high_grow_thr = np.nanpercentile(zeta_cg[valid], high_grow_pct)
    low_seed_thr = np.nanpercentile(zeta_cg[valid], low_seed_pct)
    low_grow_thr = np.nanpercentile(zeta_cg[valid], low_grow_pct)

    high_seed = valid & (zeta_cg >= high_seed_thr)
    high_grow = valid & (zeta_cg >= high_grow_thr)
    low_seed = valid & (zeta_cg <= low_seed_thr)
    low_grow = valid & (zeta_cg <= low_grow_thr)

    # These two branches are normally disjoint by construction, but keep a
    # deterministic resolution for pathological percentile choices.
    overlap = high_grow & low_grow
    if resolve_overlap and np.any(overlap):
        z_mid = 0.5 * (high_grow_thr + low_grow_thr)
        prefer_high = zeta_cg >= z_mid
        high_grow = high_grow & (~overlap | prefer_high)
        high_seed = high_seed & high_grow
        low_grow = low_grow & (~overlap | ~prefer_high)
        low_seed = low_seed & low_grow

    high_labels = component_labels_from_seed_growth(
        pos_wrapped, box, high_grow, high_seed, r_connect, min_cluster_size, min_core_size
    )
    low_labels = component_labels_from_seed_growth(
        pos_wrapped, box, low_grow, low_seed, r_connect, min_cluster_size, min_core_size
    )

    high_core = (high_labels > 0) & high_seed
    high_halo = (high_labels > 0) & (~high_core)
    low_core = (low_labels > 0) & low_seed
    low_halo = (low_labels > 0) & (~low_core)

    nan_field = np.full(n, np.nan, dtype=float)

    high_clusters = build_cluster_stats(
        high_labels, high_core, pos_wrapped, box,
        nan_field, zeta_cg,
        nan_field, zeta_cg, zeta_cg,
        kind="high_zeta_structured",
    )
    low_clusters = build_cluster_stats(
        low_labels, low_core, pos_wrapped, box,
        nan_field, zeta_cg,
        nan_field, zeta_cg, -zeta_cg,
        kind="low_zeta_disordered",
    )

    return {
        "title": "Direct zeta_cg structural-domain extraction",
        "extraction_type": "zeta_cg_direct",
        "positions_wrapped": pos_wrapped,
        "box": box,
        "P3D": nan_field,
        "zeta_cg": zeta_cg,
        "local": {
            "P_raw": nan_field,
            "P_for_score": nan_field,
            "zeta": zeta_cg,
            "Pz": nan_field,
            "Zz": robust_zscore(zeta_cg, method="robust"),
        },
        "scores": {"slow": zeta_cg, "fast": -zeta_cg},
        "masks": {
            "valid": valid,
            "slow_seed_raw": high_seed,
            "slow_grow_raw": high_grow,
            "fast_seed_raw": low_seed,
            "fast_grow_raw": low_grow,
            "slow_core": high_core,
            "slow_halo": high_halo,
            "slow_all": high_labels > 0,
            "fast_core": low_core,
            "fast_halo": low_halo,
            "fast_all": low_labels > 0,
        },
        "labels": {"slow": high_labels, "fast": low_labels},
        "clusters": {"slow": high_clusters, "fast": low_clusters},
        "thresholds": {
            "mode": "zeta_cg_direct",
            "r_connect": r_connect,
            "high_seed_pct": high_seed_pct,
            "high_grow_pct": high_grow_pct,
            "low_seed_pct": low_seed_pct,
            "low_grow_pct": low_grow_pct,
            "high_seed_thr": float(high_seed_thr),
            "high_grow_thr": float(high_grow_thr),
            "low_seed_thr": float(low_seed_thr),
            "low_grow_thr": float(low_grow_thr),
            "min_cluster_size": min_cluster_size,
            "min_core_size": min_core_size,
            "no_extra_convolution": True,
        },
    }

def print_domain_summary(result, max_clusters=10):
    """Print a compact table of extracted domains."""
    print("=" * 100)
    print(result.get("title", "Domain extraction"))
    print("-" * 100)
    for k, v in result["thresholds"].items():
        if isinstance(v, float):
            print(f"{k:>24s}: {v:.6g}")
        else:
            print(f"{k:>24s}: {v}")

    for kind in ("slow", "fast"):
        clusters = result["clusters"][kind]
        print("-" * 100)
        print(f"{kind.upper()} domains: {len(clusters)}")
        print(
            f"{'id':>4s} | {'N':>6s} | {'core':>6s} | {'halo':>6s} | "
            f"{'<P3D>':>12s} | {'<zeta>':>10s} | {'Rg(Å)':>8s} | {'centroid(Å)':>28s}"
        )
        print("-" * 100)
        for c in clusters[:max_clusters]:
            cen = c["centroid"]
            print(
                f"{c['cluster_id']:4d} | {c['n_members']:6d} | "
                f"{c['n_core']:6d} | {c['n_halo']:6d} | "
                f"{c['mean_P3D']:12.5g} | {c['mean_zeta_cg']:10.5g} | "
                f"{c['rg']:8.3f} | ({cen[0]:6.2f}, {cen[1]:6.2f}, {cen[2]:6.2f})"
            )
    print("=" * 100)
