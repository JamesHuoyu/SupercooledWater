import numpy as np

from .geometry import minimum_image, cell_matrix_from_box


def flatten_frame_clusters(results, kinds=("slow", "fast")):
    """
    Flatten framewise extraction results into a list of cluster records.

    Each record stores:
        frame, frame_step, kind, local_rank, cluster_id, member/core/halo indices,
        centroid, Rg, N, and scalar summary statistics.
    """
    records = []

    for step, result in enumerate(results):
        frame = int(result.get("frame", step))
        box = np.asarray(result["box"], dtype=float)
        cell_matrix_from_box(box)

        for kind in kinds:
            for rank, c in enumerate(result["clusters"].get(kind, []), start=1):
                rec = {
                    "frame": frame,
                    "frame_step": step,
                    "kind": kind,
                    "local_rank": rank,
                    "cluster_id": int(c["cluster_id"]),
                    "record_id": f"{kind}:f{frame}:c{int(c['cluster_id'])}",
                    "member_indices": np.asarray(c["member_indices"], dtype=int),
                    "core_indices": np.asarray(c["core_indices"], dtype=int),
                    "halo_indices": np.asarray(c["halo_indices"], dtype=int),
                    "centroid": np.asarray(c["centroid"], dtype=float),
                    "box": box,
                    "n_members": int(c["n_members"]),
                    "n_core": int(c["n_core"]),
                    "n_halo": int(c["n_halo"]),
                    "rg": float(c["rg"]),
                }

                for key in (
                    "mean_P3D",
                    "median_P3D",
                    "mean_zeta_cg",
                    "median_zeta_cg",
                    "mean_local_P_for_score",
                    "mean_local_zeta",
                    "mean_score",
                    "max_score",
                    "min_score",
                ):
                    if key in c:
                        rec[key] = c[key]

                records.append(rec)

    return records


def _overlap_metrics(a_indices, b_indices):
    a = np.asarray(a_indices, dtype=int)
    b = np.asarray(b_indices, dtype=int)

    if a.size == 0 or b.size == 0:
        return {"n_overlap": 0, "jaccard": 0.0, "containment": 0.0}

    inter = np.intersect1d(a, b, assume_unique=False).size
    union = a.size + b.size - inter

    jaccard = inter / union if union > 0 else 0.0
    containment = inter / min(a.size, b.size) if min(a.size, b.size) > 0 else 0.0

    return {
        "n_overlap": int(inter),
        "jaccard": float(jaccard),
        "containment": float(containment),
    }


def _centroid_distance(a_centroid, b_centroid, box):
    delta = minimum_image(np.asarray(b_centroid)[None, :] - np.asarray(a_centroid)[None, :], box)[0]
    return float(np.linalg.norm(delta))


def _match_score(
    previous,
    current,
    min_overlap=3,
    min_jaccard=0.15,
    min_containment=0.35,
    max_centroid_distance=8.0,
    jaccard_weight=1.0,
    containment_weight=0.5,
    distance_weight=0.25,
):
    if previous["kind"] != current["kind"]:
        return None

    metrics = _overlap_metrics(previous["member_indices"], current["member_indices"])
    dist = _centroid_distance(previous["centroid"], current["centroid"], previous["box"])

    if max_centroid_distance is not None and dist > max_centroid_distance:
        return None

    passes_overlap = (
        metrics["n_overlap"] >= min_overlap
        or metrics["jaccard"] >= min_jaccard
        or metrics["containment"] >= min_containment
    )
    if not passes_overlap:
        return None

    if max_centroid_distance is None or max_centroid_distance <= 0:
        distance_penalty = 0.0
    else:
        distance_penalty = dist / max_centroid_distance

    score = (
        jaccard_weight * metrics["jaccard"]
        + containment_weight * metrics["containment"]
        - distance_weight * distance_penalty
    )

    metrics["centroid_distance"] = dist
    metrics["score"] = float(score)
    return metrics


def track_clusters_over_time(
    results,
    kinds=("slow", "fast"),
    min_overlap=3,
    min_jaccard=0.15,
    min_containment=0.35,
    max_centroid_distance=8.0,
    max_gap_steps=0,
):
    """
    Link framewise clusters into temporal tracks.

    Matching rule
    -------------
    A cluster at the current frame is linked to an active track when it has
    sufficient member overlap/Jaccard/containment and its PBC centroid distance
    is not too large. Matching is one-to-one within each frame and kind.

    Parameters
    ----------
    results : list[dict]
        Output of extract_domains_over_frames(...) or a manually assembled list.
    kinds : tuple
        Usually ("slow", "fast").
    min_overlap, min_jaccard, min_containment
        Matching thresholds. The overlap condition is OR-based: passing any one
        of these can make the candidate valid.
    max_centroid_distance : float or None
        PBC centroid-distance cutoff in Å.
    max_gap_steps : int
        Number of missing sampled frames a track may survive before being closed.

    Returns
    -------
    tracks : list[dict]
        Each track has track_id, kind, observations, start/end frame, lifetime,
        and gap information.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("scipy is required for track_clusters_over_time.") from exc

    records = flatten_frame_clusters(results, kinds=kinds)
    by_step = {}
    for rec in records:
        by_step.setdefault(rec["frame_step"], []).append(rec)

    tracks = []
    active = []
    next_id = {kind: 1 for kind in kinds}

    n_steps = len(results)

    for step in range(n_steps):
        current_all = by_step.get(step, [])

        # Close tracks that have exceeded the allowed gap.
        still_active = []
        for tr in active:
            if step - tr["last_step"] <= max_gap_steps + 1:
                still_active.append(tr)
            else:
                tr["status"] = "closed"
                tracks.append(tr)
        active = still_active

        for kind in kinds:
            current = [r for r in current_all if r["kind"] == kind]
            active_kind = [tr for tr in active if tr["kind"] == kind]

            if not current:
                continue

            if not active_kind:
                for rec in current:
                    tid = f"{kind}-T{next_id[kind]:04d}"
                    next_id[kind] += 1
                    active.append({
                        "track_id": tid,
                        "kind": kind,
                        "observations": [rec],
                        "last_record": rec,
                        "last_step": step,
                        "status": "active",
                        "match_history": [],
                    })
                continue

            score_mat = np.full((len(active_kind), len(current)), -np.inf, dtype=float)
            metric_mat = [[None for _ in current] for _ in active_kind]

            for i, tr in enumerate(active_kind):
                prev = tr["last_record"]
                for j, rec in enumerate(current):
                    metrics = _match_score(
                        previous=prev,
                        current=rec,
                        min_overlap=min_overlap,
                        min_jaccard=min_jaccard,
                        min_containment=min_containment,
                        max_centroid_distance=max_centroid_distance,
                    )
                    if metrics is not None:
                        score_mat[i, j] = metrics["score"]
                        metric_mat[i][j] = metrics

            # Hungarian minimizes cost, so use -score. Invalid matches get huge cost.
            valid_any = np.isfinite(score_mat)
            assigned_current = set()

            if np.any(valid_any):
                cost = np.where(np.isfinite(score_mat), -score_mat, 1e9)
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_track_ids = set()
                for i, j in zip(row_ind, col_ind):
                    if not np.isfinite(score_mat[i, j]):
                        continue

                    tr = active_kind[i]
                    rec = current[j]
                    metrics = metric_mat[i][j]

                    tr["observations"].append(rec)
                    tr["last_record"] = rec
                    tr["last_step"] = step
                    tr["match_history"].append({
                        "from_record_id": tr["observations"][-2]["record_id"],
                        "to_record_id": rec["record_id"],
                        **metrics,
                    })

                    assigned_current.add(j)
                    matched_track_ids.add(tr["track_id"])

            # Unmatched current clusters start new tracks.
            for j, rec in enumerate(current):
                if j in assigned_current:
                    continue
                tid = f"{kind}-T{next_id[kind]:04d}"
                next_id[kind] += 1
                active.append({
                    "track_id": tid,
                    "kind": kind,
                    "observations": [rec],
                    "last_record": rec,
                    "last_step": step,
                    "status": "active",
                    "match_history": [],
                })

        # Remove duplicate active-track references caused by active_kind list updates.
        dedup = {}
        for tr in active:
            dedup[tr["track_id"]] = tr
        active = list(dedup.values())

    for tr in active:
        tr["status"] = "closed"
        tracks.append(tr)

    # Attach summary metadata.
    for tr in tracks:
        obs = tr["observations"]
        frames = [o["frame"] for o in obs]
        steps = [o["frame_step"] for o in obs]
        tr["start_frame"] = int(min(frames))
        tr["end_frame"] = int(max(frames))
        tr["start_step"] = int(min(steps))
        tr["end_step"] = int(max(steps))
        tr["n_observations"] = int(len(obs))
        tr["duration_steps"] = int(max(steps) - min(steps) + 1)
        tr["n_missing_steps"] = int(tr["duration_steps"] - len(obs))
        tr["max_n_members"] = int(max(o["n_members"] for o in obs))
        tr["mean_n_members"] = float(np.mean([o["n_members"] for o in obs]))
        tr["mean_rg"] = float(np.mean([o["rg"] for o in obs]))

    tracks.sort(key=lambda t: (t["kind"], t["start_step"], -t["max_n_members"]))
    return tracks


def summarize_tracks(tracks, dt_per_sample=None, as_dataframe=True):
    """
    Summarize cluster tracks.

    If dt_per_sample is provided, lifetime is additionally reported in the
    corresponding time unit, e.g. ps.
    """
    rows = []
    for tr in tracks:
        row = {
            "track_id": tr["track_id"],
            "kind": tr["kind"],
            "start_frame": tr["start_frame"],
            "end_frame": tr["end_frame"],
            "n_observations": tr["n_observations"],
            "duration_steps": tr["duration_steps"],
            "n_missing_steps": tr["n_missing_steps"],
            "max_n_members": tr["max_n_members"],
            "mean_n_members": tr["mean_n_members"],
            "mean_rg": tr["mean_rg"],
        }
        if dt_per_sample is not None:
            row["duration_time"] = tr["duration_steps"] * dt_per_sample
        rows.append(row)

    if not as_dataframe:
        return rows

    try:
        import pandas as pd
    except ImportError:
        return rows

    return pd.DataFrame(rows)


def _values_for_frame(frame_values, frame, frame_step):
    if isinstance(frame_values, dict):
        return np.asarray(frame_values[frame], dtype=float)

    arr = np.asarray(frame_values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[frame_step]
    raise ValueError("frame_values must be shape (N,), shape (n_frames,N), or dict[frame]->array.")


def _reduce_values(values, reducer):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan

    if reducer == "mean":
        return float(np.mean(values))
    if reducer == "sum":
        return float(np.sum(values))
    if reducer == "median":
        return float(np.median(values))
    if reducer == "abs_mean":
        return float(np.mean(np.abs(values)))
    if reducer == "abs_sum":
        return float(np.sum(np.abs(values)))
    if reducer == "rms":
        return float(np.sqrt(np.mean(values * values)))

    raise ValueError("reducer must be 'mean', 'sum', 'median', 'abs_mean', 'abs_sum', or 'rms'.")


def track_observable(
    tracks,
    frame_values,
    index_set="member_indices",
    reducer="sum",
    as_dataframe=True,
):
    """
    Evaluate a per-molecule observable along cluster tracks.

    This is the stress-analysis hook. For example, pass molecular pxy[frame, mol]
    and reducer="sum" to estimate the cluster's instantaneous stress contribution,
    or reducer="abs_sum"/"rms" to quantify stress amplitude.

    Parameters
    ----------
    tracks : list[dict]
        Output of track_clusters_over_time.
    frame_values : ndarray or dict
        Per-molecule observable. Supported forms:
            shape (N,), fixed for all frames
            shape (n_sampled_frames, N)
            dict[frame] -> shape (N,)
    index_set : {"member_indices", "core_indices", "halo_indices"}
        Which cluster particles to use.
    reducer : {"mean", "sum", "median", "abs_mean", "abs_sum", "rms"}
        How to reduce per-particle values.

    Returns
    -------
    rows : pandas.DataFrame or list[dict]
    """
    rows = []

    for tr in tracks:
        for obs in tr["observations"]:
            idx = np.asarray(obs[index_set], dtype=int)
            vals = _values_for_frame(frame_values, obs["frame"], obs["frame_step"])
            reduced = _reduce_values(vals[idx], reducer)

            rows.append({
                "track_id": tr["track_id"],
                "kind": tr["kind"],
                "frame": obs["frame"],
                "frame_step": obs["frame_step"],
                "n_particles": int(idx.size),
                "observable": reduced,
                "index_set": index_set,
                "reducer": reducer,
            })

    if not as_dataframe:
        return rows

    try:
        import pandas as pd
    except ImportError:
        return rows

    return pd.DataFrame(rows)
