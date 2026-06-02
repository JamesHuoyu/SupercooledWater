import numpy as np

from .geometry import wrap_positions_orthorhombic, cell_matrix_from_box
from .domain_extraction import (
    extract_joint_propensity_structure_domains,
    extract_zeta_structure_domains,
)


def frame_payload_from_zop(zop, frame, box=None, field="zeta_cg"):
    """
    Extract positions and zeta-like scalar values from a ZOP object for one frame.

    Parameters
    ----------
    zop
        Your ZOP analysis object. It should provide spatial_zeta_map(frame) and,
        for zeta_cg, spatial_zeta_cg_map(frame).
    frame : int
        Trajectory frame index.
    box : array-like or None
        Box representation: [Lx,Ly,Lz], MDAnalysis dimensions [lx,ly,lz,alpha,beta,gamma],
        or a 3x3 triclinic cell matrix. If None, the function tries to read box from
        the returned ZOP dictionary; otherwise it leaves box as None.
    field : {"zeta", "zeta_cg"}
        Which scalar field to extract.

    Returns
    -------
    payload : dict
        {"frame", "positions", "positions_wrapped", "zeta", "box"}
    """
    if field == "zeta_cg":
        data = zop.spatial_zeta_cg_map(frame)
        values = np.asarray(data["zeta_cg"], dtype=float)
    elif field == "zeta":
        data = zop.spatial_zeta_map(frame)
        values = np.asarray(data["zeta"], dtype=float)
    else:
        raise ValueError("field must be 'zeta' or 'zeta_cg'.")

    positions = np.asarray(data["positions"], dtype=float)

    if box is None:
        box = data.get("box", None)

    payload = {
        "frame": int(frame),
        "positions": positions,
        "zeta": values,
        "box": None if box is None else np.asarray(box, dtype=float),
    }

    if box is not None:
        cell_matrix_from_box(box)
        payload["positions_wrapped"] = wrap_positions_orthorhombic(positions, box)
    else:
        payload["positions_wrapped"] = positions

    return payload


def _frame_values(values, frame, frame_index=None):
    """
    Return per-particle values for a frame. Accepts either:
    - a fixed vector with shape (N,)
    - a frame matrix with shape (n_frames, N)
    - a dict mapping frame -> vector
    """
    if isinstance(values, dict):
        return np.asarray(values[frame], dtype=float)

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if frame_index is None:
            return arr[frame]
        return arr[frame_index]
    raise ValueError("values must be shape (N,), shape (n_frames,N), or dict[frame]->array.")


def _box_for_frame(box, frame, frame_index=None):
    """
    Return the simulation cell for one trajectory frame.

    ``box`` may be one of:
      - a fixed box: shape (3,), shape (6,), or shape (3, 3)
      - a callable: box(frame) -> box
      - a dict keyed by the actual trajectory frame number
      - an array/list aligned with ``frames``: shape (n_frames, 3),
        shape (n_frames, 6), or shape (n_frames, 3, 3)

    For sheared LAMMPS/MDAnalysis trajectories, prefer a callable such as
    ``lambda frame: (u.trajectory[frame], u.dimensions.copy())[1]`` so that
    each sampled frame uses its own triclinic dimensions.
    """
    if box is None:
        return None

    if callable(box):
        out = box(int(frame))
        return None if out is None else np.asarray(out, dtype=float)

    if isinstance(box, dict):
        out = box[int(frame)]
        return None if out is None else np.asarray(out, dtype=float)

    arr = np.asarray(box, dtype=float)

    # Fixed orthorhombic/MDAnalysis dimensions or fixed 3x3 cell matrix.
    if arr.ndim == 1:
        return arr
    if arr.shape == (3, 3):
        return arr

    # Frame-aligned boxes. Use frame_index for sparse sampled ranges such as
    # range(4000, 4800, 10). Falling back to frame allows dense arrays indexed
    # by the original trajectory frame number.
    if arr.ndim == 2 and arr.shape[1] in (3, 6):
        idx = int(frame_index) if frame_index is not None else int(frame)
        return arr[idx]
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        idx = int(frame_index) if frame_index is not None else int(frame)
        return arr[idx]

    raise ValueError(
        "box must be fixed shape (3,), (6,), (3,3), a callable, a dict, "
        "or a frame-aligned array with shape (n_frames,3), (n_frames,6), "
        "or (n_frames,3,3)."
    )


def extract_domains_for_frame(
    zop,
    frame,
    P3D,
    box,
    frame_index=None,
    zeta_field="zeta_cg",
    **domain_kwargs,
):
    """
    Convenience wrapper: read one frame from ZOP and run domain extraction.

    P3D may be a fixed vector, a frame matrix, or a dict keyed by frame.
    """
    box_frame = _box_for_frame(box, frame, frame_index=frame_index)
    payload = frame_payload_from_zop(zop, frame, box=box_frame, field=zeta_field)
    box_frame = payload["box"] if box_frame is None else box_frame
    if box_frame is None:
        raise ValueError("A box is required for PBC-aware domain extraction.")

    p_frame = _frame_values(P3D, frame, frame_index=frame_index)

    result = extract_joint_propensity_structure_domains(
        positions=payload["positions"],
        P3D=p_frame,
        zeta_cg=payload["zeta"],
        box=box_frame,
        **domain_kwargs,
    )
    result["frame"] = int(frame)
    return result


def extract_domains_over_frames(
    zop,
    frames,
    P3D,
    box,
    zeta_field="zeta_cg",
    **domain_kwargs,
):
    """
    Run the same domain-extraction protocol over multiple frames.

    Returns
    -------
    results : list[dict]
        One domain-extraction result per frame. Each result has result["frame"].
    """
    frames = list(frames)
    results = []

    for k, frame in enumerate(frames):
        results.append(
            extract_domains_for_frame(
                zop=zop,
                frame=int(frame),
                P3D=P3D,
                box=box,
                frame_index=k,
                zeta_field=zeta_field,
                **domain_kwargs,
            )
        )

    return results



def extract_zeta_domains_for_frame(
    zop,
    frame,
    box,
    frame_index=None,
    zeta_field="zeta_cg",
    **domain_kwargs,
):
    """
    Convenience wrapper: read one frame from ZOP and extract domains directly
    from zeta_cg, without any additional convolution/local averaging.
    """
    box_frame = _box_for_frame(box, frame, frame_index=frame_index)
    payload = frame_payload_from_zop(zop, frame, box=box_frame, field=zeta_field)
    box_frame = payload["box"] if box_frame is None else box_frame
    if box_frame is None:
        raise ValueError("A box is required for PBC-aware zeta_cg domain extraction.")

    result = extract_zeta_structure_domains(
        positions=payload["positions"],
        zeta_cg=payload["zeta"],
        box=box_frame,
        **domain_kwargs,
    )
    result["frame"] = int(frame)
    return result


def extract_zeta_domains_over_frames(
    zop,
    frames,
    box,
    zeta_field="zeta_cg",
    **domain_kwargs,
):
    """
    Run direct zeta_cg domain extraction over multiple frames.
    """
    frames = list(frames)
    results = []

    for k, frame in enumerate(frames):
        results.append(
            extract_zeta_domains_for_frame(
                zop=zop,
                frame=int(frame),
                box=box,
                frame_index=k,
                zeta_field=zeta_field,
                **domain_kwargs,
            )
        )

    return results
