"""Geometry and PBC utilities for zeta-cluster analysis.

Version 4 uses MDAnalysis' own triclinic PBC machinery for wrapping,
minimum-image vectors, and cutoff neighbour searches.  This avoids the simpler
fractional-coordinate MIC used in v3.
"""

from __future__ import annotations

import numpy as np

try:
    from MDAnalysis.lib import distances as mda_distances
    from MDAnalysis.lib.mdamath import triclinic_vectors, triclinic_box
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "zeta_cluster_toolkit v4 requires MDAnalysis for triclinic PBC operations. "
        "Install MDAnalysis or use an earlier toolkit version."
    ) from exc


# -----------------------------------------------------------------------------
# Box conversion helpers
# -----------------------------------------------------------------------------

def box_to_mda_dimensions(box) -> np.ndarray:
    """
    Return MDAnalysis-style box dimensions ``[lx, ly, lz, alpha, beta, gamma]``.

    Accepted inputs
    ---------------
    - ``(3,)``: orthorhombic lengths ``[Lx, Ly, Lz]``.
    - ``(6,)``: already MDAnalysis-style dimensions.
    - ``(3, 3)``: row-vector triclinic matrix.  For a LAMMPS restricted
      triclinic cell this is typically ``[[lx,0,0], [xy,ly,0], [xz,yz,lz]]``.

    Notes
    -----
    MDAnalysis distance routines require the ``(6,)`` dimensions format at the
    Python API level.  They internally convert triclinic boxes to the flattened
    lower-triangular matrix used by the C/Cython kernels.
    """
    arr = np.asarray(box, dtype=float)

    if arr.shape == (3, 3):
        dims = triclinic_box(arr[0], arr[1], arr[2]).astype(np.float32)
    elif arr.size == 3:
        Lx, Ly, Lz = arr.reshape(-1)[:3]
        dims = np.array([Lx, Ly, Lz, 90.0, 90.0, 90.0], dtype=np.float32)
    elif arr.size == 6:
        dims = arr.reshape(-1)[:6].astype(np.float32)
    else:
        raise ValueError(
            "box must be [Lx,Ly,Lz], [lx,ly,lz,alpha,beta,gamma], "
            "or a 3x3 row-vector triclinic cell matrix."
        )

    if not np.all(np.isfinite(dims)):
        raise ValueError("box contains non-finite values.")
    if np.any(dims[:3] <= 0.0):
        raise ValueError("box lengths must be positive.")
    if np.any(dims[3:] <= 0.0) or np.any(dims[3:] >= 180.0):
        raise ValueError("box angles must lie in the open interval (0, 180).")
    return dims


def cell_matrix_from_box(box) -> np.ndarray:
    """
    Return MDAnalysis' triclinic row-vector matrix for a box.

    This function is retained for visualization/projection utilities.  It uses
    ``MDAnalysis.lib.mdamath.triclinic_vectors`` instead of a local manual
    reconstruction.
    """
    dims = box_to_mda_dimensions(box)
    H = triclinic_vectors(dims, dtype=np.float64)
    if abs(np.linalg.det(H)) < np.finfo(float).eps:
        raise ValueError("box/cell matrix is singular.")
    return H


def lammps_triclinic_matrix(lx, ly, lz, xy=0.0, xz=0.0, yz=0.0) -> np.ndarray:
    """Return a row-vector matrix for a LAMMPS restricted triclinic box."""
    return np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)


def lammps_triclinic_dimensions(lx, ly, lz, xy=0.0, xz=0.0, yz=0.0) -> np.ndarray:
    """Return MDAnalysis dimensions for a LAMMPS restricted triclinic box."""
    return box_to_mda_dimensions(lammps_triclinic_matrix(lx, ly, lz, xy, xz, yz))


def cell_lengths(box) -> np.ndarray:
    """Return ``[lx, ly, lz]`` from MDAnalysis-style dimensions."""
    return box_to_mda_dimensions(box)[:3].astype(float)


def is_orthorhombic_box(box, atol=1.0e-6) -> bool:
    dims = box_to_mda_dimensions(box)
    return bool(np.allclose(dims[3:], 90.0, atol=atol))


# -----------------------------------------------------------------------------
# MDAnalysis-backed PBC operations
# -----------------------------------------------------------------------------

def wrap_positions_pbc(positions, box) -> np.ndarray:
    """Wrap coordinates into the primary unit cell using MDAnalysis.apply_PBC."""
    positions = np.asarray(positions, dtype=np.float32)
    dims = box_to_mda_dimensions(box)
    return mda_distances.apply_PBC(positions, box=dims).astype(float, copy=False)


def wrap_positions_orthorhombic(positions, box) -> np.ndarray:
    """
    Backward-compatible wrapper.  Despite the historical name, this function
    now delegates to MDAnalysis and handles orthorhombic and triclinic boxes.
    """
    return wrap_positions_pbc(positions, box)


def minimum_image(delta, box) -> np.ndarray:
    """
    Apply MDAnalysis' minimum-image convention to displacement vectors.

    For triclinic boxes this calls ``MDAnalysis.lib.distances.minimize_vectors``.
    MDAnalysis' Cython implementation first shifts the vector to within a
    single neighbouring cell and then checks the 27 adjacent triclinic images,
    rather than relying on a single fractional-coordinate rounding step.
    """
    delta = np.asarray(delta, dtype=float)
    original_shape = delta.shape
    vectors = np.ascontiguousarray(delta.reshape(-1, 3), dtype=np.float64)
    dims = box_to_mda_dimensions(box)
    out = mda_distances.minimize_vectors(vectors, dims)
    return out.reshape(original_shape)


def mic_distance(a, b, box) -> np.ndarray:
    """Return MDAnalysis MIC distances between paired coordinates ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dims = box_to_mda_dimensions(box)
    return mda_distances.calc_bonds(a, b, box=dims).astype(float, copy=False)


def unwrap_positions_relative(positions_wrapped, reference_point, box) -> np.ndarray:
    """Unwrap positions around ``reference_point`` using MDAnalysis MIC vectors."""
    positions_wrapped = wrap_positions_pbc(positions_wrapped, box)
    reference_point = np.asarray(reference_point, dtype=float)
    delta = minimum_image(positions_wrapped - reference_point[None, :], box)
    return reference_point[None, :] + delta


def pbc_centroid(positions_wrapped, box, reference_index=0, n_iter=2) -> np.ndarray:
    """
    Compact-cluster centroid under PBC using MDAnalysis MIC unwrapping.

    This replaces the v3 fractional circular-mean centroid.  The algorithm is:
    choose a reference particle, unwrap all members around that reference using
    MDAnalysis MIC vectors, compute the arithmetic centroid, optionally repeat
    around the new centroid, and wrap the final centroid back into the primary
    cell.

    This definition is appropriate for compact structural domains whose spatial
    extent is smaller than roughly half the shortest box span.
    """
    pos = np.asarray(positions_wrapped, dtype=float)
    if pos.size == 0:
        return np.full(3, np.nan, dtype=float)
    pos = wrap_positions_pbc(pos, box)
    ref = pos[int(reference_index) % len(pos)].astype(float)
    centroid = ref.copy()
    for _ in range(max(1, int(n_iter))):
        unwrapped = centroid[None, :] + minimum_image(pos - centroid[None, :], box)
        centroid = np.nanmean(unwrapped, axis=0)
    return wrap_positions_pbc(centroid[None, :], box)[0]


def pbc_radius_of_gyration(positions_wrapped, centroid, box) -> float:
    """Radius of gyration computed from MDAnalysis MIC displacements."""
    delta = minimum_image(np.asarray(positions_wrapped) - np.asarray(centroid)[None, :], box)
    return float(np.sqrt(np.nanmean(np.sum(delta * delta, axis=1))))


# -----------------------------------------------------------------------------
# Slab selection and neighbour searches
# -----------------------------------------------------------------------------

def _cart_to_frac_for_visualization(positions, box) -> np.ndarray:
    """
    Cartesian to fractional coordinates for visualization only.

    This is not used for MIC or connectivity.  It is retained only for slab
    selection/unsheared plotting, where a fractional representation is a useful
    coordinate system rather than a distance algorithm.
    """
    positions = np.asarray(positions, dtype=float)
    H = cell_matrix_from_box(box)
    return positions @ np.linalg.inv(H)


def _frac_to_cart_for_visualization(frac, box) -> np.ndarray:
    frac = np.asarray(frac, dtype=float)
    H = cell_matrix_from_box(box)
    return frac @ H


def slab_mask_pbc(positions_wrapped, box, axis, slice_center, slice_width, coordinate_system="cell"):
    """
    Select particles inside a finite periodic slab.

    ``coordinate_system='cell'`` uses the selected MDAnalysis cell coordinate
    only for defining a plotting/inspection slab.  It is not used in distance
    calculations.  ``coordinate_system='cartesian'`` reproduces old Cartesian
    slicing and is safest for z slices in pure xy shear.
    """
    positions_wrapped = wrap_positions_pbc(positions_wrapped, box)
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    if coordinate_system == "cartesian":
        L = cell_lengths(box)[axis]
        d = positions_wrapped[:, axis] - slice_center
        if is_orthorhombic_box(box):
            d = d - L * np.round(d / L)
        # for triclinic Cartesian x/y cuts are geometric cuts through the tilted
        # cell; use only for visual diagnostics when this is what you intend.
        return np.abs(d) <= 0.5 * slice_width

    if coordinate_system not in ("cell", "fractional"):
        raise ValueError("coordinate_system must be 'cell', 'fractional', or 'cartesian'.")

    lengths = cell_lengths(box)
    frac = _cart_to_frac_for_visualization(positions_wrapped, box)
    f_center = slice_center / lengths[axis]
    f_width = slice_width / lengths[axis]
    d = frac[:, axis] - f_center
    d = d - np.round(d)
    return np.abs(d) <= 0.5 * f_width


def query_pairs_pbc(positions, box, cutoff, active_mask=None, method="nsgrid") -> np.ndarray:
    """
    Return unique ``i < j`` neighbour pairs using MDAnalysis.self_capped_distance.

    ``method='nsgrid'`` uses MDAnalysis' grid-based neighbour search; its docs
    note that it was rewritten to fix triclinic-box bugs.  Use
    ``method='bruteforce'`` for small systems or validation.
    """
    positions = wrap_positions_pbc(positions, box).astype(np.float32, copy=False)
    dims = box_to_mda_dimensions(box)
    n = len(positions)

    if active_mask is None:
        active_ids = np.arange(n, dtype=int)
        active_pos = positions
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape[0] != n:
            raise ValueError("active_mask must have the same length as positions.")
        active_ids = np.where(active)[0]
        active_pos = positions[active_ids]

    if len(active_pos) < 2:
        return np.empty((0, 2), dtype=int)

    pairs = mda_distances.self_capped_distance(
        active_pos,
        max_cutoff=float(cutoff),
        box=dims,
        method=method,
        return_distances=False,
    )
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    mapped = np.column_stack([active_ids[pairs[:, 0]], active_ids[pairs[:, 1]]])
    mapped.sort(axis=1)
    # deduplicate defensively; MDAnalysis should already return unique pairs.
    mapped = np.unique(mapped, axis=0)
    return mapped.astype(int, copy=False)


def query_ball_point_pbc(positions, box, cutoff, active_mask=None, method="nsgrid"):
    """
    Return neighbour lists for each particle using MDAnalysis pair search.

    Each list includes the particle itself, matching scipy cKDTree's behaviour
    in the old toolkit.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    neigh = [set([i]) for i in range(n)]
    pairs = query_pairs_pbc(positions, box, cutoff, active_mask=active_mask, method=method)
    for i, j in pairs:
        neigh[int(i)].add(int(j))
        neigh[int(j)].add(int(i))
    return [sorted(x) for x in neigh]


# -----------------------------------------------------------------------------
# Visualization transforms
# -----------------------------------------------------------------------------

def positions_to_unsheared_orthogonal(positions, box, reference=None) -> np.ndarray:
    """
    Map positions to an orthogonalized cell-coordinate representation.

    This is only for visualization of domain footprints after removing the
    trivial triclinic cell skew.  It is deliberately separated from the PBC/MIC
    implementation, which uses MDAnalysis distance kernels.
    """
    positions = np.asarray(positions, dtype=float)
    if reference is not None:
        positions = unwrap_positions_relative(positions, reference, box)
    frac = _cart_to_frac_for_visualization(positions, box)
    return frac * cell_lengths(box)[None, :]

# Backward-compatible names for old notebooks.  These are visualization/projection
# helpers only; MIC, connectivity, wrapping, and tracking use MDAnalysis kernels.
def cart_to_frac(positions, box):
    return _cart_to_frac_for_visualization(positions, box)


def frac_to_cart(frac, box):
    return _frac_to_cart_for_visualization(frac, box)
