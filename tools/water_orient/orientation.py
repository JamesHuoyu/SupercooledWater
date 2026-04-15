from __future__ import annotations

import numpy as np

from .geometry import minimum_image

ArrayLike = np.ndarray


def _normalize(v: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return v / norms


def dipole_vectors(O: ArrayLike, H1: ArrayLike, H2: ArrayLike, box: ArrayLike | None = None) -> ArrayLike:
    """Return normalized molecular dipole/bisector vectors.

    For rigid water this is the direction of OH1 + OH2.
    """
    if box is None:
        v1 = H1 - O
        v2 = H2 - O
    else:
        v1 = minimum_image(H1 - O, box)
        v2 = minimum_image(H2 - O, box)
    mu = v1 + v2
    return _normalize(mu)


def body_frames(O: ArrayLike, H1: ArrayLike, H2: ArrayLike, box: ArrayLike | None = None) -> ArrayLike:
    """Construct an orthonormal body-fixed frame for each water molecule.

    Returns
    -------
    frames : ndarray
        Shape (..., 3, 3). The last two axes are basis vectors [e1, e2, e3], where
        e1 is the dipole/bisector, e2 is along H1-H2, and e3 is the molecular-plane
        normal.
    """
    if box is None:
        oh1 = H1 - O
        oh2 = H2 - O
    else:
        oh1 = minimum_image(H1 - O, box)
        oh2 = minimum_image(H2 - O, box)

    e1 = _normalize(oh1 + oh2)
    e2_raw = _normalize(oh1 - oh2)
    e3 = _normalize(np.cross(e1, e2_raw))
    e2 = _normalize(np.cross(e3, e1))
    return np.stack([e1, e2, e3], axis=-2)


def angular_displacement(v1: ArrayLike, v2: ArrayLike, degrees: bool = True) -> ArrayLike:
    """Angular displacement between two vector fields."""
    a = _normalize(np.asarray(v1, dtype=float))
    b = _normalize(np.asarray(v2, dtype=float))
    cosang = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    ang = np.arccos(cosang)
    return np.degrees(ang) if degrees else ang


def angular_displacement_from_frames(frames_a: ArrayLike, frames_b: ArrayLike, degrees: bool = True) -> ArrayLike:
    """Geodesic rotation angle between two 3D body frames.

    Uses tr(R_a^T R_b) = 1 + 2 cos(theta).
    """
    fa = np.asarray(frames_a, dtype=float)
    fb = np.asarray(frames_b, dtype=float)
    rel = np.einsum("...ij,...jk->...ik", np.swapaxes(fa, -1, -2), fb)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cosang = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cosang)
    return np.degrees(ang) if degrees else ang


def rotational_correlation(v_t0: ArrayLike, v_t: ArrayLike, l: int = 2) -> ArrayLike:
    """Single-time Legendre rotational correlation C_l.

    Parameters
    ----------
    v_t0, v_t
        Vector fields of shape (..., 3).
    l
        Legendre order. l=1 for dipole, l=2 for common water reorientation analysis.
    """
    x = np.clip(np.sum(_normalize(v_t0) * _normalize(v_t), axis=-1), -1.0, 1.0)
    if l == 1:
        return x
    if l == 2:
        return 0.5 * (3.0 * x**2 - 1.0)
    coeffs = [0] * l + [1]
    return np.polynomial.legendre.legval(x, coeffs)
