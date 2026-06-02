import numpy as np


def compute_propensity_3d_from_ensemble(
    traj_positions,
    gamma_dot,
    dt_ps=0.025,
    end_frame=108,
):
    """
    Compute translational dynamic propensity from an iso-configurational ensemble.

    Parameters
    ----------
    traj_positions : ndarray, shape (n_traj, n_frames, n_molecules, 3)
        Unwrapped molecular positions.
    gamma_dot : float
        Shear rate in ps^-1.
    dt_ps : float
        Time interval between stored frames in ps.
    end_frame : int
        Final frame used to compute displacement from frame 0.

    Returns
    -------
    P3D : ndarray, shape (n_molecules,)
        Mean non-affine squared displacement over the ensemble.
    dr2_all : ndarray, shape (n_traj, n_molecules)
        Per-trajectory non-affine squared displacements.
    """
    traj_positions = np.asarray(traj_positions, dtype=float)

    if traj_positions.ndim != 4 or traj_positions.shape[-1] != 3:
        raise ValueError("traj_positions must have shape (n_traj,n_frames,N,3).")
    if end_frame >= traj_positions.shape[1]:
        raise ValueError("end_frame is outside traj_positions.")

    r0 = traj_positions[:, 0, :, :]
    rt = traj_positions[:, end_frame, :, :]

    y_path = traj_positions[:, :end_frame, :, 1]
    dx_aff = gamma_dot * dt_ps * np.sum(y_path, axis=1)

    dr = rt - r0
    dr[..., 0] -= dx_aff

    dr2_all = np.sum(dr * dr, axis=-1)
    P3D = np.mean(dr2_all, axis=0)

    return P3D, dr2_all
