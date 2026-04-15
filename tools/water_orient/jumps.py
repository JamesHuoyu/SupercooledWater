from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

ArrayLike = np.ndarray


@dataclass
class JumpEvent:
    donor: int
    arm: int
    from_acceptor: int
    to_acceptor: int
    start_frame: int
    end_frame: int
    dwell_before: int
    dwell_after: int


def _run_length_encode(x: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("_run_length_encode expects a 1D array.")
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    change = np.r_[True, x[1:] != x[:-1]]
    starts = np.flatnonzero(change)
    values = x[starts]
    ends = np.r_[starts[1:], x.size]
    lengths = ends - starts
    return values, starts, lengths


def identify_donor_arm_jumps(
    arm_acceptor_series: ArrayLike,
    min_dwell: int = 3,
    ignore_vacant: bool = True,
) -> List[JumpEvent]:
    """Identify donor-arm acceptor switch events across time.

    Parameters
    ----------
    arm_acceptor_series
        Shape (n_frames, n_mol, 2), each entry is an acceptor index or -1.
    min_dwell
        Minimum dwell length before and after a switch.
    ignore_vacant
        If True, transitions involving -1 on either side are ignored.
    """
    a = np.asarray(arm_acceptor_series, dtype=int)
    if a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError("arm_acceptor_series must have shape (n_frames, n_mol, 2).")

    n_frames, n_mol, _ = a.shape
    events: List[JumpEvent] = []

    for donor in range(n_mol):
        for arm in range(2):
            values, starts, lengths = _run_length_encode(a[:, donor, arm])
            if values.size < 2:
                continue
            for k in range(values.size - 1):
                frm = int(values[k])
                to = int(values[k + 1])
                dwell_before = int(lengths[k])
                dwell_after = int(lengths[k + 1])
                if dwell_before < min_dwell or dwell_after < min_dwell:
                    continue
                if ignore_vacant and (frm < 0 or to < 0):
                    continue
                start_frame = int(starts[k + 1])
                end_frame = int(starts[k + 1] + lengths[k + 1] - 1)
                events.append(
                    JumpEvent(
                        donor=donor,
                        arm=arm,
                        from_acceptor=frm,
                        to_acceptor=to,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        dwell_before=dwell_before,
                        dwell_after=dwell_after,
                    )
                )
    return events


def jumps_to_frame_mask(
    events: Sequence[JumpEvent],
    n_frames: int,
    n_mol: int,
    mark: str = "start",
) -> ArrayLike:
    """Map jump events onto a (n_frames, n_mol) boolean mask.

    Parameters
    ----------
    mark
        "start" marks the start frame only;
        "span" marks the full [start_frame, end_frame] interval.
    """
    mask = np.zeros((n_frames, n_mol), dtype=bool)
    for ev in events:
        if mark == "start":
            if 0 <= ev.start_frame < n_frames:
                mask[ev.start_frame, ev.donor] = True
        elif mark == "span":
            s = max(0, ev.start_frame)
            e = min(n_frames - 1, ev.end_frame)
            mask[s : e + 1, ev.donor] = True
        else:
            raise ValueError("mark must be 'start' or 'span'.")
    return mask
