#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, math, re, json
from collections import deque
import numpy as np

try:
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

HEADER_RE = re.compile(r"ITEM: (\w+)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute self-intermediate scattering function Fs(k,t) from LAMMPS dump of O atoms."
    )
    p.add_argument(
        "--dumps",
        nargs="+",
        required=True,
        help="LAMMPS dump files (custom) with id type x y z ix iy iz; O atoms only recommended.",
    )
    p.add_argument(
        "--dt_fs", type=float, required=True, help="Time between saved frames (fs), e.g., 10 or 20."
    )
    p.add_argument("--k_target", type=float, default=2.0, help="Target |k| in 1/Å (default 2.0).")
    p.add_argument("--dk", type=float, default=0.15, help="Half-width of k-shell (1/Å).")
    p.add_argument("--tmax_ps", type=float, required=True, help="Maximum time lag to compute (ps).")
    p.add_argument(
        "--origin_stride_ps", type=float, default=0.5, help="Spacing between time origins (ps)."
    )
    p.add_argument(
        "--logtime", action="store_true", help="Use log-spaced time lags (otherwise uniform)."
    )
    p.add_argument(
        "--n_log", type=int, default=200, help="Number of log-spaced lag points (if --logtime)."
    )
    p.add_argument(
        "--lag_stride_frames",
        type=int,
        default=1,
        help="Use every Nth frame as lag point (uniform mode).",
    )
    p.add_argument(
        "--min_F_for_fit",
        type=float,
        default=0.6,
        help="Lower Fs threshold to start KWW fit (fit where Fs<=this).",
    )
    p.add_argument("--out_prefix", default="Fs", help="Output prefix.")
    p.add_argument(
        "--require_orthorhombic", action="store_true", help="Abort if dump shows triclinic box."
    )
    p.add_argument(
        "--no_com_removal", action="store_true", help="Do not remove COM drift (not recommended)."
    )
    p.add_argument("--print_k_list", action="store_true", help="Print selected k-vectors and exit.")
    p.add_argument(
        "--mode",
        choices=["self", "coherent"],
        default="self",
        help="Choose 'self' for self-intermediate scattering function, 'coherent' for coherent scattering function.",
    )
    p.add_argument(
        "--unwrapped", action="store_true", help="Input positions are already unwrapped."
    )
    return p.parse_args()


def k_shell_vectors(Lx, Ly, Lz, k_target, dk):
    kx0, ky0, kz0 = 2 * np.pi / Lx, 2 * np.pi / Ly, 2 * np.pi / Lz
    # Heuristic bound for n range
    nmax = int(max(k_target + dk, 8.0) / min(kx0, ky0, kz0)) + 2
    ks = []
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                kvec = np.array([nx * kx0, ny * ky0, nz * kz0], dtype=float)
                kmag = np.linalg.norm(kvec)
                if abs(kmag - k_target) <= dk:
                    ks.append(kvec)
    ks = np.unique(np.round(np.array(ks), decimals=12), axis=0)  # dedupe numerically
    return ks  # shape (M,3)


def read_dump_frames(filenames, unwrapped=False):
    """Generator yields (timestep, box_bounds, triclinic_flag, positions, images).
    positions: (N,3) float, images: (N,3) int, sorted by atom id.
    Assumes dump contains only a single species (O) OR arbitrary but we will use all atoms as oxygen group.
    """
    for fname in filenames:
        with open(fname, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.startswith("ITEM:"):
                    # skip anything until header
                    continue
                if "TIMESTEP" not in line:
                    # read until TIMESTEP
                    continue
                nstep = int(f.readline().strip())
                # NUMBER OF ATOMS
                assert (
                    "NUMBER OF ATOMS" in f.readline()
                ), "Unexpected dump format: expecting NUMBER OF ATOMS"
                Nat = int(f.readline().strip())
                # BOX BOUNDS
                bb_line = f.readline().strip()
                assert "BOX BOUNDS" in bb_line, "Unexpected dump format: expecting BOX BOUNDS"
                triclinic = ("xy" in bb_line) or ("xz" in bb_line) or ("yz" in bb_line)
                bounds = []
                for _ in range(3):
                    parts = f.readline().strip().split()
                    if len(parts) < 2:
                        raise RuntimeError("Bad BOX BOUNDS line")
                    bounds.append((float(parts[0]), float(parts[1])))
                bounds = np.array(bounds, dtype=float)  # [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
                # ATOMS
                atoms_hdr = f.readline().strip()
                assert atoms_hdr.startswith(
                    "ITEM: ATOMS"
                ), "Unexpected dump format: expecting ATOMS"
                fields = atoms_hdr.split()[2:]
                field_idx = {name: i for i, name in enumerate(fields)}

                if unwrapped:
                    required = ["id", "xu", "yu", "zu"]
                    for r in required:
                        if r not in field_idx:
                            raise RuntimeError(f"Dump must include field '{r}'. Found: {fields}")
                else:
                    required = ["id", "x", "y", "z", "ix", "iy", "iz"]
                    for r in required:
                        if r not in field_idx:
                            raise RuntimeError(f"Dump must include field '{r}'. Found: {fields}")
                # read atoms
                arr = np.empty((Nat, len(fields)), dtype=float)
                ids = np.empty(Nat, dtype=int)
                for i in range(Nat):
                    parts = f.readline().split()
                    for j, val in enumerate(parts):
                        if fields[j] in ("id", "type", "ix", "iy", "iz"):
                            arr[i, j] = int(val)
                        else:
                            arr[i, j] = float(val)
                    ids[i] = int(arr[i, field_idx["id"]])
                # sort by id
                order = np.argsort(ids)
                arr = arr[order]

                if unwrapped:
                    pos = np.stack(
                        [arr[:, field_idx["xu"]], arr[:, field_idx["yu"]], arr[:, field_idx["zu"]]],
                        axis=1,
                    ).astype(float)
                    img = np.zeros_like(pos, dtype=int)
                else:
                    pos = np.stack(
                        [arr[:, field_idx["x"]], arr[:, field_idx["y"]], arr[:, field_idx["z"]]],
                        axis=1,
                    ).astype(float)
                    img = np.stack(
                        [arr[:, field_idx["ix"]], arr[:, field_idx["iy"]], arr[:, field_idx["iz"]]],
                        axis=1,
                    ).astype(int)
                yield (nstep, bounds, triclinic, pos, img)


def unwrap_positions(pos, img, bounds):
    # Orthorhombic box
    L = bounds[:, 1] - bounds[:, 0]  # [Lx,Ly,Lz]
    unwrapped = pos + img * L
    return unwrapped


def welford_update(mean, M2, count, x):
    # x: array-like, elementwise update
    count_new = count + 1
    delta = x - mean
    mean_new = mean + delta / count_new
    delta2 = x - mean_new
    M2_new = M2 + delta * delta2
    return mean_new, M2_new, count_new


def kww(t, A, tau, beta):
    return A * np.exp(-((t / tau) ** beta))


def main():
    args = parse_args()
    mode = args.mode
    dt_ps = args.dt_fs * 1e-3
    # Build lag indices
    n_lag_max = int(round(args.tmax_ps / dt_ps))
    if args.logtime:
        # log-spaced unique integer frame lags starting from 1
        lags = np.unique(
            np.clip(
                np.round(np.logspace(0, np.log10(max(1, n_lag_max)), args.n_log)).astype(int),
                1,
                n_lag_max,
            )
        )
    else:
        lags = np.arange(1, n_lag_max + 1, args.lag_stride_frames, dtype=int)
    t_lags_ps = lags * dt_ps

    # Read first frame to determine box and k-vectors
    gen = read_dump_frames(args.dumps, unwrapped=args.unwrapped)
    try:
        nstep0, bounds0, triclinic0, pos0, img0 = next(gen)
    except StopIteration:
        print("No frames found in dump files.", file=sys.stderr)
        sys.exit(1)
    if triclinic0 and args.require_orthorhombic:
        raise RuntimeError(
            "Triclinic box detected. Re-run with orthorhombic or omit --require_orthorhombic."
        )
    L0 = bounds0[:, 1] - bounds0[:, 0]
    ks = k_shell_vectors(L0[0], L0[1], L0[2], args.k_target, args.dk)
    if ks.shape[0] == 0:
        raise RuntimeError(
            "No k-vectors found in the requested shell. Check k_target/dk and box size."
        )
    if args.print_k_list:
        for kv in ks:
            print(
                "{:.6f} {:.6f} {:.6f} | |k|={:.6f}".format(kv[0], kv[1], kv[2], np.linalg.norm(kv))
            )
        sys.exit(0)

    # Buffers
    unwrapped_buf = deque()
    com_buf = deque()
    # Time origin bookkeeping
    origin_stride_frames = max(1, int(round(args.origin_stride_ps / dt_ps)))
    # origin_queue = deque()  # stores indices (relative to buffer start) of origins
    # use list for active origins and dict to track last computed lag
    active_origins = []
    origin_state = {}
    # Statistics over origins (Welford): for each lag
    Fs_mean = np.zeros_like(t_lags_ps, dtype=float)
    Fs_M2 = np.zeros_like(t_lags_ps, dtype=float)
    Fs_count = np.zeros_like(t_lags_ps, dtype=int)

    # Process first frame
    if args.unwrapped:
        uw0 = pos0
    else:
        uw0 = unwrap_positions(pos0, img0, bounds0)
    if args.no_com_removal:
        com0 = np.zeros(3)
    else:
        com0 = uw0.mean(axis=0)
    unwrapped_buf.append(uw0)
    com_buf.append(com0)
    # First origin at index 0
    active_origins.append(0)
    origin_state[0] = 0  # last computed lag = 0 frames
    # origin_queue.append(0)
    n_frames_total = 1
    n_atoms = uw0.shape[0]
    ksT = ks.T  # (3,M)

    # Helper to compute ISF for one origin against all lags currently available
    def compute_ISF_for_origin(o_idx, lag_frames):
        # Reference positions and COM
        r0 = unwrapped_buf[o_idx]
        com0 = com_buf[o_idx]
        # calculate the frame index for this lag
        frame_idx = o_idx + lag_frames
        r1 = unwrapped_buf[frame_idx]
        com1 = com_buf[frame_idx]
        if mode == "self":
            dr = (r1 - com1) - (r0 - com0)
            phases = dr @ ksT
            return np.cos(phases).mean()
        elif mode == "coherent":
            # Memory-efficient calculation
            chunk_size = 50
            total_cos = 0.0
            n_chunks = (n_atoms + chunk_size - 1) // chunk_size

            for i in range(n_chunks):
                start_i = i * chunk_size
                end_i = min((i + 1) * chunk_size, n_atoms)
                chunk_i = r1[start_i:end_i]

                for j in range(n_chunks):
                    start_j = j * chunk_size
                    end_j = min((j + 1) * chunk_size, n_atoms)
                    chunk_j = r0[start_j:end_j]

                    dr = chunk_i[:, np.newaxis, :] - chunk_j[np.newaxis, :, :]
                    phases = np.tensordot(dr, ksT, axes=([2], [0]))
                    total_cos += np.sum(np.cos(phases))
            return total_cos / (n_atoms * n_atoms)

    # Consume the rest frames
    for nstep, bounds, tric, pos, img in gen:
        if args.unwrapped:
            uw = pos
        else:
            uw = unwrap_positions(pos, img, bounds)
        com = np.zeros(3) if args.no_com_removal else uw.mean(axis=0)
        unwrapped_buf.append(uw)
        com_buf.append(com)
        n_frames_total += 1
        current_frame_idx = len(unwrapped_buf) - 1
        # Process all active origins
        for o_idx in list(active_origins):
            # Get the last computed lag for this origin
            last_lag = origin_state[o_idx]

            # Determine which lags are now available and not yet computed
            max_available = current_frame_idx - o_idx
            lag_mask = (lags > last_lag) & (lags <= max_available)

            if not np.any(lag_mask):
                # If no new lags are available,check if we've completed all lags
                if last_lag >= lags[-1]:
                    active_origins.remove(o_idx)
                    del origin_state[o_idx]  # remove from state
                continue

            # Get the lag values and their indices
            new_lags = lags[lag_mask]
            lag_indices = np.where(lag_mask)[0]

            # Compute ISF for these new lags
            Fs_vals = []
            for lag_frames in new_lags:
                Fs_val = compute_ISF_for_origin(o_idx, lag_frames)
                Fs_vals.append(Fs_val)

            # Update the statistics for these lags
            for i, val in zip(lag_indices, Fs_vals):
                Fs_mean[i], Fs_M2[i], Fs_count[i] = welford_update(
                    Fs_mean[i], Fs_M2[i], Fs_count[i], val
                )

            # Update last computed lag for this origin
            origin_state[o_idx] = new_lags[-1]

            # Check if we've completed all lags for this origin
            if origin_state[o_idx] >= lags[-1]:
                active_origins.remove(o_idx)
                del origin_state[o_idx]

            # Add new time origin at the required stride
            if (n_frames_total - 1) % origin_stride_frames == 0:
                # Only add if we can compute at least some lags
                if current_frame_idx >= lags[0]:
                    active_origins.append(current_frame_idx)
                    origin_state[current_frame_idx] = 0  # Start with no lags computed

            # Trim buffers to conserve memory
            if active_origins:
                # Find the oldest origin still in use
                oldest_origin = min(active_origins)
                # Calculate how many frames we need to keep before this origin
                max_lag_needed = lags[-1]
                keep_from = max(0, oldest_origin - max_lag_needed)

                # Remove unnecessary frames from the beginning of the buffer
                while len(unwrapped_buf) > 0 and keep_from > 0:
                    unwrapped_buf.popleft()
                    com_buf.popleft()
                    keep_from -= 1

                    # Adjust all origin indices
                    new_active_origins = []
                    new_origin_state = {}
                    for idx in active_origins:
                        new_idx = idx - 1
                        if new_idx >= 0:  # Only keep if still valid
                            new_active_origins.append(new_idx)
                            new_origin_state[new_idx] = origin_state[idx]

                    active_origins = new_active_origins
                    origin_state = new_origin_state

    # After reading all frames, process any remaining lags for active origins
    for o_idx in list(active_origins):
        # Determine which lags are available and not yet computed
        max_available = len(unwrapped_buf) - 1 - o_idx
        lag_mask = (lags > origin_state[o_idx]) & (lags <= max_available)

        if np.any(lag_mask):
            # Get the lag values and their indices
            new_lags = lags[lag_mask]
            lag_indices = np.where(lag_mask)[0]

            # Compute Fs for these new lag times
            Fs_vals = []
            for lag_frames in new_lags:
                Fs_val = compute_ISF_for_origin(o_idx, lag_frames)
                Fs_vals.append(Fs_val)

            # Update statistics
            for i, val in zip(lag_indices, Fs_vals):
                Fs_mean[i], Fs_M2[i], Fs_count[i] = welford_update(
                    Fs_mean[i], Fs_M2[i], Fs_count[i], val
                )

            # Update last computed lag
            origin_state[o_idx] = new_lags[-1]

    # After reading all frames, we may still have pending origins whose larger lags are not available; they contributed partially.
    # We are done accumulating.

    # Prepare outputs
    # Standard error over origins for each lag
    with np.errstate(invalid="ignore", divide="ignore"):
        Fs_var = np.zeros_like(Fs_mean)
        valid_mask = Fs_count > 1
        Fs_var[valid_mask] = Fs_M2[valid_mask] / (Fs_count[valid_mask] - 1)
        Fs_sem = np.sqrt(Fs_var / np.maximum(Fs_count, 1))

    # Estimate tau_alpha by e^-1 crossing
    e_inv = math.e**-1
    tau_alpha_ps = np.nan
    # Only consider monotonic decays; find first crossing
    Fs = Fs_mean.copy()
    below = np.where(Fs <= e_inv)[0]
    if below.size > 0:
        i2 = below[0]
        if i2 == 0:
            tau_alpha_ps = t_lags_ps[0]
        else:
            i1 = i2 - 1
            # Linear interpolation between (t1,Fs1) and (t2,Fs2)
            t1, t2 = t_lags_ps[i1], t_lags_ps[i2]
            f1, f2 = Fs[i1], Fs[i2]
            if f2 != f1:
                tau_alpha_ps = t1 + (e_inv - f1) * (t2 - t1) / (f2 - f1)
            else:
                tau_alpha_ps = t2

    # Optional KWW fit
    fit_res = None
    if HAS_SCIPY:
        mask_fit = (Fs <= args.min_F_for_fit) & (Fs > 0.0)
        t_fit = t_lags_ps[mask_fit]
        F_fit = Fs[mask_fit]
        if t_fit.size >= 8:
            # Initial guesses: A≈F(0+)~1, tau≈tau_alpha or t where Fs~0.3, beta≈0.6–0.9
            tau_guess = (
                tau_alpha_ps
                if np.isfinite(tau_alpha_ps)
                else (t_fit[len(t_fit) // 3] if t_fit.size > 0 else 10.0)
            )
            p0 = [1.0, max(tau_guess, 1e-3), 0.7]
            try:
                popt, pcov = curve_fit(
                    kww, t_fit, F_fit, p0=p0, bounds=([0.5, 1e-4, 0.2], [1.2, 1e6, 1.0])
                )
                fit_res = {"A": float(popt[0]), "tau_ps": float(popt[1]), "beta": float(popt[2])}
            except Exception:
                fit_res = None

    # Save CSV
    out_csv = f"{args.out_prefix}_{'Fs' if mode=='self' else 'Fk'}_k{args.k_target:.2f}_dk{args.dk:.2f}.csv"
    with open(out_csv, "w") as g:
        g.write(f"# {'Fs(k,t)' if mode=='self' else 'F(k,t)'} intermediate scattering function\n")
        meta = {
            "k_target_1_per_A": args.k_target,
            "dk_1_per_A": args.dk,
            "dt_fs": args.dt_fs,
            "tmax_ps": args.tmax_ps,
            "origin_stride_ps": args.origin_stride_ps,
            "logtime": args.logtime,
            "n_atoms": int(n_atoms),
            "n_kvectors": int(ks.shape[0]),
            "tau_alpha_ps_e1": float(tau_alpha_ps) if np.isfinite(tau_alpha_ps) else None,
            "kww_fit": fit_res,
            "unwrapped_coords": args.unwrapped,
        }
        g.write("# " + json.dumps(meta) + "\n")
        g.write("t_ps,Fs,sem,n_origins\n")
        for t, f, se, n in zip(t_lags_ps, Fs_mean, Fs_sem, Fs_count):
            g.write(f"{t:.6f},{f:.8f},{se:.8f},{int(n)}\n")

    # Print summary
    print(f"Output: {out_csv}")
    print(
        f"Frames processed: {n_frames_total}, atoms per frame: {n_atoms}, k-vectors in shell: {ks.shape[0]}"
    )
    print(
        f"tau_alpha (e^-1 crossing): {tau_alpha_ps:.3f} ps"
        if np.isfinite(tau_alpha_ps)
        else "tau_alpha not reached within t_max."
    )
    if fit_res:
        print(
            f"KWW fit: A={fit_res['A']:.3f}, tau={fit_res['tau_ps']:.3f} ps, beta={fit_res['beta']:.3f}"
        )
    else:
        print("KWW fit not performed or failed (install scipy or adjust --min_F_for_fit).")


if __name__ == "__main__":
    main()
