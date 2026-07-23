#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMI electrode/electrolyte analysis modes.

MODES
-----
Run selected modes in one command, with one Universe/frame pass per trajectory:
    python DMI_edlanal.py run HEADER DATA TRJ1 [TRJ2 ...] \
        --modes angle1 angle2 iondist coordination rdf watermassdensity \
        --z-adlayer Z_ADLAYER --iontype Na --z-ihl Z_IHL --z-ohl Z_OHL \
        --coord-cutoff CUTOFF --rdf-z-low Z --rdf-z-high Z --rdf-rmax R \
        --nprocs N --skip-frames N

Run the same selected modes for multiple header/data/trajectory groups.
Trajectory tasks are queued case-major and executed in chunks of --nprocs:
    python DMI_edlanal.py batch \
        --case HEADER1 DATA1 TRJ1 [TRJ2 ...] \
        --case HEADER2 DATA2 TRJ1 [TRJ2 ...] \
        --modes all --z-adlayer Z_ADLAYER --iontype Na ...

1) O-H bond orientation classes for water adlayer:
    python DMI_edlanal.py angle1 Z_ADLAYER HEADER DATA TRJ1 [TRJ2 ...] --nprocs N --skip-frames N

2) Water bisector / molecular-plane angle histograms for water adlayer:
    python DMI_edlanal.py angle2 Z_ADLAYER HEADER DATA TRJ1 [TRJ2 ...] --nprocs N --skip-frames N

3) Ionic charge distribution over IHL/OHL/DifL:
    python DMI_edlanal.py iondist IONTYPE Z_IHL Z_OHL HEADER DATA TRJ1 [TRJ2 ...] --nprocs N --skip-frames N

4) Interfacial-water / ion coordination statistics:
    python DMI_edlanal.py coordination IONTYPE Z_ADLAYER CUTOFF HEADER DATA TRJ1 [TRJ2 ...] --nprocs N --skip-frames N

5) Water mass density profile:
    python DMI_edlanal.py watermassdensity HEADER DATA TRJ1 [TRJ2 ...] --nprocs N --skip-frames N

6) Bulk-region RDF, globally pooled over all input trajectories:
    python DMI_edlanal.py rdf IONTYPE Z_LOW Z_HIGH R_MAX HEADER DATA TRJ1 [TRJ2 ...] --nprocs N --skip-frames N --dr DR

Z inputs are offsets from Z_REF + Z_SLAB, matching the old scripts:
    z_abs = 200.0 + 7.1782 + z_input

Default atom types:
    H: 1, O: 2, ion: 3
"""

import argparse
import os
import sys
from multiprocessing import get_context
from typing import Iterable, Tuple

import numpy as np


# ----------------- System constants -----------------
H_TYPES = {1}
O_TYPE = 2
ION_TYPE = 3

Z_REF = 200.0
Z_SLAB = 7.1782
Z_ORIGIN = Z_REF + Z_SLAB

DEFAULT_SKIP_FRAMES = 1
OH_MAX_CHECK = 1.0

ANGLE_BINS = np.arange(0.0, 181.0, 1.0, dtype=np.float64)
ANGLE_CENTERS = 0.5 * (ANGLE_BINS[:-1] + ANGLE_BINS[1:])

SIGMA_PER_ION = 0.66495  # uC/cm^2 per ion, used by existing plotting scripts

BIN_DZ_A = 0.1
DEFAULT_WATER_Z_MIN = 0.0
DEFAULT_WATER_Z_MAX = 15.0
DEFAULT_RDF_DR = 0.1
O_MASS_AMU = 15.999
H_MASS_AMU = 1.008
AMU_TO_G = 1.66053906660e-24
MASS_WATER_G = (O_MASS_AMU + 2.0 * H_MASS_AMU) * AMU_TO_G
ANG3_TO_CM3 = 1.0e-24

ZHAT = np.array([0.0, 0.0, 1.0], dtype=float)


def get_mda():
    try:
        import MDAnalysis as mda
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MDAnalysis is required for trajectory analysis. "
            "Run this script with the Python environment where MDAnalysis is installed."
        ) from exc
    return mda


def abs_z(z_offset: float) -> float:
    return Z_ORIGIN + float(z_offset)


def safe_fraction(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def check_nion_frame(ions, expected_nion: int, context: str):
    current = int(ions.positions.shape[0])
    if current != expected_nion:
        raise RuntimeError(
            f"type {ION_TYPE} ion count changed ({context}): "
            f"expected {expected_nion}, got {current}."
        )


def check_global_nion(results, label: str):
    nions = sorted({int(result["nion_total"]) for result in results if "nion_total" in result})
    if len(nions) > 1:
        raise RuntimeError(
            f"type {ION_TYPE} ion count differs between trajectories for {label}: {nions}. "
            "RDF mode allows mixed ion counts, but this mode requires one fixed nion."
        )
    if nions:
        print(f"[CHECK] {label}: global type{ION_TYPE}_atom_count={nions[0]}", file=sys.stderr)


def use_first_trajectory_frame(u, dump_path: str, label: str):
    if len(u.trajectory) == 0:
        raise RuntimeError(f"No trajectory frames found for {label}: {dump_path}")
    u.trajectory[0]


def minimage_vec_xy(d: np.ndarray, box_lengths: Iterable[float]) -> np.ndarray:
    lx, ly, _lz = box_lengths
    out = d.copy()
    out[..., 0] -= lx * np.rint(out[..., 0] / lx)
    out[..., 1] -= ly * np.rint(out[..., 1] / ly)
    return out


def pair_hits_xy(site_pos: np.ndarray, ion_pos: np.ndarray, box, cutoff: float, chunk_ion: int = 512) -> np.ndarray:
    """
    Boolean hit matrix for site-ion pairs using x/y PBC and non-periodic z.
    hit[i, j] is True when site i is within cutoff of ion j.
    """
    n_site = site_pos.shape[0]
    n_ion = ion_pos.shape[0]
    hits = np.zeros((n_site, n_ion), dtype=bool)
    if n_site == 0 or n_ion == 0:
        return hits

    cutoff2 = cutoff * cutoff
    for j0 in range(0, n_ion, chunk_ion):
        j1 = min(j0 + chunk_ion, n_ion)
        disp = site_pos[:, None, :] - ion_pos[None, j0:j1, :]
        disp = minimage_vec_xy(disp, box)
        d2 = np.sum(disp * disp, axis=2)
        hits[:, j0:j1] = d2 <= cutoff2
    return hits


def normalize(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(v, axis=1)
    good = norms > 1.0e-12
    out = np.zeros_like(v, dtype=float)
    out[good] = v[good] / norms[good, None]
    return out, good


def build_water_triplets(u, o_type=O_TYPE, h_types=H_TYPES, max_oh=1.2) -> np.ndarray:
    oxy = u.select_atoms(f"type {o_type}")
    waters = []

    try:
        has_bonds = any(len(atom.bonds) > 0 for atom in oxy)
    except Exception:
        has_bonds = False

    if has_bonds:
        def is_h(atom):
            try:
                return int(atom.type) in h_types
            except Exception:
                return (getattr(atom, "element", "") or "").upper() == "H"

        for oxygen in oxy:
            neigh = set()
            for bond in oxygen.bonds:
                neigh.update(atom for atom in bond.atoms if atom.index != oxygen.index)
            hs = [atom for atom in neigh if is_h(atom)]
            if len(hs) == 2:
                waters.append((oxygen.index, hs[0].index, hs[1].index))
    else:
        h_sel = " or ".join(f"type {t}" for t in sorted(h_types))
        hyd = u.select_atoms(h_sel)
        h_pos = hyd.positions
        for oxygen in oxy:
            d = h_pos - oxygen.position
            dist = np.linalg.norm(d, axis=1)
            idx = np.where(dist <= max_oh)[0]
            if idx.size >= 2:
                ids = idx[np.argsort(dist[idx])[:2]]
                waters.append((oxygen.index, hyd[ids[0]].index, hyd[ids[1]].index))

    waters = np.asarray(waters, dtype=int)
    if waters.size == 0:
        raise RuntimeError("No water triplets found. Check atom types or bonds.")
    return waters


def frame_water_vectors(
    pos: np.ndarray,
    waters: np.ndarray,
    box,
    context: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    oxygen = pos[waters[:, 0]]
    h1 = pos[waters[:, 1]]
    h2 = pos[waters[:, 2]]
    v1 = minimage_vec_xy(h1 - oxygen, box)
    v2 = minimage_vec_xy(h2 - oxygen, box)

    oh1 = np.linalg.norm(v1, axis=1)
    oh2 = np.linalg.norm(v2, axis=1)
    bad = (oh1 > OH_MAX_CHECK) | (oh2 > OH_MAX_CHECK)
    if np.any(bad):
        max_oh = max(float(oh1.max()), float(oh2.max()))
        where = f" ({context})" if context else ""
        raise RuntimeError(
            f"OH sanity check failed{where}: {int(bad.sum())}/{waters.shape[0]} "
            f"water triplets have OH > {OH_MAX_CHECK:.2f} Ang (max={max_oh:.3f} Ang)."
        )
    return oxygen, v1, v2


def oh_angles_from_vectors(v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n1, good1 = normalize(v1)
    n2, good2 = normalize(v2)
    good = good1 & good2
    n1 = n1[good]
    n2 = n2[good]
    th1 = np.degrees(np.arccos(np.clip(n1 @ ZHAT, -1.0, 1.0)))
    th2 = np.degrees(np.arccos(np.clip(n2 @ ZHAT, -1.0, 1.0)))
    return th1, th2


def bisector_plane_angles(v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n1, good1 = normalize(v1)
    n2, good2 = normalize(v2)
    good = good1 & good2
    n1 = n1[good]
    n2 = n2[good]

    bis = n1 + n2
    bis, good_bis = normalize(bis)
    phi = np.degrees(np.arccos(np.clip(bis[good_bis] @ ZHAT, -1.0, 1.0)))

    normal = np.cross(n1, n2)
    normal, good_normal = normalize(normal)
    chi = np.degrees(np.arccos(np.clip(normal[good_normal] @ ZHAT, -1.0, 1.0)))

    return phi, chi


def classify_oh_angles(theta1: np.ndarray, theta2: np.ndarray) -> dict:
    up1 = (theta1 > 0.0) & (theta1 <= 60.0)
    up2 = (theta2 > 0.0) & (theta2 <= 60.0)
    mid1 = (theta1 > 60.0) & (theta1 <= 120.0)
    mid2 = (theta2 > 60.0) & (theta2 <= 120.0)
    down1 = (theta1 > 120.0) & (theta1 <= 180.0)
    down2 = (theta2 > 120.0) & (theta2 <= 180.0)

    two_h_up = up1 & up2
    one_h_up = (up1 & mid2) | (up2 & mid1)
    flat = mid1 & mid2
    one_h_down = (down1 & mid2) | (down2 & mid1)
    two_h_down = down1 & down2
    up_down_mixed = (up1 & down2) | (up2 & down1)

    h_up = two_h_up | one_h_up
    h_down = two_h_down | one_h_down

    return {
        "total": int(theta1.size),
        "flat": int(flat.sum()),
        "h_down": int(h_down.sum()),
        "h_up": int(h_up.sum()),
        "up_down_mixed": int(up_down_mixed.sum()),
        "two_h_up": int(two_h_up.sum()),
        "one_h_up": int(one_h_up.sum()),
        "one_h_down": int(one_h_down.sum()),
        "two_h_down": int(two_h_down.sum()),
    }


def add_counts(a: dict, b: dict) -> dict:
    out = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, 0) + value
    return out


def analyze_angle1_dump(args):
    data_path, dump_path, z_adlayer_abs, skip_frames = args
    print(f"[INFO] angle1: {dump_path}", file=sys.stderr)

    u = get_mda().Universe(data_path, dump_path, format="LAMMPSDUMP")
    use_first_trajectory_frame(u, dump_path, "angle1")
    waters = build_water_triplets(u)
    ions = u.select_atoms(f"type {ION_TYPE}")
    expected_nion = int(ions.n_atoms)
    print(f"[TRIPLETS] angle1: {dump_path}: water_triplets={len(waters)}", file=sys.stderr)
    print(f"[IONS] angle1: {dump_path}: type{ION_TYPE}_atoms={expected_nion}", file=sys.stderr)

    counts = {
        "total": 0,
        "flat": 0,
        "h_down": 0,
        "h_up": 0,
        "up_down_mixed": 0,
        "two_h_up": 0,
        "one_h_up": 0,
        "one_h_down": 0,
        "two_h_down": 0,
        "frames": 0,
        "frames_with_adlayer": 0,
        "adlayer_waters": 0,
        "water_triplets_sum": len(waters),
        "nion_total": expected_nion,
        "dumps": 1,
    }

    for local_frame, ts in enumerate(u.trajectory):
        if local_frame < skip_frames:
            continue

        check_nion_frame(ions, expected_nion, f"angle1, dump={dump_path}, frame={ts.frame}")
        counts["frames"] += 1
        oxygen, v1, v2 = frame_water_vectors(
            ts.positions,
            waters,
            ts.dimensions[:3],
            context=f"angle1, dump={dump_path}, frame={ts.frame}",
        )
        mask = oxygen[:, 2] < z_adlayer_abs
        if not np.any(mask):
            continue

        counts["frames_with_adlayer"] += 1
        counts["adlayer_waters"] += int(np.count_nonzero(mask))
        th1, th2 = oh_angles_from_vectors(v1[mask], v2[mask])
        counts = add_counts(counts, classify_oh_angles(th1, th2))

    print(
        f"[SUMMARY] angle1: {dump_path}: frames={counts['frames']}, "
        f"frames_with_adlayer={counts['frames_with_adlayer']}, "
        f"adlayer_waters={counts['adlayer_waters']}, classified={counts['total']}",
        file=sys.stderr,
    )
    return counts


def run_angle1(args):
    z_adlayer_abs = abs_z(args.z_adlayer)
    print(
        f"[SETUP] angle1: header={args.header}, z_adlayer_input={args.z_adlayer:.6f}, "
        f"z_adlayer_abs={z_adlayer_abs:.6f}, skip_frames={args.skip_frames}, data={args.data}",
        file=sys.stderr,
    )
    tasks = [(args.data, dump, z_adlayer_abs, args.skip_frames) for dump in args.trajectories]
    results = run_tasks(analyze_angle1_dump, tasks, args.nprocs, label="angle1")
    check_global_nion(results, "angle1")

    total = {
        "total": 0,
        "flat": 0,
        "h_down": 0,
        "h_up": 0,
        "up_down_mixed": 0,
        "two_h_up": 0,
        "one_h_up": 0,
        "one_h_down": 0,
        "two_h_down": 0,
        "frames": 0,
        "frames_with_adlayer": 0,
        "adlayer_waters": 0,
        "water_triplets_sum": 0,
        "dumps": 0,
    }
    for result in results:
        total = add_counts(total, result)

    if total["total"] == 0:
        sys.exit("No adlayer waters found for angle1.")

    avg_triplets = safe_fraction(total["water_triplets_sum"], total["dumps"])
    print(
        f"[TOTAL] angle1: dumps={total['dumps']}, frames={total['frames']}, "
        f"frames_with_adlayer={total['frames_with_adlayer']}, "
        f"adlayer_waters={total['adlayer_waters']}, classified={total['total']}, "
        f"avg_water_triplets_per_dump={avg_triplets:.2f}",
        file=sys.stderr,
    )

    out_path = f"{args.header}_angle1_classes.txt"
    with open(out_path, "w") as f:
        f.write("# O-H bond orientation classes for water adlayer\n")
        f.write(f"# z_adlayer_input={args.z_adlayer:.8f}\n")
        f.write(f"# z_adlayer_abs={z_adlayer_abs:.8f}\n")
        f.write("# class count fraction\n")
        for key in ("flat", "h_down", "h_up", "up_down_mixed"):
            f.write(f"{key} {total[key]} {safe_fraction(total[key], total['total']):.10f}\n")
        f.write("# fine_class count fraction\n")
        for key in ("two_h_up", "one_h_up", "one_h_down", "two_h_down"):
            f.write(f"{key} {total[key]} {safe_fraction(total[key], total['total']):.10f}\n")
        f.write(f"total {total['total']} 1.0000000000\n")

    print(f"[DONE] wrote {out_path}")


def analyze_angle2_dump(args):
    data_path, dump_path, z_adlayer_abs, skip_frames = args
    print(f"[INFO] angle2: {dump_path}", file=sys.stderr)

    u = get_mda().Universe(data_path, dump_path, format="LAMMPSDUMP")
    use_first_trajectory_frame(u, dump_path, "angle2")
    waters = build_water_triplets(u)
    ions = u.select_atoms(f"type {ION_TYPE}")
    expected_nion = int(ions.n_atoms)
    print(f"[TRIPLETS] angle2: {dump_path}: water_triplets={len(waters)}", file=sys.stderr)
    print(f"[IONS] angle2: {dump_path}: type{ION_TYPE}_atoms={expected_nion}", file=sys.stderr)

    phi_hist = np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64)
    chi_hist = np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64)
    phi_sum = phi_sumsq = 0.0
    chi_sum = chi_sumsq = 0.0
    phi_n = chi_n = 0
    frames = 0
    frames_with_adlayer = 0
    adlayer_waters = 0

    for local_frame, ts in enumerate(u.trajectory):
        if local_frame < skip_frames:
            continue

        check_nion_frame(ions, expected_nion, f"angle2, dump={dump_path}, frame={ts.frame}")
        frames += 1
        oxygen, v1, v2 = frame_water_vectors(
            ts.positions,
            waters,
            ts.dimensions[:3],
            context=f"angle2, dump={dump_path}, frame={ts.frame}",
        )
        mask = oxygen[:, 2] < z_adlayer_abs
        if not np.any(mask):
            continue

        frames_with_adlayer += 1
        adlayer_waters += int(np.count_nonzero(mask))
        phi, chi = bisector_plane_angles(v1[mask], v2[mask])
        if phi.size:
            hist, _ = np.histogram(phi, bins=ANGLE_BINS)
            phi_hist += hist
            phi_sum += float(phi.sum())
            phi_sumsq += float(np.square(phi).sum())
            phi_n += int(phi.size)
        if chi.size:
            hist, _ = np.histogram(chi, bins=ANGLE_BINS)
            chi_hist += hist
            chi_sum += float(chi.sum())
            chi_sumsq += float(np.square(chi).sum())
            chi_n += int(chi.size)

    print(
        f"[SUMMARY] angle2: {dump_path}: frames={frames}, "
        f"frames_with_adlayer={frames_with_adlayer}, adlayer_waters={adlayer_waters}, "
        f"phi_count={phi_n}, chi_count={chi_n}",
        file=sys.stderr,
    )
    return {
        "phi_hist": phi_hist,
        "chi_hist": chi_hist,
        "phi_sum": phi_sum,
        "phi_sumsq": phi_sumsq,
        "phi_n": phi_n,
        "chi_sum": chi_sum,
        "chi_sumsq": chi_sumsq,
        "chi_n": chi_n,
        "frames": frames,
        "frames_with_adlayer": frames_with_adlayer,
        "adlayer_waters": adlayer_waters,
        "water_triplets_sum": len(waters),
        "nion_total": expected_nion,
        "dumps": 1,
    }


def mean_std(sum_val: float, sumsq_val: float, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    mean = sum_val / n
    var = max(0.0, sumsq_val / n - mean * mean)
    return mean, float(np.sqrt(var))


def write_angle_hist(path: str, label: str, hist: np.ndarray, sum_val: float, sumsq_val: float, n: int):
    mean, std = mean_std(sum_val, sumsq_val, n)
    with open(path, "w") as f:
        f.write(f"# {label} angle histogram, 1 degree bins from 0 to 180\n")
        f.write("# mean_deg std_deg count\n")
        f.write(f"{mean:.8f} {std:.8f} {n}\n")
        f.write("# angle_center_deg raw_counts normalized_frequency\n")
        total = int(hist.sum())
        for center, count in zip(ANGLE_CENTERS, hist):
            f.write(f"{center:.1f} {int(count)} {safe_fraction(count, total):.10e}\n")


def run_angle2(args):
    z_adlayer_abs = abs_z(args.z_adlayer)
    print(
        f"[SETUP] angle2: header={args.header}, z_adlayer_input={args.z_adlayer:.6f}, "
        f"z_adlayer_abs={z_adlayer_abs:.6f}, skip_frames={args.skip_frames}, data={args.data}",
        file=sys.stderr,
    )
    tasks = [(args.data, dump, z_adlayer_abs, args.skip_frames) for dump in args.trajectories]
    results = run_tasks(analyze_angle2_dump, tasks, args.nprocs, label="angle2")
    check_global_nion(results, "angle2")

    phi_hist = np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64)
    chi_hist = np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64)
    phi_sum = phi_sumsq = 0.0
    chi_sum = chi_sumsq = 0.0
    phi_n = chi_n = 0
    frames = 0
    frames_with_adlayer = 0
    adlayer_waters = 0
    water_triplets_sum = 0
    dumps = 0

    for result in results:
        phi_hist += result["phi_hist"]
        chi_hist += result["chi_hist"]
        phi_sum += result["phi_sum"]
        phi_sumsq += result["phi_sumsq"]
        phi_n += result["phi_n"]
        chi_sum += result["chi_sum"]
        chi_sumsq += result["chi_sumsq"]
        chi_n += result["chi_n"]
        frames += result["frames"]
        frames_with_adlayer += result["frames_with_adlayer"]
        adlayer_waters += result["adlayer_waters"]
        water_triplets_sum += result["water_triplets_sum"]
        dumps += result["dumps"]

    if phi_n == 0 and chi_n == 0:
        sys.exit("No adlayer waters found for angle2.")

    print(
        f"[TOTAL] angle2: dumps={dumps}, frames={frames}, "
        f"frames_with_adlayer={frames_with_adlayer}, adlayer_waters={adlayer_waters}, "
        f"phi_count={phi_n}, chi_count={chi_n}, "
        f"avg_water_triplets_per_dump={safe_fraction(water_triplets_sum, dumps):.2f}",
        file=sys.stderr,
    )

    phi_path = f"{args.header}_phi_bisector_hist.txt"
    chi_path = f"{args.header}_chi_plane_hist.txt"
    write_angle_hist(phi_path, "phi_bisector_vs_z", phi_hist, phi_sum, phi_sumsq, phi_n)
    write_angle_hist(chi_path, "chi_plane_normal_vs_z", chi_hist, chi_sum, chi_sumsq, chi_n)
    print(f"[DONE] wrote {phi_path}")
    print(f"[DONE] wrote {chi_path}")


def ion_sign(iontype: str) -> float:
    key = iontype.strip().lower().replace("+", "").replace("-", "")
    key = key.replace("–", "").replace("−", "")
    if key == "na":
        return +1.0
    if key == "f":
        return -1.0
    raise ValueError("IONTYPE must be Na or F")


def analyze_iondist_dump(args):
    data_path, dump_path, z_ihl_abs, z_ohl_abs, skip_frames = args
    print(f"[INFO] iondist: {dump_path}", file=sys.stderr)

    u = get_mda().Universe(data_path, dump_path, format="LAMMPSDUMP")
    ions = u.select_atoms(f"type {ION_TYPE}")
    if ions.n_atoms == 0:
        raise RuntimeError(f"No ions found for type={ION_TYPE} in {dump_path}")
    expected_nion = int(ions.n_atoms)
    print(f"[IONS] iondist: {dump_path}: ion_atoms={expected_nion}", file=sys.stderr)

    counts = {
        "ihl": 0,
        "ohl": 0,
        "difl": 0,
        "total": 0,
        "frames": 0,
        "nion_total": expected_nion,
        "dumps": 1,
    }

    for local_frame, ts in enumerate(u.trajectory):
        if local_frame < skip_frames:
            continue

        check_nion_frame(ions, expected_nion, f"iondist, dump={dump_path}, frame={ts.frame}")
        z = ions.positions[:, 2]
        n_ihl = int(np.count_nonzero(z < z_ihl_abs))
        n_ohl = int(np.count_nonzero((z >= z_ihl_abs) & (z < z_ohl_abs)))
        n_total = int(ions.n_atoms)
        n_difl = n_total - n_ihl - n_ohl

        counts["ihl"] += n_ihl
        counts["ohl"] += n_ohl
        counts["difl"] += n_difl
        counts["total"] += n_total
        counts["frames"] += 1

    if counts["frames"] > 0:
        print(
            f"[SUMMARY] iondist: {dump_path}: frames={counts['frames']}, "
            f"avg_ihl={counts['ihl'] / counts['frames']:.4f}, "
            f"avg_ohl={counts['ohl'] / counts['frames']:.4f}, "
            f"avg_difl={counts['difl'] / counts['frames']:.4f}, "
            f"avg_total={counts['total'] / counts['frames']:.4f}",
            file=sys.stderr,
        )
    return counts


def run_iondist(args):
    try:
        sign = ion_sign(args.iontype)
    except ValueError as exc:
        sys.exit(str(exc))

    z_ihl_abs = abs_z(args.z_ihl)
    z_ohl_abs = abs_z(args.z_ohl)
    if z_ohl_abs <= z_ihl_abs:
        sys.exit("Z_OHL must be larger than Z_IHL.")

    print(
        f"[SETUP] iondist: header={args.header}, iontype={args.iontype}, "
        f"z_ihl_input={args.z_ihl:.6f}, z_ohl_input={args.z_ohl:.6f}, "
        f"z_ihl_abs={z_ihl_abs:.6f}, z_ohl_abs={z_ohl_abs:.6f}, "
        f"skip_frames={args.skip_frames}, data={args.data}",
        file=sys.stderr,
    )
    tasks = [(args.data, dump, z_ihl_abs, z_ohl_abs, args.skip_frames) for dump in args.trajectories]
    results = run_tasks(analyze_iondist_dump, tasks, args.nprocs, label="iondist")
    check_global_nion(results, "iondist")

    total = {"ihl": 0, "ohl": 0, "difl": 0, "total": 0, "frames": 0, "dumps": 0}
    for result in results:
        total = add_counts(total, result)

    if total["frames"] == 0:
        sys.exit("No frames processed for iondist.")

    print(
        f"[TOTAL] iondist: dumps={total['dumps']}, frames={total['frames']}, "
        f"avg_ihl={total['ihl'] / total['frames']:.4f}, "
        f"avg_ohl={total['ohl'] / total['frames']:.4f}, "
        f"avg_difl={total['difl'] / total['frames']:.4f}, "
        f"avg_total={total['total'] / total['frames']:.4f}",
        file=sys.stderr,
    )

    out_path = f"{args.header}_ion_charge_distribution.txt"
    with open(out_path, "w") as f:
        f.write("# ionic charge distribution\n")
        f.write(f"# iontype={args.iontype}\n")
        f.write(f"# ion_type={ION_TYPE}\n")
        f.write(f"# z_ihl_input={args.z_ihl:.8f}\n")
        f.write(f"# z_ohl_input={args.z_ohl:.8f}\n")
        f.write(f"# z_ihl_abs={z_ihl_abs:.8f}\n")
        f.write(f"# z_ohl_abs={z_ohl_abs:.8f}\n")
        f.write(f"# sigma_per_ion_uC_cm2={SIGMA_PER_ION:.8f}\n")
        f.write(f"# frames={total['frames']}\n")
        f.write("# region avg_count_per_frame fraction sigma_abs_uC_cm2 sigma_signed_uC_cm2 total_count\n")
        for region in ("ihl", "ohl", "difl"):
            avg = total[region] / total["frames"]
            frac = safe_fraction(total[region], total["total"])
            sigma_abs = avg * SIGMA_PER_ION
            sigma_signed = sign * sigma_abs
            f.write(
                f"{region} {avg:.10f} {frac:.10f} "
                f"{sigma_abs:.10f} {sigma_signed:.10f} {total[region]}\n"
            )
        avg_total = total["total"] / total["frames"]
        f.write(f"total {avg_total:.10f} 1.0000000000 0.0000000000 0.0000000000 {total['total']}\n")

    print(f"[DONE] wrote {out_path}")


def normalize_ion_mode(iontype: str) -> str:
    key = iontype.strip().lower().replace("+", "").replace("-", "")
    key = key.replace("–", "").replace("−", "")
    if key == "na":
        return "Na"
    if key == "f":
        return "F"
    raise ValueError("IONTYPE must be Na or F")


def analyze_coordination_dump(args):
    data_path, dump_path, ion_mode, z_adlayer_abs, cutoff, skip_frames = args
    print(f"[INFO] coordination: {dump_path} ({ion_mode})", file=sys.stderr)

    u = get_mda().Universe(data_path, dump_path, format="LAMMPSDUMP")
    use_first_trajectory_frame(u, dump_path, "coordination")
    waters = build_water_triplets(u)
    ions = u.select_atoms(f"type {ION_TYPE}")
    expected_nion = int(ions.n_atoms)
    print(
        f"[TRIPLETS] coordination: {dump_path}: water_triplets={len(waters)}, "
        f"ion_atoms={expected_nion}",
        file=sys.stderr,
    )

    stats = {
        "frames": 0,
        "frames_with_adlayer": 0,
        "adlayer_waters": 0,
        "sum_iw_pair_count": 0,
        "sum_iw_water_count": 0,
        "sum_selected_ion_count": 0,
        "sum_selected_ion_total_cn": 0,
        "water_triplets_sum": len(waters),
        "ion_atoms_sum": int(ions.n_atoms),
        "nion_total": expected_nion,
        "dumps": 1,
    }

    for local_frame, ts in enumerate(u.trajectory):
        if local_frame < skip_frames:
            continue

        check_nion_frame(ions, expected_nion, f"coordination, dump={dump_path}, frame={ts.frame}")
        box = ts.dimensions[:3]
        oxygen, v1, v2 = frame_water_vectors(
            ts.positions,
            waters,
            box,
            context=f"coordination, dump={dump_path}, frame={ts.frame}",
        )
        h1 = oxygen + v1
        h2 = oxygen + v2
        ion_pos = ions.positions

        adlayer_mask = oxygen[:, 2] < z_adlayer_abs
        n_adlayer = int(np.count_nonzero(adlayer_mask))

        if ion_mode == "Na":
            iw_site_pos = oxygen[adlayer_mask]
            all_site_pos = oxygen
            site_water_ids = np.arange(oxygen.shape[0], dtype=int)[adlayer_mask]
            pair_detail = "O(adlayer water)-Na"
        else:
            adlayer_water_ids = np.arange(oxygen.shape[0], dtype=int)[adlayer_mask]
            h1_adlayer = h1[adlayer_mask]
            h2_adlayer = h2[adlayer_mask]
            iw_site_pos = (
                np.vstack((h1_adlayer, h2_adlayer))
                if h1_adlayer.size > 0
                else np.empty((0, 3), dtype=float)
            )
            site_water_ids = (
                np.concatenate((adlayer_water_ids, adlayer_water_ids))
                if adlayer_water_ids.size > 0
                else np.empty(0, dtype=int)
            )
            all_site_pos = np.vstack((h1, h2))
            pair_detail = "H(adlayer water)-F"

        hit_iw = pair_hits_xy(iw_site_pos, ion_pos, box, cutoff)
        iw_pair_count = int(np.count_nonzero(hit_iw))
        if hit_iw.shape[0] > 0:
            iw_water_count = int(np.unique(site_water_ids[np.any(hit_iw, axis=1)]).size)
        else:
            iw_water_count = 0

        if hit_iw.shape[1] > 0:
            selected_ion_mask = np.any(hit_iw, axis=0)
        else:
            selected_ion_mask = np.zeros(ion_pos.shape[0], dtype=bool)
        selected_ion_count = int(np.count_nonzero(selected_ion_mask))

        if selected_ion_count > 0:
            hit_all = pair_hits_xy(all_site_pos, ion_pos[selected_ion_mask], box, cutoff)
            selected_ion_total_cn = int(np.count_nonzero(hit_all))
        else:
            selected_ion_total_cn = 0

        stats["frames"] += 1
        if n_adlayer > 0:
            stats["frames_with_adlayer"] += 1
        stats["adlayer_waters"] += n_adlayer
        stats["sum_iw_pair_count"] += iw_pair_count
        stats["sum_iw_water_count"] += iw_water_count
        stats["sum_selected_ion_count"] += selected_ion_count
        stats["sum_selected_ion_total_cn"] += selected_ion_total_cn

    if stats["frames"] > 0:
        cn_total_if = safe_fraction(
            stats["sum_selected_ion_total_cn"],
            stats["sum_selected_ion_count"],
        )
        print(
            f"[SUMMARY] coordination: {dump_path}: pair={pair_detail}, "
            f"frames={stats['frames']}, frames_with_adlayer={stats['frames_with_adlayer']}, "
            f"avg_iw_pairs={stats['sum_iw_pair_count'] / stats['frames']:.4f}, "
            f"avg_iw_waters={stats['sum_iw_water_count'] / stats['frames']:.4f}, "
            f"avg_selected_ions={stats['sum_selected_ion_count'] / stats['frames']:.4f}, "
            f"CN_total_IF={cn_total_if:.4f}",
            file=sys.stderr,
        )
    return stats


def run_coordination(args):
    try:
        ion_mode = normalize_ion_mode(args.iontype)
    except ValueError as exc:
        sys.exit(str(exc))

    cutoff = float(args.coord_cutoff)
    if cutoff <= 0.0:
        sys.exit("coordination cutoff must be > 0.")

    z_adlayer_abs = abs_z(args.z_adlayer)
    pair_label = "Na-O" if ion_mode == "Na" else "F-H"
    print(
        f"[SETUP] coordination: header={args.header}, iontype={ion_mode}, pair={pair_label}, "
        f"z_adlayer_input={args.z_adlayer:.6f}, z_adlayer_abs={z_adlayer_abs:.6f}, "
        f"cutoff={cutoff:.6f}, skip_frames={args.skip_frames}, data={args.data}",
        file=sys.stderr,
    )

    tasks = [
        (args.data, dump, ion_mode, z_adlayer_abs, cutoff, args.skip_frames)
        for dump in args.trajectories
    ]
    results = run_tasks(analyze_coordination_dump, tasks, args.nprocs, label="coordination")
    check_global_nion(results, "coordination")

    total = {
        "frames": 0,
        "frames_with_adlayer": 0,
        "adlayer_waters": 0,
        "sum_iw_pair_count": 0,
        "sum_iw_water_count": 0,
        "sum_selected_ion_count": 0,
        "sum_selected_ion_total_cn": 0,
        "water_triplets_sum": 0,
        "ion_atoms_sum": 0,
        "dumps": 0,
    }
    for result in results:
        total = add_counts(total, result)

    if total["frames"] == 0:
        sys.exit("No frames processed for coordination.")

    avg_iw_pair_count = total["sum_iw_pair_count"] / total["frames"]
    avg_iw_water_count = total["sum_iw_water_count"] / total["frames"]
    avg_selected_ion_count = total["sum_selected_ion_count"] / total["frames"]
    avg_selected_ion_total_cn = total["sum_selected_ion_total_cn"] / total["frames"]
    mean_total_cn_per_selected_ion = safe_fraction(
        total["sum_selected_ion_total_cn"],
        total["sum_selected_ion_count"],
    )

    print(
        f"[TOTAL] coordination: dumps={total['dumps']}, frames={total['frames']}, "
        f"frames_with_adlayer={total['frames_with_adlayer']}, "
        f"adlayer_waters={total['adlayer_waters']}, "
        f"avg_iw_pairs={avg_iw_pair_count:.8f}, "
        f"avg_iw_waters={avg_iw_water_count:.8f}, "
        f"avg_selected_ions={avg_selected_ion_count:.8f}, "
        f"avg_selected_ion_total_cn={avg_selected_ion_total_cn:.8f}, "
        f"CN_total_IF={mean_total_cn_per_selected_ion:.8f}",
        file=sys.stderr,
    )

    out_path = f"{args.header}_coordination_{ion_mode}.txt"
    with open(out_path, "w") as f:
        f.write("# interfacial-water / ion coordination statistics for Figure S9 and Figure S10\n")
        f.write(f"# iontype={ion_mode}\n")
        f.write(f"# pair={pair_label}\n")
        f.write(f"# ion_type={ION_TYPE}\n")
        f.write(f"# cutoff_A={cutoff:.8f}\n")
        f.write(f"# z_adlayer_input={args.z_adlayer:.8f}\n")
        f.write(f"# z_adlayer_abs={z_adlayer_abs:.8f}\n")
        f.write(f"# skip_frames_per_dump={args.skip_frames}\n")
        f.write(f"# frames={total['frames']}\n")
        f.write(f"# dumps={total['dumps']}\n")
        f.write(f"# avg_water_triplets_per_dump={safe_fraction(total['water_triplets_sum'], total['dumps']):.8f}\n")
        f.write(f"# avg_ion_atoms_per_dump={safe_fraction(total['ion_atoms_sum'], total['dumps']):.8f}\n")
        if ion_mode == "Na":
            f.write("# interfacial_pair_count=O(adlayer water)-Na pairs\n")
            f.write("# interfacial_water_count=unique adlayer water molecules with at least one O-Na coordination\n")
            f.write("# selected_ion_total_cn=all O-Na coordination pairs for ions touching adlayer water\n")
        else:
            f.write("# interfacial_pair_count=H(adlayer water)-F pairs; two H atoms are counted separately\n")
            f.write("# interfacial_water_count=unique adlayer water molecules with at least one H-F coordination\n")
            f.write("# selected_ion_total_cn=all H-F coordination pairs for ions touching adlayer water\n")
        f.write("# quantity value\n")
        f.write(f"avg_interfacial_pair_count_per_frame {avg_iw_pair_count:.10f}\n")
        f.write(f"avg_interfacial_water_count_per_frame_Gamma_total_IF {avg_iw_water_count:.10f}\n")
        f.write(f"avg_selected_ion_count_per_frame {avg_selected_ion_count:.10f}\n")
        f.write(f"avg_selected_ion_total_cn_per_frame {avg_selected_ion_total_cn:.10f}\n")
        f.write(f"mean_total_cn_per_selected_ion_CN_total_IF {mean_total_cn_per_selected_ion:.10f}\n")
        f.write(f"total_interfacial_pair_count {total['sum_iw_pair_count']}\n")
        f.write(f"total_interfacial_water_count {total['sum_iw_water_count']}\n")
        f.write(f"total_selected_ion_count {total['sum_selected_ion_count']}\n")
        f.write(f"total_selected_ion_total_cn {total['sum_selected_ion_total_cn']}\n")

    print(f"[DONE] wrote {out_path}")


def rdf_neighbor_for_ion(ion_mode: str) -> Tuple[str, int, str]:
    if ion_mode == "Na":
        return "O", O_TYPE, "Na-O"
    if ion_mode == "F":
        return "H", min(H_TYPES), "F-H"
    raise ValueError("IONTYPE must be Na or F")


def make_rdf_bins(rmax: float, dr: float) -> np.ndarray:
    return np.arange(0.0, rmax + dr, dr, dtype=np.float64)


def update_rdf_histogram(hist: np.ndarray, pos_neighbor: np.ndarray, pos_ion: np.ndarray, box, bins: np.ndarray, rmax: float):
    if pos_neighbor.shape[0] == 0 or pos_ion.shape[0] == 0:
        return

    lx, ly = float(box[0]), float(box[1])
    chunk_ion = 256
    for i0 in range(0, pos_ion.shape[0], chunk_ion):
        i1 = min(i0 + chunk_ion, pos_ion.shape[0])
        disp = pos_neighbor[None, :, :] - pos_ion[i0:i1, None, :]
        disp[..., 0] -= lx * np.rint(disp[..., 0] / lx)
        disp[..., 1] -= ly * np.rint(disp[..., 1] / ly)
        dist = np.linalg.norm(disp, axis=2).ravel()
        dist = dist[dist <= rmax]
        if dist.size:
            counts, _ = np.histogram(dist, bins=bins)
            hist += counts


def analyze_rdf_dump(args):
    data_path, dump_path, ion_mode, neighbor_type, z_low_abs, z_high_abs, rmax, dr, skip_frames = args
    print(f"[INFO] rdf: {dump_path} ({ion_mode})", file=sys.stderr)

    u = get_mda().Universe(data_path, dump_path, format="LAMMPSDUMP")
    neighbors = u.select_atoms(f"type {neighbor_type}")
    ions = u.select_atoms(f"type {ION_TYPE}")
    print(
        f"[ATOMS] rdf: {dump_path}: neighbor_type={neighbor_type}, "
        f"neighbor_atoms={neighbors.n_atoms}, ion_atoms={ions.n_atoms}",
        file=sys.stderr,
    )
    if neighbors.n_atoms == 0:
        raise RuntimeError(f"No RDF neighbor atoms found for type={neighbor_type} in {dump_path}")
    if ions.n_atoms == 0:
        raise RuntimeError(f"No RDF ion atoms found for type={ION_TYPE} in {dump_path}")

    bins = make_rdf_bins(rmax, dr)
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    nframes = 0
    sum_neighbor = 0
    sum_ion = 0
    sum_volume = 0.0

    for local_frame, ts in enumerate(u.trajectory):
        if local_frame < skip_frames:
            continue

        dims = ts.dimensions
        lx, ly = float(dims[0]), float(dims[1])
        pos_neighbor_all = neighbors.positions
        pos_ion_all = ions.positions

        z_neighbor = pos_neighbor_all[:, 2]
        z_ion = pos_ion_all[:, 2]
        neighbor_mask = (z_neighbor > z_low_abs - rmax) & (z_neighbor < z_high_abs + rmax)
        ion_mask = (z_ion > z_low_abs) & (z_ion < z_high_abs)

        pos_neighbor = pos_neighbor_all[neighbor_mask]
        pos_ion = pos_ion_all[ion_mask]
        update_rdf_histogram(hist, pos_neighbor, pos_ion, dims, bins, rmax)

        nframes += 1
        sum_neighbor += int(pos_neighbor.shape[0])
        sum_ion += int(pos_ion.shape[0])
        sum_volume += lx * ly * (z_high_abs - z_low_abs + 2.0 * rmax)

    print(
        f"[SUMMARY] rdf: {dump_path}: frames={nframes}, "
        f"sum_ion_in_region={sum_ion}, sum_neighbor_ext={sum_neighbor}, "
        f"raw_pairs={int(hist.sum())}",
        file=sys.stderr,
    )
    return {
        "hist": hist,
        "nframes": nframes,
        "sum_neighbor": sum_neighbor,
        "sum_ion": sum_ion,
        "sum_volume": sum_volume,
        "nion_total": int(ions.n_atoms),
        "neighbor_total": int(neighbors.n_atoms),
        "dump": dump_path,
    }


def run_rdf(args):
    try:
        ion_mode = normalize_ion_mode(args.iontype)
        neighbor_label, neighbor_type, pair_label = rdf_neighbor_for_ion(ion_mode)
    except ValueError as exc:
        sys.exit(str(exc))

    z_low_abs = abs_z(args.z_low)
    z_high_abs = abs_z(args.z_high)
    if z_high_abs <= z_low_abs:
        sys.exit("RDF Z_HIGH must be larger than Z_LOW.")
    if args.rmax <= 0.0:
        sys.exit("RDF R_MAX must be > 0.")
    if args.dr <= 0.0:
        sys.exit("RDF --dr must be > 0.")

    print(
        f"[SETUP] rdf: header={args.header}, pair={pair_label}, iontype={ion_mode}, "
        f"z_low_input={args.z_low:.6f}, z_high_input={args.z_high:.6f}, "
        f"z_low_abs={z_low_abs:.6f}, z_high_abs={z_high_abs:.6f}, "
        f"rmax={args.rmax:.6f}, dr={args.dr:.6f}, skip_frames={args.skip_frames}, data={args.data}",
        file=sys.stderr,
    )
    print(
        "[NOTE] rdf: all input trajectories are globally pooled; type3 ion counts may differ between trajectories.",
        file=sys.stderr,
    )

    tasks = [
        (
            args.data,
            dump,
            ion_mode,
            neighbor_type,
            z_low_abs,
            z_high_abs,
            args.rmax,
            args.dr,
            args.skip_frames,
        )
        for dump in args.trajectories
    ]
    results = run_tasks(analyze_rdf_dump, tasks, args.nprocs, label="rdf")

    bins = make_rdf_bins(args.rmax, args.dr)
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    nframes = 0
    sum_neighbor = 0
    sum_ion = 0
    sum_volume = 0.0

    for result in results:
        hist += result["hist"]
        nframes += result["nframes"]
        sum_neighbor += result["sum_neighbor"]
        sum_ion += result["sum_ion"]
        sum_volume += result["sum_volume"]

    if nframes == 0:
        sys.exit("No frames processed for RDF.")
    if sum_neighbor == 0:
        sys.exit("No RDF neighbor atoms selected in the extended bulk region.")
    if sum_ion == 0:
        sys.exit("No RDF ion atoms selected in the bulk region.")

    rho_neighbor = sum_neighbor / sum_volume
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    shell_volume = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    g_r = hist.astype(np.float64) / (sum_ion * rho_neighbor * shell_volume)

    print(
        f"[TOTAL] rdf: trajectories={len(results)}, frames={nframes}, "
        f"sum_ion_in_region={sum_ion}, sum_neighbor_ext={sum_neighbor}, "
        f"rho_neighbor={rho_neighbor:.8e}, raw_pairs={int(hist.sum())}",
        file=sys.stderr,
    )

    out_path = f"{args.header}_rdf_{ion_mode}_{neighbor_label}_bulkregion.txt"
    with open(out_path, "w") as f:
        f.write("# bulk-region RDF (global pooling over all input trajectories)\n")
        f.write("# x/y PBC, z non-PBC\n")
        f.write(f"# pair={pair_label}\n")
        f.write(f"# ion_type={ION_TYPE}\n")
        f.write(f"# neighbor_type={neighbor_type}\n")
        f.write(f"# ion_region_abs_A={z_low_abs:.8f} {z_high_abs:.8f}\n")
        f.write(f"# neighbor_region_abs_A={z_low_abs - args.rmax:.8f} {z_high_abs + args.rmax:.8f}\n")
        f.write(f"# z_low_input={args.z_low:.8f}\n")
        f.write(f"# z_high_input={args.z_high:.8f}\n")
        f.write(f"# rmax_A={args.rmax:.8f}\n")
        f.write(f"# dr_A={args.dr:.8f}\n")
        f.write(f"# skip_frames_per_dump={args.skip_frames}\n")
        f.write(f"# frames={nframes}\n")
        f.write(f"# rho_neighbor={rho_neighbor:.10e}\n")
        f.write(f"# sum_neighbor={sum_neighbor}\n")
        f.write(f"# sum_ion={sum_ion}\n")
        for idx, result in enumerate(results, start=1):
            f.write(
                f"# dump{idx}={result['dump']} nion_total={result['nion_total']} "
                f"neighbor_total={result['neighbor_total']} frames={result['nframes']}\n"
            )
        f.write("# r_A g_r raw_counts\n")
        for r, g, count in zip(r_centers, g_r, hist):
            f.write(f"{r:10.4f} {g:14.8e} {int(count)}\n")

    print(f"[DONE] wrote {out_path}")


def read_first_box_bounds_orthogonal(dump_path: str) -> Tuple[float, float, float, float, float, float]:
    with open(dump_path, "r") as f:
        for line in f:
            if line.startswith("ITEM: BOX BOUNDS"):
                if any(tok in line for tok in ("xy", "xz", "yz")):
                    raise NotImplementedError("Triclinic box bounds are not supported.")
                xlo, xhi = [float(v) for v in f.readline().split()[:2]]
                ylo, yhi = [float(v) for v in f.readline().split()[:2]]
                zlo, zhi = [float(v) for v in f.readline().split()[:2]]
                return xlo, xhi, ylo, yhi, zlo, zhi
    raise RuntimeError(f"BOX BOUNDS not found in {dump_path}")


def analyze_waterdensity_dump(args):
    data_path, dump_path, z_edges_rel, slab_vol_cm3, skip_frames = args
    print(f"[INFO] watermassdensity: {dump_path}", file=sys.stderr)

    u = get_mda().Universe(data_path, dump_path, format="LAMMPSDUMP")
    oxy = u.select_atoms(f"type {O_TYPE}")
    ions = u.select_atoms(f"type {ION_TYPE}")
    expected_nion = int(ions.n_atoms)
    if oxy.n_atoms == 0:
        raise RuntimeError(f"No oxygen atoms found for type={O_TYPE} in {dump_path}")
    print(f"[OXYGEN] watermassdensity: {dump_path}: oxygen_atoms={oxy.n_atoms}", file=sys.stderr)
    print(f"[IONS] watermassdensity: {dump_path}: type{ION_TYPE}_atoms={expected_nion}", file=sys.stderr)

    rho_sum = np.zeros(len(z_edges_rel) - 1, dtype=float)
    nframes = 0

    for local_frame, ts in enumerate(u.trajectory):
        if local_frame < skip_frames:
            continue

        check_nion_frame(ions, expected_nion, f"watermassdensity, dump={dump_path}, frame={ts.frame}")
        z_rel = oxy.positions[:, 2] - Z_ORIGIN
        counts, _ = np.histogram(z_rel, bins=z_edges_rel)
        rho_sum += counts.astype(float) * MASS_WATER_G / slab_vol_cm3
        nframes += 1

    print(
        f"[SUMMARY] watermassdensity: {dump_path}: frames={nframes}, "
        f"oxygen_atoms={oxy.n_atoms}",
        file=sys.stderr,
    )
    return {
        "rho_sum": rho_sum,
        "frames": nframes,
        "oxygen_atoms": int(oxy.n_atoms),
        "nion_total": expected_nion,
        "dumps": 1,
    }


def run_watermassdensity(args):
    if args.water_z_max <= args.water_z_min:
        sys.exit("--water-z-max must be larger than --water-z-min.")

    xlo, xhi, ylo, yhi, zlo, zhi = read_first_box_bounds_orthogonal(args.trajectories[0])
    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo
    nbins = int(np.ceil((args.water_z_max - args.water_z_min) / BIN_DZ_A))
    z_edges_rel = args.water_z_min + np.arange(nbins + 1, dtype=float) * BIN_DZ_A
    z_edges_rel[-1] = args.water_z_max
    z_centers_rel = 0.5 * (z_edges_rel[:-1] + z_edges_rel[1:])
    slab_vol_cm3 = lx * ly * BIN_DZ_A * ANG3_TO_CM3

    print(
        f"[SETUP] watermassdensity: box=({lx:.6f}, {ly:.6f}, {lz:.6f}) Ang, "
        f"area={lx * ly:.6f} Ang^2, bins={nbins}, dz={BIN_DZ_A:.3f} Ang, "
        f"z_rel_range=({args.water_z_min:.3f}, {args.water_z_max:.3f}), "
        f"skip_frames={args.skip_frames}",
        file=sys.stderr,
    )

    tasks = [
        (args.data, dump, z_edges_rel, slab_vol_cm3, args.skip_frames)
        for dump in args.trajectories
    ]
    results = run_tasks(analyze_waterdensity_dump, tasks, args.nprocs, label="watermassdensity")
    check_global_nion(results, "watermassdensity")

    rho_total = np.zeros(nbins, dtype=float)
    frames = 0
    oxygen_atoms_sum = 0
    dumps = 0
    for result in results:
        rho_total += result["rho_sum"]
        frames += result["frames"]
        oxygen_atoms_sum += result["oxygen_atoms"]
        dumps += result["dumps"]

    if frames == 0:
        sys.exit("No frames processed for watermassdensity.")

    print(
        f"[TOTAL] watermassdensity: dumps={dumps}, frames={frames}, "
        f"avg_oxygen_atoms_per_dump={safe_fraction(oxygen_atoms_sum, dumps):.2f}, bins={nbins}",
        file=sys.stderr,
    )

    rho_avg = rho_total / float(frames)
    out_path = f"{args.header}_watermassdensity.txt"
    with open(out_path, "w") as f:
        f.write("# z-resolved water mass density\n")
        f.write(f"# dz_A={BIN_DZ_A:.8f}\n")
        f.write(f"# z_origin_abs={Z_ORIGIN:.8f}\n")
        f.write(f"# z_rel_range_A={args.water_z_min:.8f} {args.water_z_max:.8f}\n")
        f.write(f"# frames={frames}\n")
        f.write("# z_rel_center_A rho_g_mL\n")
        for zc, rho in zip(z_centers_rel, rho_avg):
            f.write(f"{zc:.6f} {rho:.8f}\n")

    print(f"[DONE] wrote {out_path}")


def analyze_combined_dump(args):
    data_path, dump_path, cfg = args
    modes = set(cfg["modes"])
    non_rdf_modes = modes - {"rdf"}
    need_waters = bool(modes & {"angle1", "angle2", "coordination"})

    print(f"[INFO] combined-run: {dump_path}: modes={','.join(cfg['modes'])}", file=sys.stderr)

    u = get_mda().Universe(data_path, dump_path, format="LAMMPSDUMP")
    ions = u.select_atoms(f"type {ION_TYPE}")
    expected_nion = int(ions.n_atoms)
    if ions.n_atoms == 0 and modes & {"iondist", "coordination", "rdf"}:
        raise RuntimeError(f"No ions found for type={ION_TYPE} in {dump_path}")

    waters = None
    if need_waters:
        use_first_trajectory_frame(u, dump_path, "combined-run")
        waters = build_water_triplets(u)
        print(
            f"[TRIPLETS] combined-run: {dump_path}: water_triplets={len(waters)}",
            file=sys.stderr,
        )

    oxy_density = None
    if "watermassdensity" in modes:
        oxy_density = u.select_atoms(f"type {O_TYPE}")
        if oxy_density.n_atoms == 0:
            raise RuntimeError(f"No oxygen atoms found for type={O_TYPE} in {dump_path}")
        print(
            f"[OXYGEN] combined-run: {dump_path}: oxygen_atoms={oxy_density.n_atoms}",
            file=sys.stderr,
        )

    rdf_neighbors = None
    if "rdf" in modes:
        rdf_neighbors = u.select_atoms(f"type {cfg['rdf_neighbor_type']}")
        if rdf_neighbors.n_atoms == 0:
            raise RuntimeError(
                f"No RDF neighbor atoms found for type={cfg['rdf_neighbor_type']} in {dump_path}"
            )
        print(
            f"[ATOMS] combined-run rdf: {dump_path}: "
            f"neighbor_type={cfg['rdf_neighbor_type']}, neighbor_atoms={rdf_neighbors.n_atoms}, "
            f"ion_atoms={ions.n_atoms}",
            file=sys.stderr,
        )

    print(
        f"[IONS] combined-run: {dump_path}: type{ION_TYPE}_atoms={expected_nion}",
        file=sys.stderr,
    )

    result = {
        "dump": dump_path,
        "nion_total": expected_nion,
        "case_index": cfg.get("case_index", 0),
        "header": cfg.get("header", ""),
    }

    if "angle1" in modes:
        result["angle1"] = {
            "total": 0,
            "flat": 0,
            "h_down": 0,
            "h_up": 0,
            "up_down_mixed": 0,
            "two_h_up": 0,
            "one_h_up": 0,
            "one_h_down": 0,
            "two_h_down": 0,
            "frames": 0,
            "frames_with_adlayer": 0,
            "adlayer_waters": 0,
            "water_triplets_sum": len(waters),
            "dumps": 1,
        }

    if "angle2" in modes:
        result["angle2"] = {
            "phi_hist": np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64),
            "chi_hist": np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64),
            "phi_sum": 0.0,
            "phi_sumsq": 0.0,
            "phi_n": 0,
            "chi_sum": 0.0,
            "chi_sumsq": 0.0,
            "chi_n": 0,
            "frames": 0,
            "frames_with_adlayer": 0,
            "adlayer_waters": 0,
            "water_triplets_sum": len(waters),
            "dumps": 1,
        }

    if "iondist" in modes:
        result["iondist"] = {
            "ihl": 0,
            "ohl": 0,
            "difl": 0,
            "total": 0,
            "frames": 0,
            "dumps": 1,
        }

    if "coordination" in modes:
        result["coordination"] = {
            "frames": 0,
            "frames_with_adlayer": 0,
            "adlayer_waters": 0,
            "sum_iw_pair_count": 0,
            "sum_iw_water_count": 0,
            "sum_selected_ion_count": 0,
            "sum_selected_ion_total_cn": 0,
            "water_triplets_sum": len(waters),
            "ion_atoms_sum": expected_nion,
            "dumps": 1,
        }

    if "rdf" in modes:
        result["rdf"] = {
            "hist": np.zeros(len(cfg["rdf_bins"]) - 1, dtype=np.int64),
            "nframes": 0,
            "sum_neighbor": 0,
            "sum_ion": 0,
            "sum_volume": 0.0,
            "nion_total": expected_nion,
            "neighbor_total": int(rdf_neighbors.n_atoms),
            "dump": dump_path,
        }

    if "watermassdensity" in modes:
        result["watermassdensity"] = {
            "rho_sum": np.zeros(cfg["water_nbins"], dtype=float),
            "frames": 0,
            "oxygen_atoms": int(oxy_density.n_atoms),
            "dumps": 1,
        }

    for local_frame, ts in enumerate(u.trajectory):
        if local_frame < cfg["skip_frames"]:
            continue

        if non_rdf_modes:
            check_nion_frame(ions, expected_nion, f"combined-run, dump={dump_path}, frame={ts.frame}")

        box = ts.dimensions[:3]
        oxygen = v1 = v2 = adlayer_mask = None
        n_adlayer = 0
        if need_waters:
            oxygen, v1, v2 = frame_water_vectors(
                ts.positions,
                waters,
                box,
                context=f"combined-run, dump={dump_path}, frame={ts.frame}",
            )
            adlayer_mask = oxygen[:, 2] < cfg["z_adlayer_abs"]
            n_adlayer = int(np.count_nonzero(adlayer_mask))

        if "angle1" in modes:
            a1 = result["angle1"]
            a1["frames"] += 1
            if n_adlayer > 0:
                a1["frames_with_adlayer"] += 1
                a1["adlayer_waters"] += n_adlayer
                th1, th2 = oh_angles_from_vectors(v1[adlayer_mask], v2[adlayer_mask])
                result["angle1"] = add_counts(a1, classify_oh_angles(th1, th2))

        if "angle2" in modes:
            a2 = result["angle2"]
            a2["frames"] += 1
            if n_adlayer > 0:
                a2["frames_with_adlayer"] += 1
                a2["adlayer_waters"] += n_adlayer
                phi, chi = bisector_plane_angles(v1[adlayer_mask], v2[adlayer_mask])
                if phi.size:
                    hist, _ = np.histogram(phi, bins=ANGLE_BINS)
                    a2["phi_hist"] += hist
                    a2["phi_sum"] += float(phi.sum())
                    a2["phi_sumsq"] += float(np.square(phi).sum())
                    a2["phi_n"] += int(phi.size)
                if chi.size:
                    hist, _ = np.histogram(chi, bins=ANGLE_BINS)
                    a2["chi_hist"] += hist
                    a2["chi_sum"] += float(chi.sum())
                    a2["chi_sumsq"] += float(np.square(chi).sum())
                    a2["chi_n"] += int(chi.size)

        if "iondist" in modes:
            iondist = result["iondist"]
            z = ions.positions[:, 2]
            n_ihl = int(np.count_nonzero(z < cfg["z_ihl_abs"]))
            n_ohl = int(np.count_nonzero((z >= cfg["z_ihl_abs"]) & (z < cfg["z_ohl_abs"])))
            n_total = int(ions.n_atoms)
            iondist["ihl"] += n_ihl
            iondist["ohl"] += n_ohl
            iondist["difl"] += n_total - n_ihl - n_ohl
            iondist["total"] += n_total
            iondist["frames"] += 1

        if "coordination" in modes:
            coord = result["coordination"]
            h1 = oxygen + v1
            h2 = oxygen + v2
            ion_pos = ions.positions
            if cfg["ion_mode"] == "Na":
                iw_site_pos = oxygen[adlayer_mask]
                all_site_pos = oxygen
                site_water_ids = np.arange(oxygen.shape[0], dtype=int)[adlayer_mask]
            else:
                adlayer_water_ids = np.arange(oxygen.shape[0], dtype=int)[adlayer_mask]
                h1_adlayer = h1[adlayer_mask]
                h2_adlayer = h2[adlayer_mask]
                iw_site_pos = (
                    np.vstack((h1_adlayer, h2_adlayer))
                    if h1_adlayer.size > 0
                    else np.empty((0, 3), dtype=float)
                )
                site_water_ids = (
                    np.concatenate((adlayer_water_ids, adlayer_water_ids))
                    if adlayer_water_ids.size > 0
                    else np.empty(0, dtype=int)
                )
                all_site_pos = np.vstack((h1, h2))

            hit_iw = pair_hits_xy(iw_site_pos, ion_pos, box, cfg["coord_cutoff"])
            iw_pair_count = int(np.count_nonzero(hit_iw))
            if hit_iw.shape[0] > 0:
                iw_water_count = int(np.unique(site_water_ids[np.any(hit_iw, axis=1)]).size)
            else:
                iw_water_count = 0
            selected_ion_mask = (
                np.any(hit_iw, axis=0)
                if hit_iw.shape[1] > 0
                else np.zeros(ion_pos.shape[0], dtype=bool)
            )
            selected_ion_count = int(np.count_nonzero(selected_ion_mask))
            if selected_ion_count > 0:
                hit_all = pair_hits_xy(all_site_pos, ion_pos[selected_ion_mask], box, cfg["coord_cutoff"])
                selected_ion_total_cn = int(np.count_nonzero(hit_all))
            else:
                selected_ion_total_cn = 0

            coord["frames"] += 1
            if n_adlayer > 0:
                coord["frames_with_adlayer"] += 1
            coord["adlayer_waters"] += n_adlayer
            coord["sum_iw_pair_count"] += iw_pair_count
            coord["sum_iw_water_count"] += iw_water_count
            coord["sum_selected_ion_count"] += selected_ion_count
            coord["sum_selected_ion_total_cn"] += selected_ion_total_cn

        if "rdf" in modes:
            rdf = result["rdf"]
            dims = ts.dimensions
            lx, ly = float(dims[0]), float(dims[1])
            pos_neighbor_all = rdf_neighbors.positions
            pos_ion_all = ions.positions
            z_neighbor = pos_neighbor_all[:, 2]
            z_ion = pos_ion_all[:, 2]
            neighbor_mask = (
                (z_neighbor > cfg["rdf_z_low_abs"] - cfg["rdf_rmax"])
                & (z_neighbor < cfg["rdf_z_high_abs"] + cfg["rdf_rmax"])
            )
            ion_mask = (z_ion > cfg["rdf_z_low_abs"]) & (z_ion < cfg["rdf_z_high_abs"])
            pos_neighbor = pos_neighbor_all[neighbor_mask]
            pos_ion = pos_ion_all[ion_mask]
            update_rdf_histogram(rdf["hist"], pos_neighbor, pos_ion, dims, cfg["rdf_bins"], cfg["rdf_rmax"])
            rdf["nframes"] += 1
            rdf["sum_neighbor"] += int(pos_neighbor.shape[0])
            rdf["sum_ion"] += int(pos_ion.shape[0])
            rdf["sum_volume"] += lx * ly * (cfg["rdf_z_high_abs"] - cfg["rdf_z_low_abs"] + 2.0 * cfg["rdf_rmax"])

        if "watermassdensity" in modes:
            wd = result["watermassdensity"]
            z_rel = oxy_density.positions[:, 2] - Z_ORIGIN
            counts, _ = np.histogram(z_rel, bins=cfg["water_z_edges_rel"])
            wd["rho_sum"] += counts.astype(float) * MASS_WATER_G / cfg["water_slab_vol_cm3"]
            wd["frames"] += 1

    if "angle1" in modes:
        a1 = result["angle1"]
        print(
            f"[SUMMARY] combined-run angle1: {dump_path}: frames={a1['frames']}, "
            f"adlayer_waters={a1['adlayer_waters']}, classified={a1['total']}",
            file=sys.stderr,
        )
    if "angle2" in modes:
        a2 = result["angle2"]
        print(
            f"[SUMMARY] combined-run angle2: {dump_path}: frames={a2['frames']}, "
            f"phi_count={a2['phi_n']}, chi_count={a2['chi_n']}",
            file=sys.stderr,
        )
    if "iondist" in modes and result["iondist"]["frames"] > 0:
        iondist = result["iondist"]
        print(
            f"[SUMMARY] combined-run iondist: {dump_path}: frames={iondist['frames']}, "
            f"avg_ihl={iondist['ihl'] / iondist['frames']:.4f}, "
            f"avg_ohl={iondist['ohl'] / iondist['frames']:.4f}, "
            f"avg_difl={iondist['difl'] / iondist['frames']:.4f}",
            file=sys.stderr,
        )
    if "coordination" in modes and result["coordination"]["frames"] > 0:
        coord = result["coordination"]
        print(
            f"[SUMMARY] combined-run coordination: {dump_path}: frames={coord['frames']}, "
            f"avg_selected_ions={coord['sum_selected_ion_count'] / coord['frames']:.4f}, "
            f"CN_total_IF={safe_fraction(coord['sum_selected_ion_total_cn'], coord['sum_selected_ion_count']):.4f}",
            file=sys.stderr,
        )
    if "rdf" in modes:
        rdf = result["rdf"]
        print(
            f"[SUMMARY] combined-run rdf: {dump_path}: frames={rdf['nframes']}, "
            f"sum_ion_in_region={rdf['sum_ion']}, sum_neighbor_ext={rdf['sum_neighbor']}, "
            f"raw_pairs={int(rdf['hist'].sum())}",
            file=sys.stderr,
        )
    if "watermassdensity" in modes:
        wd = result["watermassdensity"]
        print(
            f"[SUMMARY] combined-run watermassdensity: {dump_path}: frames={wd['frames']}, "
            f"oxygen_atoms={wd['oxygen_atoms']}",
            file=sys.stderr,
        )

    return result


def prepare_combined_cfg(args, modes):
    cfg = {"modes": list(modes), "skip_frames": args.skip_frames}
    cfg["case_index"] = getattr(args, "case_index", 0)
    cfg["header"] = args.header
    cfg["data"] = args.data

    if any(mode in modes for mode in ("angle1", "angle2", "coordination")):
        cfg["z_adlayer_abs"] = abs_z(args.z_adlayer)

    if any(mode in modes for mode in ("iondist", "coordination", "rdf")):
        try:
            cfg["ion_mode"] = normalize_ion_mode(args.iontype)
        except ValueError as exc:
            sys.exit(str(exc))

    if "iondist" in modes:
        cfg["ion_sign"] = ion_sign(args.iontype)
        cfg["z_ihl_abs"] = abs_z(args.z_ihl)
        cfg["z_ohl_abs"] = abs_z(args.z_ohl)
        if cfg["z_ohl_abs"] <= cfg["z_ihl_abs"]:
            sys.exit("Z_OHL must be larger than Z_IHL.")

    if "coordination" in modes:
        cfg["coord_cutoff"] = float(args.coord_cutoff)
        if cfg["coord_cutoff"] <= 0.0:
            sys.exit("coordination cutoff must be > 0.")

    if "rdf" in modes:
        neighbor_label, neighbor_type, pair_label = rdf_neighbor_for_ion(cfg["ion_mode"])
        cfg["rdf_neighbor_label"] = neighbor_label
        cfg["rdf_neighbor_type"] = neighbor_type
        cfg["rdf_pair_label"] = pair_label
        cfg["rdf_z_low_abs"] = abs_z(args.rdf_z_low)
        cfg["rdf_z_high_abs"] = abs_z(args.rdf_z_high)
        cfg["rdf_rmax"] = float(args.rdf_rmax)
        cfg["rdf_dr"] = float(args.rdf_dr)
        if cfg["rdf_z_high_abs"] <= cfg["rdf_z_low_abs"]:
            sys.exit("RDF Z_HIGH must be larger than Z_LOW.")
        if cfg["rdf_rmax"] <= 0.0:
            sys.exit("RDF R_MAX must be > 0.")
        if cfg["rdf_dr"] <= 0.0:
            sys.exit("RDF --dr must be > 0.")
        cfg["rdf_bins"] = make_rdf_bins(cfg["rdf_rmax"], cfg["rdf_dr"])

    if "watermassdensity" in modes:
        if args.water_z_max <= args.water_z_min:
            sys.exit("--water-z-max must be larger than --water-z-min.")

        xlo, xhi, ylo, yhi, _zlo, _zhi = read_first_box_bounds_orthogonal(args.trajectories[0])
        lx = xhi - xlo
        ly = yhi - ylo
        nbins = int(np.ceil((args.water_z_max - args.water_z_min) / BIN_DZ_A))
        z_edges_rel = args.water_z_min + np.arange(nbins + 1, dtype=float) * BIN_DZ_A
        z_edges_rel[-1] = args.water_z_max
        cfg["water_nbins"] = nbins
        cfg["water_z_edges_rel"] = z_edges_rel
        cfg["water_z_min"] = args.water_z_min
        cfg["water_z_max"] = args.water_z_max
        cfg["water_z_centers_rel"] = 0.5 * (z_edges_rel[:-1] + z_edges_rel[1:])
        cfg["water_slab_vol_cm3"] = lx * ly * BIN_DZ_A * ANG3_TO_CM3

    return cfg


def write_rdf_from_combined_results(header, cfg, results, z_low_input, z_high_input, skip_frames, label="rdf"):
    bins = cfg["rdf_bins"]
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    nframes = sum_neighbor = sum_ion = 0
    sum_volume = 0.0
    for result in results:
        rdf = result["rdf"]
        hist += rdf["hist"]
        nframes += rdf["nframes"]
        sum_neighbor += rdf["sum_neighbor"]
        sum_ion += rdf["sum_ion"]
        sum_volume += rdf["sum_volume"]
    if nframes == 0:
        sys.exit("No frames processed for RDF.")
    if sum_neighbor == 0:
        sys.exit("No RDF neighbor atoms selected in the extended bulk region.")
    if sum_ion == 0:
        sys.exit("No RDF ion atoms selected in the bulk region.")
    rho_neighbor = sum_neighbor / sum_volume
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    shell_volume = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    g_r = hist.astype(np.float64) / (sum_ion * rho_neighbor * shell_volume)
    print(
        f"[TOTAL] {label}: trajectories={len(results)}, frames={nframes}, "
        f"sum_ion_in_region={sum_ion}, sum_neighbor_ext={sum_neighbor}, "
        f"rho_neighbor={rho_neighbor:.8e}, raw_pairs={int(hist.sum())}",
        file=sys.stderr,
    )
    out_path = f"{header}_rdf_{cfg['ion_mode']}_{cfg['rdf_neighbor_label']}_bulkregion.txt"
    with open(out_path, "w") as f:
        f.write("# bulk-region RDF (global pooling over all input trajectories)\n")
        f.write("# x/y PBC, z non-PBC\n")
        f.write(f"# pair={cfg['rdf_pair_label']}\n")
        f.write(f"# ion_type={ION_TYPE}\n")
        f.write(f"# neighbor_type={cfg['rdf_neighbor_type']}\n")
        f.write(f"# ion_region_abs_A={cfg['rdf_z_low_abs']:.8f} {cfg['rdf_z_high_abs']:.8f}\n")
        f.write(
            f"# neighbor_region_abs_A={cfg['rdf_z_low_abs'] - cfg['rdf_rmax']:.8f} "
            f"{cfg['rdf_z_high_abs'] + cfg['rdf_rmax']:.8f}\n"
        )
        f.write(f"# z_low_input={z_low_input:.8f}\n")
        f.write(f"# z_high_input={z_high_input:.8f}\n")
        f.write(f"# rmax_A={cfg['rdf_rmax']:.8f}\n")
        f.write(f"# dr_A={cfg['rdf_dr']:.8f}\n")
        f.write(f"# skip_frames_per_dump={skip_frames}\n")
        f.write(f"# frames={nframes}\n")
        f.write(f"# rho_neighbor={rho_neighbor:.10e}\n")
        f.write(f"# sum_neighbor={sum_neighbor}\n")
        f.write(f"# sum_ion={sum_ion}\n")
        for idx, result in enumerate(results, start=1):
            rdf = result["rdf"]
            case_header = result.get("header", "")
            case_text = f" case={case_header}" if case_header else ""
            f.write(
                f"# dump{idx}={rdf['dump']}{case_text} nion_total={rdf['nion_total']} "
                f"neighbor_total={rdf['neighbor_total']} frames={rdf['nframes']}\n"
            )
        f.write("# r_A g_r raw_counts\n")
        for r, g, count in zip(r_centers, g_r, hist):
            f.write(f"{r:10.4f} {g:14.8e} {int(count)}\n")
    print(f"[DONE] wrote {out_path}")


def run_combined(args, modes):
    cfg = getattr(args, "_precomputed_cfg", None)
    if cfg is None:
        cfg = prepare_combined_cfg(args, modes)

    print("=== Selected DMI_edlanal modes ===")
    print("Modes:", ", ".join(modes))
    print(
        f"[SETUP] combined-run: header={args.header}, data={args.data}, "
        f"trajectories={len(args.trajectories)}, skip_frames={args.skip_frames}",
        file=sys.stderr,
    )
    if "rdf" in modes:
        print(
            "[NOTE] combined-run rdf: all input trajectories are globally pooled; "
            "type3 ion counts may differ when RDF is run alone.",
            file=sys.stderr,
        )

    results = getattr(args, "_precomputed_results", None)
    if results is None:
        tasks = [(args.data, dump, cfg) for dump in args.trajectories]
        results = run_tasks(analyze_combined_dump, tasks, args.nprocs, label="combined-run")

    non_rdf_modes = [mode for mode in modes if mode != "rdf"]
    if non_rdf_modes:
        check_global_nion(results, "combined-run non-RDF modes")

    if "angle1" in modes:
        total = {
            "total": 0,
            "flat": 0,
            "h_down": 0,
            "h_up": 0,
            "up_down_mixed": 0,
            "two_h_up": 0,
            "one_h_up": 0,
            "one_h_down": 0,
            "two_h_down": 0,
            "frames": 0,
            "frames_with_adlayer": 0,
            "adlayer_waters": 0,
            "water_triplets_sum": 0,
            "dumps": 0,
        }
        for result in results:
            total = add_counts(total, result["angle1"])
        if total["total"] == 0:
            sys.exit("No adlayer waters found for angle1.")
        print(
            f"[TOTAL] angle1: dumps={total['dumps']}, frames={total['frames']}, "
            f"adlayer_waters={total['adlayer_waters']}, classified={total['total']}",
            file=sys.stderr,
        )
        out_path = f"{args.header}_angle1_classes.txt"
        with open(out_path, "w") as f:
            f.write("# O-H bond orientation classes for water adlayer\n")
            f.write(f"# z_adlayer_input={args.z_adlayer:.8f}\n")
            f.write(f"# z_adlayer_abs={cfg['z_adlayer_abs']:.8f}\n")
            f.write("# class count fraction\n")
            for key in ("flat", "h_down", "h_up", "up_down_mixed"):
                f.write(f"{key} {total[key]} {safe_fraction(total[key], total['total']):.10f}\n")
            f.write("# fine_class count fraction\n")
            for key in ("two_h_up", "one_h_up", "one_h_down", "two_h_down"):
                f.write(f"{key} {total[key]} {safe_fraction(total[key], total['total']):.10f}\n")
            f.write(f"total {total['total']} 1.0000000000\n")
        print(f"[DONE] wrote {out_path}")

    if "angle2" in modes:
        phi_hist = np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64)
        chi_hist = np.zeros(len(ANGLE_BINS) - 1, dtype=np.int64)
        phi_sum = phi_sumsq = 0.0
        chi_sum = chi_sumsq = 0.0
        phi_n = chi_n = frames = frames_with_adlayer = adlayer_waters = 0
        water_triplets_sum = dumps = 0
        for result in results:
            a2 = result["angle2"]
            phi_hist += a2["phi_hist"]
            chi_hist += a2["chi_hist"]
            phi_sum += a2["phi_sum"]
            phi_sumsq += a2["phi_sumsq"]
            phi_n += a2["phi_n"]
            chi_sum += a2["chi_sum"]
            chi_sumsq += a2["chi_sumsq"]
            chi_n += a2["chi_n"]
            frames += a2["frames"]
            frames_with_adlayer += a2["frames_with_adlayer"]
            adlayer_waters += a2["adlayer_waters"]
            water_triplets_sum += a2["water_triplets_sum"]
            dumps += a2["dumps"]
        if phi_n == 0 and chi_n == 0:
            sys.exit("No adlayer waters found for angle2.")
        print(
            f"[TOTAL] angle2: dumps={dumps}, frames={frames}, "
            f"frames_with_adlayer={frames_with_adlayer}, adlayer_waters={adlayer_waters}, "
            f"phi_count={phi_n}, chi_count={chi_n}, "
            f"avg_water_triplets_per_dump={safe_fraction(water_triplets_sum, dumps):.2f}",
            file=sys.stderr,
        )
        phi_path = f"{args.header}_phi_bisector_hist.txt"
        chi_path = f"{args.header}_chi_plane_hist.txt"
        write_angle_hist(phi_path, "phi_bisector_vs_z", phi_hist, phi_sum, phi_sumsq, phi_n)
        write_angle_hist(chi_path, "chi_plane_normal_vs_z", chi_hist, chi_sum, chi_sumsq, chi_n)
        print(f"[DONE] wrote {phi_path}")
        print(f"[DONE] wrote {chi_path}")

    if "iondist" in modes:
        total = {"ihl": 0, "ohl": 0, "difl": 0, "total": 0, "frames": 0, "dumps": 0}
        for result in results:
            total = add_counts(total, result["iondist"])
        if total["frames"] == 0:
            sys.exit("No frames processed for iondist.")
        print(
            f"[TOTAL] iondist: dumps={total['dumps']}, frames={total['frames']}, "
            f"avg_ihl={total['ihl'] / total['frames']:.4f}, "
            f"avg_ohl={total['ohl'] / total['frames']:.4f}, "
            f"avg_difl={total['difl'] / total['frames']:.4f}",
            file=sys.stderr,
        )
        out_path = f"{args.header}_ion_charge_distribution.txt"
        with open(out_path, "w") as f:
            f.write("# ionic charge distribution\n")
            f.write(f"# iontype={args.iontype}\n")
            f.write(f"# ion_type={ION_TYPE}\n")
            f.write(f"# z_ihl_input={args.z_ihl:.8f}\n")
            f.write(f"# z_ohl_input={args.z_ohl:.8f}\n")
            f.write(f"# z_ihl_abs={cfg['z_ihl_abs']:.8f}\n")
            f.write(f"# z_ohl_abs={cfg['z_ohl_abs']:.8f}\n")
            f.write(f"# sigma_per_ion_uC_cm2={SIGMA_PER_ION:.8f}\n")
            f.write(f"# frames={total['frames']}\n")
            f.write("# region avg_count_per_frame fraction sigma_abs_uC_cm2 sigma_signed_uC_cm2 total_count\n")
            for region in ("ihl", "ohl", "difl"):
                avg = total[region] / total["frames"]
                frac = safe_fraction(total[region], total["total"])
                sigma_abs = avg * SIGMA_PER_ION
                sigma_signed = cfg["ion_sign"] * sigma_abs
                f.write(
                    f"{region} {avg:.10f} {frac:.10f} "
                    f"{sigma_abs:.10f} {sigma_signed:.10f} {total[region]}\n"
                )
            avg_total = total["total"] / total["frames"]
            f.write(f"total {avg_total:.10f} 1.0000000000 0.0000000000 0.0000000000 {total['total']}\n")
        print(f"[DONE] wrote {out_path}")

    if "coordination" in modes:
        total = {
            "frames": 0,
            "frames_with_adlayer": 0,
            "adlayer_waters": 0,
            "sum_iw_pair_count": 0,
            "sum_iw_water_count": 0,
            "sum_selected_ion_count": 0,
            "sum_selected_ion_total_cn": 0,
            "water_triplets_sum": 0,
            "ion_atoms_sum": 0,
            "dumps": 0,
        }
        for result in results:
            total = add_counts(total, result["coordination"])
        if total["frames"] == 0:
            sys.exit("No frames processed for coordination.")
        avg_iw_pair_count = total["sum_iw_pair_count"] / total["frames"]
        avg_iw_water_count = total["sum_iw_water_count"] / total["frames"]
        avg_selected_ion_count = total["sum_selected_ion_count"] / total["frames"]
        avg_selected_ion_total_cn = total["sum_selected_ion_total_cn"] / total["frames"]
        mean_total_cn_per_selected_ion = safe_fraction(
            total["sum_selected_ion_total_cn"],
            total["sum_selected_ion_count"],
        )
        pair_label = "Na-O" if cfg["ion_mode"] == "Na" else "F-H"
        print(
            f"[TOTAL] coordination: dumps={total['dumps']}, frames={total['frames']}, "
            f"avg_iw_waters={avg_iw_water_count:.8f}, "
            f"avg_selected_ions={avg_selected_ion_count:.8f}, "
            f"CN_total_IF={mean_total_cn_per_selected_ion:.8f}",
            file=sys.stderr,
        )
        out_path = f"{args.header}_coordination_{cfg['ion_mode']}.txt"
        with open(out_path, "w") as f:
            f.write("# interfacial-water / ion coordination statistics for Figure S9 and Figure S10\n")
            f.write(f"# iontype={cfg['ion_mode']}\n")
            f.write(f"# pair={pair_label}\n")
            f.write(f"# ion_type={ION_TYPE}\n")
            f.write(f"# cutoff_A={cfg['coord_cutoff']:.8f}\n")
            f.write(f"# z_adlayer_input={args.z_adlayer:.8f}\n")
            f.write(f"# z_adlayer_abs={cfg['z_adlayer_abs']:.8f}\n")
            f.write(f"# skip_frames_per_dump={args.skip_frames}\n")
            f.write(f"# frames={total['frames']}\n")
            f.write(f"# dumps={total['dumps']}\n")
            f.write(f"# avg_water_triplets_per_dump={safe_fraction(total['water_triplets_sum'], total['dumps']):.8f}\n")
            f.write(f"# avg_ion_atoms_per_dump={safe_fraction(total['ion_atoms_sum'], total['dumps']):.8f}\n")
            if cfg["ion_mode"] == "Na":
                f.write("# interfacial_pair_count=O(adlayer water)-Na pairs\n")
                f.write("# interfacial_water_count=unique adlayer water molecules with at least one O-Na coordination\n")
                f.write("# selected_ion_total_cn=all O-Na coordination pairs for ions touching adlayer water\n")
            else:
                f.write("# interfacial_pair_count=H(adlayer water)-F pairs; two H atoms are counted separately\n")
                f.write("# interfacial_water_count=unique adlayer water molecules with at least one H-F coordination\n")
                f.write("# selected_ion_total_cn=all H-F coordination pairs for ions touching adlayer water\n")
            f.write("# quantity value\n")
            f.write(f"avg_interfacial_pair_count_per_frame {avg_iw_pair_count:.10f}\n")
            f.write(f"avg_interfacial_water_count_per_frame_Gamma_total_IF {avg_iw_water_count:.10f}\n")
            f.write(f"avg_selected_ion_count_per_frame {avg_selected_ion_count:.10f}\n")
            f.write(f"avg_selected_ion_total_cn_per_frame {avg_selected_ion_total_cn:.10f}\n")
            f.write(f"mean_total_cn_per_selected_ion_CN_total_IF {mean_total_cn_per_selected_ion:.10f}\n")
            f.write(f"total_interfacial_pair_count {total['sum_iw_pair_count']}\n")
            f.write(f"total_interfacial_water_count {total['sum_iw_water_count']}\n")
            f.write(f"total_selected_ion_count {total['sum_selected_ion_count']}\n")
            f.write(f"total_selected_ion_total_cn {total['sum_selected_ion_total_cn']}\n")
        print(f"[DONE] wrote {out_path}")

    if "rdf" in modes:
        write_rdf_from_combined_results(
            args.header,
            cfg,
            results,
            args.rdf_z_low,
            args.rdf_z_high,
            args.skip_frames,
            label="rdf",
        )

    if "watermassdensity" in modes:
        rho_total = np.zeros(cfg["water_nbins"], dtype=float)
        frames = oxygen_atoms_sum = dumps = 0
        for result in results:
            wd = result["watermassdensity"]
            rho_total += wd["rho_sum"]
            frames += wd["frames"]
            oxygen_atoms_sum += wd["oxygen_atoms"]
            dumps += wd["dumps"]
        if frames == 0:
            sys.exit("No frames processed for watermassdensity.")
        print(
            f"[TOTAL] watermassdensity: dumps={dumps}, frames={frames}, "
            f"avg_oxygen_atoms_per_dump={safe_fraction(oxygen_atoms_sum, dumps):.2f}, "
            f"bins={cfg['water_nbins']}",
            file=sys.stderr,
        )
        out_path = f"{args.header}_watermassdensity.txt"
        with open(out_path, "w") as f:
            f.write("# z-resolved water mass density\n")
            f.write(f"# dz_A={BIN_DZ_A:.8f}\n")
            f.write(f"# z_origin_abs={Z_ORIGIN:.8f}\n")
            f.write(f"# z_rel_range_A={cfg['water_z_min']:.8f} {cfg['water_z_max']:.8f}\n")
            f.write(f"# frames={frames}\n")
            f.write("# z_rel_center_A rho_g_mL\n")
            for zc, rho in zip(cfg["water_z_centers_rel"], rho_total / float(frames)):
                f.write(f"{zc:.6f} {rho:.8f}\n")
        print(f"[DONE] wrote {out_path}")


def parse_mode_list(raw_modes):
    valid = ("angle1", "angle2", "iondist", "coordination", "rdf", "watermassdensity")
    out = []
    for item in raw_modes:
        for mode in item.split(","):
            mode = mode.strip().lower()
            if not mode:
                continue
            if mode == "all":
                for valid_mode in valid:
                    if valid_mode not in out:
                        out.append(valid_mode)
                continue
            if mode not in valid:
                raise ValueError(
                    "Invalid mode '{}'. Choose from: {}, or all.".format(
                        mode, ", ".join(valid)
                    )
                )
            if mode not in out:
                out.append(mode)
    if not out:
        raise ValueError("At least one mode is required.")
    return out


def validate_common_mode_args(args, modes):
    needs_adlayer = any(mode in modes for mode in ("angle1", "angle2", "coordination"))
    if needs_adlayer and args.z_adlayer is None:
        sys.exit("--z-adlayer is required when angle1, angle2, or coordination is selected.")

    if "iondist" in modes or "coordination" in modes or "rdf" in modes:
        missing = []
        if args.iontype is None:
            missing.append("--iontype")
        if "iondist" in modes:
            if args.z_ihl is None:
                missing.append("--z-ihl")
            if args.z_ohl is None:
                missing.append("--z-ohl")
        if "coordination" in modes and args.coord_cutoff is None:
            missing.append("--coord-cutoff")
        if "rdf" in modes:
            if args.rdf_z_low is None:
                missing.append("--rdf-z-low")
            if args.rdf_z_high is None:
                missing.append("--rdf-z-high")
            if args.rdf_rmax is None:
                missing.append("--rdf-rmax")
        if missing:
            sys.exit("{} required for selected mode(s).".format(", ".join(missing)))


def run_selected(args):
    try:
        modes = parse_mode_list(args.modes)
    except ValueError as exc:
        sys.exit(str(exc))

    validate_common_mode_args(args, modes)
    run_combined(args, modes)


def make_case_args(batch_args, case_index: int, tokens):
    if len(tokens) < 3:
        sys.exit("--case requires at least HEADER DATA TRJ.")
    header, data, *trajectories = tokens
    if not trajectories:
        sys.exit(f"--case {header}: at least one trajectory is required.")
    return argparse.Namespace(
        header=header,
        data=data,
        trajectories=trajectories,
        modes=batch_args.modes,
        z_adlayer=batch_args.z_adlayer,
        iontype=batch_args.iontype,
        z_ihl=batch_args.z_ihl,
        z_ohl=batch_args.z_ohl,
        coord_cutoff=batch_args.coord_cutoff,
        rdf_z_low=batch_args.rdf_z_low,
        rdf_z_high=batch_args.rdf_z_high,
        rdf_rmax=batch_args.rdf_rmax,
        rdf_dr=batch_args.rdf_dr,
        water_z_min=batch_args.water_z_min,
        water_z_max=batch_args.water_z_max,
        nprocs=batch_args.nprocs,
        skip_frames=batch_args.skip_frames,
        case_index=case_index,
    )


def run_batch(args):
    try:
        modes = parse_mode_list(args.modes)
    except ValueError as exc:
        sys.exit(str(exc))

    validate_common_mode_args(args, modes)
    if args.rdf_global_header and modes != ["rdf"]:
        sys.exit("--rdf-global-header can only be used when --modes rdf is selected alone.")

    case_args = [make_case_args(args, idx, tokens) for idx, tokens in enumerate(args.cases)]
    headers = [case.header for case in case_args]
    duplicate_headers = sorted({header for header in headers if headers.count(header) > 1})
    if duplicate_headers:
        sys.exit("Duplicate batch header(s) would overwrite outputs: {}".format(", ".join(duplicate_headers)))

    print("=== Batch DMI_edlanal modes ===")
    print("Modes:", ", ".join(modes))
    print(
        f"[SETUP] batch: cases={len(case_args)}, total_trajectories="
        f"{sum(len(case.trajectories) for case in case_args)}, "
        f"requested_nprocs={args.nprocs}, skip_frames={args.skip_frames}",
        file=sys.stderr,
    )

    case_cfgs = []
    per_case_tasks = []
    for case in case_args:
        cfg = prepare_combined_cfg(case, modes)
        case_cfgs.append(cfg)
        case_tasks = []
        for dump in case.trajectories:
            case_tasks.append((case.data, dump, cfg))
        per_case_tasks.append(case_tasks)
        print(
            f"[CASE] index={case.case_index}, header={case.header}, "
            f"data={case.data}, trajectories={len(case.trajectories)}",
            file=sys.stderr,
        )

    tasks = []
    for case_tasks in per_case_tasks:
        tasks.extend(case_tasks)
    print("[PARALLEL] batch: task_order=case-major", file=sys.stderr)

    results = run_tasks(analyze_combined_dump, tasks, args.nprocs, label="batch-combined-run")

    if args.rdf_global_header:
        print(f"\n[WRITE] batch global RDF header={args.rdf_global_header}")
        write_rdf_from_combined_results(
            args.rdf_global_header,
            case_cfgs[0],
            results,
            args.rdf_z_low,
            args.rdf_z_high,
            args.skip_frames,
            label="batch-global-rdf",
        )
        return

    grouped = {idx: [] for idx in range(len(case_args))}
    for result in results:
        grouped[result["case_index"]].append(result)

    for case, cfg in zip(case_args, case_cfgs):
        case_results = grouped[case.case_index]
        if len(case_results) != len(case.trajectories):
            sys.exit(
                f"Internal batch result mismatch for {case.header}: "
                f"expected {len(case.trajectories)}, got {len(case_results)}."
            )
        case._precomputed_cfg = cfg
        case._precomputed_results = case_results
        print(f"\n[WRITE] batch case header={case.header}")
        run_combined(case, modes)


def run_tasks(func, tasks, nprocs: int, label: str = "analysis"):
    requested = int(nprocs)
    usable = min(requested, len(tasks), os.cpu_count() or 1)
    print(
        f"[PARALLEL] {label}: files={len(tasks)}, requested_nprocs={requested}, "
        f"using_processes={usable}, chunk_size={usable}",
        file=sys.stderr,
    )
    if len(tasks) == 1 or usable == 1:
        try:
            return [func(task) for task in tasks]
        except RuntimeError as exc:
            sys.exit(str(exc))

    ctx = get_context("spawn")
    results = []
    try:
        for start in range(0, len(tasks), usable):
            stop = min(start + usable, len(tasks))
            chunk = tasks[start:stop]
            print(
                f"[PARALLEL] {label}: running task_chunk={start // usable + 1}, "
                f"tasks={start + 1}-{stop}/{len(tasks)}, processes={len(chunk)}",
                file=sys.stderr,
            )
            if len(chunk) == 1:
                results.append(func(chunk[0]))
            else:
                with ctx.Pool(processes=len(chunk)) as pool:
                    results.extend(pool.map(func, chunk))
        return results
    except RuntimeError as exc:
        sys.exit(str(exc))


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="DMI analysis modes for water orientation, ion layers, RDF, and water density."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    prun = sub.add_parser("run", help="run selected analysis modes in one command")
    prun.add_argument("header")
    prun.add_argument("data")
    prun.add_argument("trajectories", nargs="+")
    prun.add_argument(
        "--modes",
        nargs="+",
        required=True,
        help="Modes to run: angle1 angle2 iondist coordination rdf watermassdensity, comma-separated values, or all",
    )
    prun.add_argument("--z-adlayer", type=float, default=None)
    prun.add_argument("--iontype", default=None, help="Na or F, required for iondist/coordination/rdf")
    prun.add_argument("--z-ihl", type=float, default=None)
    prun.add_argument("--z-ohl", type=float, default=None)
    prun.add_argument("--coord-cutoff", type=float, default=None)
    prun.add_argument("--rdf-z-low", type=float, default=None)
    prun.add_argument("--rdf-z-high", type=float, default=None)
    prun.add_argument("--rdf-rmax", type=float, default=None)
    prun.add_argument("--rdf-dr", type=float, default=DEFAULT_RDF_DR)
    prun.add_argument("--water-z-min", type=float, default=DEFAULT_WATER_Z_MIN)
    prun.add_argument("--water-z-max", type=float, default=DEFAULT_WATER_Z_MAX)
    prun.add_argument("--nprocs", type=positive_int, default=1)
    prun.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    prun.set_defaults(func=run_selected)

    pbatch = sub.add_parser(
        "batch",
        help="run selected modes for multiple HEADER/DATA/TRJ groups using one case-major process queue",
    )
    pbatch.add_argument(
        "--case",
        dest="cases",
        action="append",
        nargs="+",
        required=True,
        metavar="ITEM",
        help="One case as: HEADER DATA TRJ1 [TRJ2 ...]. Repeat --case for more cases.",
    )
    pbatch.add_argument(
        "--modes",
        nargs="+",
        required=True,
        help="Modes to run: angle1 angle2 iondist coordination rdf watermassdensity, comma-separated values, or all",
    )
    pbatch.add_argument("--z-adlayer", type=float, default=None)
    pbatch.add_argument("--iontype", default=None, help="Na or F, required for iondist/coordination/rdf")
    pbatch.add_argument("--z-ihl", type=float, default=None)
    pbatch.add_argument("--z-ohl", type=float, default=None)
    pbatch.add_argument("--coord-cutoff", type=float, default=None)
    pbatch.add_argument("--rdf-z-low", type=float, default=None)
    pbatch.add_argument("--rdf-z-high", type=float, default=None)
    pbatch.add_argument("--rdf-rmax", type=float, default=None)
    pbatch.add_argument("--rdf-dr", type=float, default=DEFAULT_RDF_DR)
    pbatch.add_argument(
        "--rdf-global-header",
        default=None,
        help="With --modes rdf only, pool RDF over all batch cases and write one HEADER_rdf_* file.",
    )
    pbatch.add_argument("--water-z-min", type=float, default=DEFAULT_WATER_Z_MIN)
    pbatch.add_argument("--water-z-max", type=float, default=DEFAULT_WATER_Z_MAX)
    pbatch.add_argument("--nprocs", type=positive_int, default=1)
    pbatch.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    pbatch.set_defaults(func=run_batch)

    p1 = sub.add_parser("angle1", help="O-H bond orientation class fractions")
    p1.add_argument("z_adlayer", type=float)
    p1.add_argument("header")
    p1.add_argument("data")
    p1.add_argument("trajectories", nargs="+")
    p1.add_argument("--nprocs", type=positive_int, default=1)
    p1.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    p1.set_defaults(func=run_angle1)

    p2 = sub.add_parser("angle2", help="phi and chi angle histograms")
    p2.add_argument("z_adlayer", type=float)
    p2.add_argument("header")
    p2.add_argument("data")
    p2.add_argument("trajectories", nargs="+")
    p2.add_argument("--nprocs", type=positive_int, default=1)
    p2.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    p2.set_defaults(func=run_angle2)

    p3 = sub.add_parser("iondist", help="IHL/OHL/DifL ionic charge distribution")
    p3.add_argument("iontype", help="Na or F")
    p3.add_argument("z_ihl", type=float)
    p3.add_argument("z_ohl", type=float)
    p3.add_argument("header")
    p3.add_argument("data")
    p3.add_argument("trajectories", nargs="+")
    p3.add_argument("--nprocs", type=positive_int, default=1)
    p3.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    p3.set_defaults(func=run_iondist)

    pc = sub.add_parser("coordination", help="Figure S9/S10 interfacial-water / ion coordination statistics")
    pc.add_argument("iontype", help="Na or F")
    pc.add_argument("z_adlayer", type=float)
    pc.add_argument("coord_cutoff", type=float)
    pc.add_argument("header")
    pc.add_argument("data")
    pc.add_argument("trajectories", nargs="+")
    pc.add_argument("--nprocs", type=positive_int, default=1)
    pc.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    pc.set_defaults(func=run_coordination)

    prdf = sub.add_parser("rdf", help="bulk-region RDF, globally pooled over all input trajectories")
    prdf.add_argument("iontype", help="Na or F")
    prdf.add_argument("z_low", type=float)
    prdf.add_argument("z_high", type=float)
    prdf.add_argument("rmax", type=float)
    prdf.add_argument("header")
    prdf.add_argument("data")
    prdf.add_argument("trajectories", nargs="+")
    prdf.add_argument("--dr", type=float, default=DEFAULT_RDF_DR)
    prdf.add_argument("--nprocs", type=positive_int, default=1)
    prdf.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    prdf.set_defaults(func=run_rdf)

    p4 = sub.add_parser("watermassdensity", help="z-resolved water mass density")
    p4.add_argument("header")
    p4.add_argument("data")
    p4.add_argument("trajectories", nargs="+")
    p4.add_argument("--water-z-min", type=float, default=DEFAULT_WATER_Z_MIN)
    p4.add_argument("--water-z-max", type=float, default=DEFAULT_WATER_Z_MAX)
    p4.add_argument("--nprocs", type=positive_int, default=1)
    p4.add_argument("--skip-frames", type=nonnegative_int, default=DEFAULT_SKIP_FRAMES)
    p4.set_defaults(func=run_watermassdensity)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

