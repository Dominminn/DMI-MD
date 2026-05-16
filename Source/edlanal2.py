#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTERFACIAL LAYER ANALYSIS

USAGE:
    python edlanal.py Z_ihp Z_ohp DATA.lammps dump1.lammpstrj ... dumpN.lammpstrj output_header

OUTPUT FILES (IHL WATER: defined as water adlayer in the main text):

1) <header>_Pz_IHL.txt
    - line 1: mean_Pz  std_Pz        # (e·Å, IHL water)
    - line 2+: bin_center  count      # (min–max, 100 bins)

2) <header>_theta1_IHL.txt
    - line 1: mean_theta1_deg  std_theta1_deg
    - line 2+: bin_center_deg  count   # 0–180°, 100 bins

3) <header>_theta2_IHL.txt
    - line 1: mean_theta2_deg  std_theta2_deg
    - line 2+: bin_center_deg  count   # 0–180°, 100 bins

4) <header>_PzIHL_total.txt
    - line 1: mean_PzIHL_total  std_PzIHL_total   # per-frame IHL total μ_z
    - line 2+: bin_center  count                  # (min–max, 100 bins)

* IHL, OHL here is not relevant to definitions in main text, which is determined based on ion distributions.
* Likewise, ion distribution printed in this code is irrelevant to Fig. 5d in main text.
"""


import sys
import numpy as np
import MDAnalysis as mda

# ----------------- USER PARAMS -----------------
O_TYPE   = 2          # water O type
H_TYPES  = {1}        # water H types
NA_TYPE  = 3          # Na+ type

# IHL / OHL window (Å)
# IHL: z < Z1
# OHL: Z1 <= z < Z2

Z_SLAB = 7.178
Z_REF = 200.0 # for MDAnalysis: in our system zmin = -200 in LAMMPS which is set to 0 in MDAnslysis

NBINS_ANG = 100       # for theta1, theta2 (0–180 degrees)
NBINS_PZ  = 100       # for Pz histograms (min–max)
COORD_CUTOFF = 3.2    # Å

# OH bond length sanity check (after minimum image)
OH_MAX_CHECK = 1.0    # Å; if >1.0, warning 
# ------------------------------------------------

zhat = np.array([0.0, 0.0, 1.0], dtype=float)


# ----------------- Helpers -----------------
def minimage_vec_xy(d, box_lengths):
    Lx, Ly, Lz = box_lengths
    out = d.copy()
    out[..., 0] -= Lx * np.rint(out[..., 0] / Lx)
    out[..., 1] -= Ly * np.rint(out[..., 1] / Ly)
    return out


def build_water_triplets(u, o_type=O_TYPE, h_types=H_TYPES, max_OH=1.2):
    oxy = u.select_atoms(f"type {o_type}")
    waters = []

    has_bonds = any(len(O.bonds) > 0 for O in oxy)
    if has_bonds:
        def is_H(atom):
            try:
                return int(atom.type) in h_types
            except Exception:
                return (getattr(atom, "element", "") or "").upper() == "H"

        for O in oxy:
            neigh = set()
            for b in O.bonds:
                neigh.update(a for a in b.atoms if a.index != O.index)
            hs = [a for a in neigh if is_H(a)]
            if len(hs) == 2:
                waters.append((O.index, hs[0].index, hs[1].index))
    else:
        H_all = u.select_atoms(" or ".join([f"type {t}" for t in h_types]))
        H_pos = H_all.positions
        for O in oxy:
            d = H_pos - O.position
            dist = np.linalg.norm(d, axis=1)
            idx = np.where(dist <= max_OH)[0]
            if idx.size >= 2:
                ids = idx[np.argsort(dist[idx])[:2]]
                waters.append((O.index, H_all[ids[0]].index, H_all[ids[1]].index))

    waters = np.array(waters, dtype=int)
    if waters.size == 0:
        raise RuntimeError("No water triplets found (check types/bonds).")
    return waters


def safe_fraction(num, den):
    return (num / den) if den > 0 else 0.0


# ----------------- Main analysis -----------------
def main():
    if len(sys.argv) < 6:
        sys.exit(
            f"Usage: {sys.argv[0]} ihp ohp DATA.lammps dump1.lammpstrj ... dumpN.lammpstrj output_header"
        )

    ihp = float(sys.argv[1])
    ohp = float(sys.argv[2])

    Z1 = Z_REF + Z_SLAB + ihp
    Z2 = Z_REF + Z_SLAB + ohp

    DATA   = sys.argv[3]
    header = sys.argv[-1]
    DUMPS  = sys.argv[4:-1]

    # Global accumulators (IHL water)
    all_Pz_IHL = []
    all_PzIHL_total = []
    all_theta1_IHL = []
    all_theta2_IHL = []

    cnt_total_IHL_orient = 0
    cnt_2H_up            = 0
    cnt_1H_up            = 0
    cnt_flat             = 0
    cnt_1H_down          = 0
    cnt_2H_down          = 0
    cnt_1Hup1Hdown       = 0

    # Ion fraction accumulators
    Na_total_count = 0
    Na_IHL_count   = 0
    Na_OHL_count   = 0

    for dump_path in DUMPS:
        print(f"[INFO] Processing dump: {dump_path}", file=sys.stderr)
        u = mda.Universe(DATA, dump_path, format="LAMMPSDUMP")

        waters   = build_water_triplets(u, O_TYPE, H_TYPES)
        na_atoms = u.select_atoms(f"type {NA_TYPE}")
        atoms    = u.atoms  

        first = True  #skip first frame

        for ts in u.trajectory:
            if first:
                first = False
                continue

            pos = ts.positions
            Lx, Ly, Lz = ts.dimensions[:3]
            box = (Lx, Ly, Lz)

            if na_atoms.n_atoms > 0:
                pos_Na = na_atoms.positions
                zNa = pos_Na[:, 2]
                Na_total_count += na_atoms.n_atoms
                Na_IHL_count   += int(np.count_nonzero(zNa < Z1))
                Na_OHL_count   += int(np.count_nonzero((zNa >= Z1) & (zNa < Z2)))

            O  = pos[waters[:, 0]]
            H1 = pos[waters[:, 1]]
            H2 = pos[waters[:, 2]]

            v1 = minimage_vec_xy(H1 - O, box)   # r_H - r_O
            v2 = minimage_vec_xy(H2 - O, box)

            oh1 = np.linalg.norm(v1, axis=1)
            oh2 = np.linalg.norm(v2, axis=1)
            bad_mask = (oh1 > OH_MAX_CHECK) | (oh2 > OH_MAX_CHECK)
            if np.any(bad_mask):
                max_oh = max(oh1.max(), oh2.max())
                n_bad = int(bad_mask.sum())
                print("WRONGWATERBONDLENGTH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                sys.stderr.write(
                    f"[WARN] {dump_path}, frame {ts.frame}: "
                    f"{n_bad} water triplets with OH > {OH_MAX_CHECK:.2f} Å "
                    f"(max = {max_oh:.3f} Å)\n"
                )

            good_mask = ~bad_mask
            if not np.any(good_mask):
                continue

            O  = O[good_mask]
            H1 = H1[good_mask]
            H2 = H2[good_mask]
            v1 = v1[good_mask]
            v2 = v2[good_mask]

            zO = O[:, 2]
            ihl_mask = (zO < Z1)
            n_ihl = int(np.count_nonzero(ihl_mask))
            if n_ihl == 0:
                continue

            O_ihl  = O[ihl_mask]
            H1_ihl = H1[ihl_mask]
            H2_ihl = H2[ihl_mask]
            v1_ihl = v1[ihl_mask]
            v2_ihl = v2[ihl_mask]

            # ---------- Pz (per-water dipole z-component, e·Å, IHL) ----------
            idx_H1_all = waters[:, 1][good_mask][ihl_mask]
            idx_H2_all = waters[:, 2][good_mask][ihl_mask]

            qH1 = np.array([atoms[i].charge for i in idx_H1_all], dtype=float)
            qH2 = np.array([atoms[i].charge for i in idx_H2_all], dtype=float)

            mu_z_ihl = qH1 * v1_ihl[:, 2] + qH2 * v2_ihl[:, 2]  # shape (n_ihl,)
            all_Pz_IHL.append(mu_z_ihl)

            # ---------- per-frame IHL total Pz ----------
            PzIHL_total_frame = float(mu_z_ihl.sum())
            all_PzIHL_total.append(PzIHL_total_frame)

            # ---------- theta1 (bisector vs z, IHL) ----------
            n1 = v1_ihl / np.linalg.norm(v1_ihl, axis=1, keepdims=True)
            n2 = v2_ihl / np.linalg.norm(v2_ihl, axis=1, keepdims=True)

            b = n1 + n2
            normb = np.linalg.norm(b, axis=1)
            good_b = normb > 1e-8
            if np.any(good_b):
                b_unit = b[good_b] / normb[good_b][:, None]
                cos_th1 = np.clip(b_unit @ zhat, -1.0, 1.0)
                th1 = np.degrees(np.arccos(cos_th1))
                all_theta1_IHL.append(th1)

            # ---------- theta2 (molecular plane normal vs z, IHL) ----------
            b_plane = np.cross(n1, n2)
            normp = np.linalg.norm(b_plane, axis=1)
            good_p = normp > 1e-8
            if np.any(good_p):
                b_plane_unit = b_plane[good_p] / normp[good_p][:, None]
                cos_th2 = np.clip(b_plane_unit @ zhat, -1.0, 1.0)
                th2 = np.degrees(np.arccos(cos_th2))
                all_theta2_IHL.append(th2)

            cos1_all = np.clip(
                (v1_ihl / np.linalg.norm(v1_ihl, axis=1, keepdims=True)) @ zhat,
                -1.0, 1.0
            )
            cos2_all = np.clip(
                (v2_ihl / np.linalg.norm(v2_ihl, axis=1, keepdims=True)) @ zhat,
                -1.0, 1.0
            )
            th1_all = np.degrees(np.arccos(cos1_all))
            th2_all = np.degrees(np.arccos(cos2_all))

            # bins: Up[0–60), Mid[60–120), Down[120–180]
            up1   = (th1_all >= 0.0)   & (th1_all < 60.0)
            mid1  = (th1_all >= 60.0)  & (th1_all < 120.0)
            down1 = (th1_all >= 120.0) & (th1_all <= 180.0)

            up2   = (th2_all >= 0.0)   & (th2_all < 60.0)
            mid2  = (th2_all >= 60.0)  & (th2_all < 120.0)
            down2 = (th2_all >= 120.0) & (th2_all <= 180.0)

            twoH_up      = up1 & up2
            twoH_down    = down1 & down2
            flat         = mid1 & mid2
            oneH_down    = (down1 & mid2) | (down2 & mid1)
            oneH_up      = (up1 & mid2)   | (up2 & mid1)
            oneHup_down  = (up1 & down2)  | (up2 & down1)

            n_ihl_frame = th1_all.size
            cnt_total_IHL_orient += n_ihl_frame
            cnt_2H_up      += int(twoH_up.sum())
            cnt_2H_down    += int(twoH_down.sum())
            cnt_flat       += int(flat.sum())
            cnt_1H_down    += int(oneH_down.sum())
            cnt_1H_up      += int(oneH_up.sum())
            cnt_1Hup1Hdown += int(oneHup_down.sum())

    # ----------------- Concatenate and finalize -----------------
    if not all_Pz_IHL:
        sys.exit("No IHL waters found in any frame (after skipping first frames).")

    Pz_IHL_arr      = np.concatenate(all_Pz_IHL)          # per-water (IHL)
    theta1_IHL_arr  = np.concatenate(all_theta1_IHL) if all_theta1_IHL else np.array([])
    theta2_IHL_arr  = np.concatenate(all_theta2_IHL) if all_theta2_IHL else np.array([])
    PzIHL_total_arr = np.array(all_PzIHL_total, dtype=float)   # per-frame IHL total

    # ===== 1) Pz_IHL file =====
    Pz_mean = float(Pz_IHL_arr.mean())
    Pz_std  = float(Pz_IHL_arr.std(ddof=0))

    Pz_min, Pz_max = Pz_IHL_arr.min(), Pz_IHL_arr.max()
    if Pz_min == Pz_max:
        Pz_min -= 1e-6
        Pz_max += 1e-6
    Pz_bins = np.linspace(Pz_min, Pz_max, NBINS_PZ + 1)
    Pz_hist, _ = np.histogram(Pz_IHL_arr, bins=Pz_bins)
    Pz_centers = 0.5 * (Pz_bins[:-1] + Pz_bins[1:])

    with open(f"{header}_Pz_IHL.txt", "w") as f:
        f.write(f"{Pz_mean:.8f}\t{Pz_std:.8f}\n")
        for c, cnt in zip(Pz_centers, Pz_hist):
            f.write(f"{c:.8f}\t{int(cnt)}\n")

    # ===== 2) theta1_IHL file =====
    if theta1_IHL_arr.size > 0:
        th1_mean = float(theta1_IHL_arr.mean())
        th1_std  = float(theta1_IHL_arr.std(ddof=0))
    else:
        th1_mean = th1_std = 0.0

    th_bins = np.linspace(0.0, 180.0, NBINS_ANG + 1)
    th1_hist, _ = np.histogram(theta1_IHL_arr, bins=th_bins)
    th_centers = 0.5 * (th_bins[:-1] + th_bins[1:])

    with open(f"{header}_theta1_IHL.txt", "w") as f:
        f.write(f"{th1_mean:.8f}\t{th1_std:.8f}\n")
        for c, cnt in zip(th_centers, th1_hist):
            f.write(f"{c:.8f}\t{int(cnt)}\n")

    # ===== 3) theta2_IHL file =====
    if theta2_IHL_arr.size > 0:
        th2_mean = float(theta2_IHL_arr.mean())
        th2_std  = float(theta2_IHL_arr.std(ddof=0))
    else:
        th2_mean = th2_std = 0.0

    th2_hist, _ = np.histogram(theta2_IHL_arr, bins=th_bins)

    with open(f"{header}_theta2_IHL.txt", "w") as f:
        f.write(f"{th2_mean:.8f}\t{th2_std:.8f}\n")
        for c, cnt in zip(th_centers, th2_hist):
            f.write(f"{c:.8f}\t{int(cnt)}\n")

    # ===== 4) PzIHL_total (per-frame IHL total μ_z) =====
    if PzIHL_total_arr.size > 0:
        PzIHL_mean = float(PzIHL_total_arr.mean())
        PzIHL_std  = float(PzIHL_total_arr.std(ddof=0))
    else:
        PzIHL_mean = PzIHL_std = 0.0

    PzIHL_min, PzIHL_max = PzIHL_total_arr.min(), PzIHL_total_arr.max()
    if PzIHL_min == PzIHL_max:
        PzIHL_min -= 1e-6
        PzIHL_max += 1e-6
    PzIHL_bins = np.linspace(PzIHL_min, PzIHL_max, NBINS_PZ + 1)
    PzIHL_hist, _ = np.histogram(PzIHL_total_arr, bins=PzIHL_bins)
    PzIHL_centers = 0.5 * (PzIHL_bins[:-1] + PzIHL_bins[1:])

    with open(f"{header}_PzIHL_total.txt", "w") as f:
        f.write(f"{PzIHL_mean:.8f}\t{PzIHL_std:.8f}\n")
        for c, cnt in zip(PzIHL_centers, PzIHL_hist):
            f.write(f"{c:.8f}\t{int(cnt)}\n")

    print("\n=== IHL water orientation categories (global) ===")
    print(f"Total IHL waters counted = {cnt_total_IHL_orient}")

    frac_2H_up      = safe_fraction(cnt_2H_up,      cnt_total_IHL_orient)
    frac_1H_up      = safe_fraction(cnt_1H_up,      cnt_total_IHL_orient)
    frac_flat       = safe_fraction(cnt_flat,       cnt_total_IHL_orient)
    frac_1H_down    = safe_fraction(cnt_1H_down,    cnt_total_IHL_orient)
    frac_2H_down    = safe_fraction(cnt_2H_down,    cnt_total_IHL_orient)
    frac_1Hup1Hdown = safe_fraction(cnt_1Hup1Hdown, cnt_total_IHL_orient)

    print(f"2H up         : count = {cnt_2H_up:8d}, fraction = {frac_2H_up:.8f}")
    print(f"1H up         : count = {cnt_1H_up:8d}, fraction = {frac_1H_up:.8f}")
    print(f"flat          : count = {cnt_flat:8d}, fraction = {frac_flat:.8f}")
    print(f"1H down       : count = {cnt_1H_down:8d}, fraction = {frac_1H_down:.8f}")
    print(f"2H down       : count = {cnt_2H_down:8d}, fraction = {frac_2H_down:.8f}")
    print(f"1H up / 1H down: count = {cnt_1Hup1Hdown:8d}, fraction = {frac_1Hup1Hdown:.8f}")

    # ===== ion fraction (IHL vs OHL, print) =====
    print("\n=== Ion (Na) region fractions ===")
    print(f"Total Na counts over frames = {Na_total_count}")
    print(f"Na in IHL (z < Z1)          = {Na_IHL_count}")
    print(f"Na in OHL (Z1 <= z < Z2)    = {Na_OHL_count}")

    frac_Na_IHL = safe_fraction(Na_IHL_count, Na_total_count)
    frac_Na_OHL = safe_fraction(Na_OHL_count, Na_total_count)

    print(f"IHL Na fraction = {frac_Na_IHL:.8f}")
    print(f"OHL Na fraction = {frac_Na_OHL:.8f}")

    print("\n[DONE] Outputs written with header:", header, file=sys.stderr)


if __name__ == "__main__":
    main()

