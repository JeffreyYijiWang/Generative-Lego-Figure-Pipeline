#!/usr/bin/env python3
"""
make_minifigs_from_csv_with_colors.py

Generates randomized minifig LDraw files from minifigs.csv AND writes a "wide" CSV:

    name,hat_part,hat_color,head_part,head_color,body_part,body_color,legs_part,legs_color,...

…but extended to include ALL of your sections/slots:
  hat, hat2, head, body, body2, body3, neck,
  larm, rarm, lhand, rhand, lhanda, rhanda,
  lleg, rleg, llega, rlega

Key behavior you requested:
- Every slot ALWAYS gets a color in the wide CSV, even if that slot is skipped / None / missing.
- Parts can be skipped by probability (attachments) or missing in the source CSV -> part field becomes "".
- Colors are randomly assigned per slot (no LEGO knowledge required).
- Symmetry defaults: L/R arms share a color, L/R hands share, L/R legs share.
- Attachments default to inheriting their base slot color (hat2 inherits hat, lhanda inherits lhand, etc.).
  You can change that in the comments below.

Outputs:
- <outdir>/mpd/ (or <outdir>/ldr/) : generated model files
- <outdir>/manifest.csv : row-per-slot audit log (unchanged)
- <outdir>/minifigs_out.csv : one row per minifig, wide format with part+color per slot

Run example:
  python make_minifigs_from_csv_with_colors.py --csv minifigs.csv --outdir out_minifigs --count 50 --seed 1 --format mpd_root
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class PartRow:
    section: str
    display_name: str
    part_file: str
    R: Tuple[float, float, float, float, float, float, float, float, float]  # row-major 3x3
    t: Tuple[float, float, float]  # (x,y,z) LDraw units


# ----------------------------
# Small linear algebra
# ----------------------------

def mat3_mul(A: Sequence[float], B: Sequence[float]) -> Tuple[float, ...]:
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = A
    b11, b12, b13, b21, b22, b23, b31, b32, b33 = B
    return (
        a11*b11 + a12*b21 + a13*b31,  a11*b12 + a12*b22 + a13*b32,  a11*b13 + a12*b23 + a13*b33,
        a21*b11 + a22*b21 + a23*b31,  a21*b12 + a22*b22 + a23*b32,  a21*b13 + a22*b23 + a23*b33,
        a31*b11 + a32*b21 + a33*b31,  a31*b12 + a32*b22 + a33*b32,  a31*b13 + a32*b23 + a33*b33,
    )

def mat3_vec3_mul(A: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = A
    x, y, z = v
    return (
        a11*x + a12*y + a13*z,
        a21*x + a22*y + a23*z,
        a31*x + a32*y + a33*z,
    )

def mat3_scale(A: Sequence[float], s: float) -> Tuple[float, ...]:
    return tuple(a * s for a in A)

def euler_deg_to_mat3(rx: float, ry: float, rz: float, order: str = "XYZ") -> Tuple[float, ...]:
    def Rx(a: float) -> Tuple[float, ...]:
        ca, sa = math.cos(a), math.sin(a)
        return (1,0,0,  0,ca,-sa,  0,sa,ca)

    def Ry(a: float) -> Tuple[float, ...]:
        ca, sa = math.cos(a), math.sin(a)
        return (ca,0,sa,  0,1,0,  -sa,0,ca)

    def Rz(a: float) -> Tuple[float, ...]:
        ca, sa = math.cos(a), math.sin(a)
        return (ca,-sa,0,  sa,ca,0,  0,0,1)

    mats = {
        "X": Rx(math.radians(rx)),
        "Y": Ry(math.radians(ry)),
        "Z": Rz(math.radians(rz)),
    }

    R = (1,0,0, 0,1,0, 0,0,1)
    for axis in order.upper():
        R = mat3_mul(R, mats[axis])
    return R


# ----------------------------
# CSV parsing (robust headers)
# ----------------------------

def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch in ("_",))

def read_minifigs_csv(path: str) -> List[PartRow]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        fields = {_norm(h): h for h in reader.fieldnames}

        def pick(*names: str) -> Optional[str]:
            for n in names:
                if n in fields:
                    return fields[n]
            return None

        col_section = pick("section", "sec")
        col_name    = pick("displayname", "display_name", "name", "title")
        col_file    = pick("partfile", "part_file", "file", "filename", "dat", "ldr")

        # Support both ox/oy/oz and offx/offy/offz:
        col_ox      = pick("offx", "ox", "x", "offsetx", "tx")
        col_oy      = pick("offy", "oy", "y", "offsety", "ty")
        col_oz      = pick("offz", "oz", "z", "offsetz", "tz")

        def mcol(r: int, c: int) -> Optional[str]:
            return pick(f"a{r}{c}", f"m{r}{c}", f"r{r}{c}")

        mcols = [mcol(1,1), mcol(1,2), mcol(1,3),
                 mcol(2,1), mcol(2,2), mcol(2,3),
                 mcol(3,1), mcol(3,2), mcol(3,3)]

        missing = []
        if not col_section: missing.append("section")
        if not col_name:    missing.append("display_name/name")
        if not col_file:    missing.append("part_file/file")
        if not (col_ox and col_oy and col_oz): missing.append("offset (offx/offy/offz or ox/oy/oz)")
        if any(c is None for c in mcols): missing.append("matrix (a11..a33 or m11..m33)")

        if missing:
            raise ValueError(
                "CSV missing required columns: " + ", ".join(missing) +
                "\nFound columns: " + ", ".join(reader.fieldnames)
            )

        rows: List[PartRow] = []
        for row in reader:
            section = (row[col_section] or "").strip()
            name    = (row[col_name] or "").strip()
            pfile   = (row[col_file] or "").strip().strip('"')

            if not section or not name:
                continue
            if name.startswith("---") or "-----" in name:
                continue

            try:
                R = tuple(float(row[c]) for c in mcols)  # type: ignore[arg-type]
                t = (float(row[col_ox]), float(row[col_oy]), float(row[col_oz]))
            except Exception:
                continue

            rows.append(PartRow(section=section, display_name=name, part_file=pfile, R=R, t=t))

        return rows


# ----------------------------
# Section indexing
# ----------------------------

def build_section_index(rows: Sequence[PartRow]) -> Dict[str, List[PartRow]]:
    out: Dict[str, List[PartRow]] = {}
    for r in rows:
        out.setdefault(r.section, []).append(r)
    return out

def is_selectable_part(r: PartRow, allow_none: bool) -> bool:
    if r.part_file.strip() == "":
        return allow_none
    return True


# ----------------------------
# LDraw emission
# ----------------------------

def fmt_float(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s

def ldraw_type1_line(color: int, R: Sequence[float], t: Sequence[float], part: str) -> str:
    a11,a12,a13,a21,a22,a23,a31,a32,a33 = R
    x,y,z = t
    return " ".join([
        "1", str(color),
        fmt_float(x), fmt_float(y), fmt_float(z),
        fmt_float(a11), fmt_float(a12), fmt_float(a13),
        fmt_float(a21), fmt_float(a22), fmt_float(a23),
        fmt_float(a31), fmt_float(a32), fmt_float(a33),
        part,
    ])

def apply_root_pose_to_part(
    root_R: Sequence[float],
    root_t: Sequence[float],
    part_R: Sequence[float],
    part_t: Sequence[float],
    pivot: Sequence[float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, ...], Tuple[float, float, float]]:
    R_out = mat3_mul(root_R, part_R)

    tp = (part_t[0] - pivot[0], part_t[1] - pivot[1], part_t[2] - pivot[2])
    rtp = mat3_vec3_mul(root_R, tp)
    t_out = (rtp[0] + pivot[0] + root_t[0],
             rtp[1] + pivot[1] + root_t[1],
             rtp[2] + pivot[2] + root_t[2])
    return R_out, t_out

def write_mpd_with_root(
    out_path: str,
    fig_subfile_name: str,
    root_line: str,
    part_lines: Sequence[str],
    comment_lines: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    main_name = "main.ldr"

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for c in comment_lines:
            f.write(f"0 {c}\n")

        f.write(f"0 FILE {fig_subfile_name}\n")
        for line in part_lines:
            f.write(line + "\n")
        f.write("0 NOFILE\n")

        f.write(f"0 FILE {main_name}\n")
        f.write(root_line + "\n")
        f.write("0 NOFILE\n")

def write_flat_ldr(
    out_path: str,
    lines: Sequence[str],
    comment_lines: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for c in comment_lines:
            f.write(f"0 {c}\n")
        for line in lines:
            f.write(line + "\n")


# ----------------------------
# Slot -> Section dictionary (your exact sections)
# ----------------------------

DEFAULT_SLOTS: Dict[str, List[str]] = {
    # Core figure
    "hat":   ["HATS"],
    "hat2":  ["HATS2"],
    "head":  ["HEAD"],

    # BODY, BODY2, BODY3 are separate sections in your data, and you said you want them all.
    # We'll sample them independently, so you might get 1-3 body layers depending on probabilities.
    "body":  ["BODY"],
    "body2": ["BODY2"],
    "body3": ["BODY3"],

    "neck":  ["NECK"],

    # Arms / hands
    "larm":  ["LARM"],
    "rarm":  ["RARM"],
    "lhand": ["LHAND"],
    "rhand": ["RHAND"],

    # Hand attachments
    "lhanda": ["LHANDA"],
    "rhanda": ["RHANDA"],

    # Legs
    "lleg":  ["LLEG"],
    "rleg":  ["RLEG"],

    # Leg attachments
    "llega": ["LLEGA"],
    "rlega": ["RLEGA"],
}

# Slot order for wide CSV columns:
SLOT_ORDER = [
    "hat","hat2","head",
    "body","body2","body3","neck",
    "larm","rarm","lhand","rhand","lhanda","rhanda",
    "lleg","rleg","llega","rlega",
]


# ----------------------------
# Random color assignment (no LEGO knowledge required)
# ----------------------------

# A conservative palette of distinct LDraw color codes.
# If any code renders oddly in your pipeline, remove it.
SAFE_LDRAW_COLORS = [1,2,4,7,14,15,19,25,26,28,71,72,80,85,86,88]

# Symmetry groups: share one color between left/right.
SYMMETRY_GROUPS = [
    ("larm","rarm"),
    ("lhand","rhand"),
    ("lleg","rleg"),
]

# Attachments inherit base slot color by default:
# If you want attachments to be independently colored, set INHERIT_ATTACHMENTS=False.
INHERIT_ATTACHMENTS = True
SLOT_INHERIT = {
    "hat2": "hat",
    "lhanda": "lhand",
    "rhanda": "rhand",
    "llega": "lleg",
    "rlega": "rleg",
}

def pick_distinct_color(rng: random.Random, used: set[int]) -> int:
    candidates = [c for c in SAFE_LDRAW_COLORS if c not in used]
    if not candidates:
        return rng.choice(SAFE_LDRAW_COLORS)
    return rng.choice(candidates)

def assign_slot_colors(
    rng: random.Random,
    slots_all: List[str],
    force_distinct_across_slots: bool = True,
) -> Dict[str, int]:
    """
    Returns slot->color mapping for *all* slots (even if a slot is skipped later).
    """
    colors: Dict[str, int] = {}
    used: set[int] = set()

    def choose_color() -> int:
        if not force_distinct_across_slots:
            return rng.choice(SAFE_LDRAW_COLORS)
        c = pick_distinct_color(rng, used)
        used.add(c)
        return c

    slots_set = set(slots_all)

    # 1) Symmetry groups first
    for a, b in SYMMETRY_GROUPS:
        if a in slots_set or b in slots_set:
            c = choose_color()
            colors[a] = c
            colors[b] = c

    # 2) Remaining slots
    for slot in slots_all:
        if slot in colors:
            continue
        # If attachment inherits, we postpone assignment until after base slots
        if INHERIT_ATTACHMENTS and slot in SLOT_INHERIT:
            continue
        colors[slot] = choose_color()

    # 3) Attachment inheritance
    if INHERIT_ATTACHMENTS:
        for att, base in SLOT_INHERIT.items():
            if att in slots_set:
                colors[att] = colors.get(base, choose_color())

    # 4) Final safety: fill anything missing
    for slot in slots_all:
        if slot not in colors:
            colors[slot] = choose_color()

    return colors


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to minifigs.csv")
    ap.add_argument("--outdir", required=True, help="Output directory (will create subfolder mpd/ or ldr/)")
    ap.add_argument("--count", type=int, default=10, help="How many minifigs to generate")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")

    ap.add_argument("--allow-none", action="store_true", help="Allow selecting 'None' entries (empty part_file)")

    # Root pose
    ap.add_argument("--rx", type=float, default=0.0)
    ap.add_argument("--ry", type=float, default=0.0)
    ap.add_argument("--rz", type=float, default=0.0)
    ap.add_argument("--order", type=str, default="XYZ")
    ap.add_argument("--root-scale", type=float, default=1.0)
    ap.add_argument("--root-tx", type=float, default=0.0)
    ap.add_argument("--root-ty", type=float, default=0.0)
    ap.add_argument("--root-tz", type=float, default=0.0)
    ap.add_argument("--pivot-x", type=float, default=0.0)
    ap.add_argument("--pivot-y", type=float, default=0.0)
    ap.add_argument("--pivot-z", type=float, default=0.0)

    # Output format
    ap.add_argument("--format", choices=["mpd_root", "ldr_flat"], default="mpd_root")

    # Optional attachment probabilities:
    # NOTE: You asked “even if a section has none it still has a color”.
    # These probabilities only decide whether a *part line* is emitted, NOT whether a color exists in the wide CSV.
    ap.add_argument("--p-hat2",   type=float, default=0.25)
    ap.add_argument("--p-lhanda", type=float, default=0.30)
    ap.add_argument("--p-rhanda", type=float, default=0.30)
    ap.add_argument("--p-llega",  type=float, default=0.20)
    ap.add_argument("--p-rlega",  type=float, default=0.20)
    ap.add_argument("--p-neck",   type=float, default=1.00)

    # Control body layers (BODY2/BODY3) as optional overlays if you want:
    ap.add_argument("--p-body2",  type=float, default=0.00, help="Probability to include BODY2 layer")
    ap.add_argument("--p-body3",  type=float, default=0.00, help="Probability to include BODY3 layer")

    # Wide CSV name
    ap.add_argument("--wide-csv", default="", help="Optional path for minifigs_out.csv (default: <outdir>/minifigs_out.csv)")

    args = ap.parse_args()

    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    optional_prob: Dict[str, float] = {
        "hat2":  clamp01(args.p_hat2),
        "lhanda": clamp01(args.p_lhanda),
        "rhanda": clamp01(args.p_rhanda),
        "llega": clamp01(args.p_llega),
        "rlega": clamp01(args.p_rlega),
        "neck":  clamp01(args.p_neck),
        "body2": clamp01(args.p_body2),
        "body3": clamp01(args.p_body3),
    }

    rows = read_minifigs_csv(args.csv)
    sec_index = build_section_index(rows)

    rng = random.Random(args.seed)

    # Root transform
    root_R = euler_deg_to_mat3(args.rx, args.ry, args.rz, order=args.order)
    pivot = (args.pivot_x, args.pivot_y, args.pivot_z)
    root_t = (args.root_tx, args.root_ty, args.root_tz)
    root_R_for_rootline = mat3_scale(root_R, args.root_scale) if args.format == "mpd_root" else root_R

    # Output folders
    os.makedirs(args.outdir, exist_ok=True)
    models_dir = os.path.join(args.outdir, "mpd" if args.format == "mpd_root" else "ldr")
    os.makedirs(models_dir, exist_ok=True)

    manifest_path = os.path.join(args.outdir, "manifest.csv")
    wide_csv_path = args.wide_csv.strip() or os.path.join(args.outdir, "minifigs_out.csv")

    # ------------------------------------------------------------
    # PRECOMPUTE candidates once per slot
    # ------------------------------------------------------------
    slot_selectable: Dict[str, List[PartRow]] = {}
    for slot, sections in DEFAULT_SLOTS.items():
        cands: List[PartRow] = []
        for sec in sections:
            cands.extend(sec_index.get(sec, []))

        selectable = [c for c in cands if is_selectable_part(c, args.allow_none)]
        slot_selectable[slot] = selectable
        # If a section does not exist in CSV, this list will be empty; we will still assign a color later.

    # Prepare wide CSV header: name + (<slot>_part, <slot>_color) for all slots
    wide_header: List[str] = ["name"]
    for slot in SLOT_ORDER:
        wide_header.append(f"{slot}_part")
        wide_header.append(f"{slot}_color")

    with open(manifest_path, "w", encoding="utf-8", newline="") as mf, \
         open(wide_csv_path, "w", encoding="utf-8", newline="") as wf:

        mw = csv.writer(mf)
        ww = csv.writer(wf)

        mw.writerow(["minifig_id", "outfile", "slot", "section", "display_name", "part_file", "color"])
        ww.writerow(wide_header)

        for i in range(args.count):
            out_base = f"minifig_{i:04d}"

            # 1) Assign colors for ALL slots (even if some parts are missing or skipped)
            slot_colors = assign_slot_colors(
                rng=rng,
                slots_all=SLOT_ORDER,
                force_distinct_across_slots=True,
            )

            # 2) Choose parts per slot (may be skipped by probability or missing)
            chosen: Dict[str, PartRow] = {}

            for slot in SLOT_ORDER:
                # probability gate for optional slots
                if slot in optional_prob:
                    if rng.random() >= optional_prob[slot]:
                        continue  # skip part for this slot (color still exists)
                # If this slot isn't present in the CSV index, cannot choose a part
                candidates = slot_selectable.get(slot, [])
                if not candidates:
                    continue
                chosen[slot] = rng.choice(candidates)

            # 3) Emit part lines with slot-specific colors
            local_part_lines: List[str] = [] 
            for slot, pr in chosen.items():
                if pr.part_file.strip() == "":
                    continue
                color = int(slot_colors[slot])
                local_part_lines.append(ldraw_type1_line(color, pr.R, pr.t, pr.part_file))

            # 4) Write model file
            out_path = os.path.join(models_dir, out_base + (".ldr" if args.format == "ldr_flat" else ".mpd"))
            comment_lines = [
                "Generated by make_minifigs_from_csv_with_colors.py",
                f"id={out_base}",
                f"seed={args.seed}",
                f"root_pose_deg rx={args.rx} ry={args.ry} rz={args.rz} order={args.order}",
                f"format={args.format}",
                "colors=slot-specific (see minifigs_out.csv)",
            ]

            if args.format == "mpd_root":
                fig_subfile = f"{out_base}_FIGURE.ldr"
                # Root line uses default color 16 (does not matter much) because child parts have explicit colors.
                root_line = ldraw_type1_line(16, root_R_for_rootline, root_t, fig_subfile)
                write_mpd_with_root(
                    out_path=out_path,
                    fig_subfile_name=fig_subfile,
                    root_line=root_line,
                    part_lines=local_part_lines,
                    comment_lines=comment_lines,
                )
            else:
                baked_lines: List[str] = []
                for slot, pr in chosen.items():
                    if pr.part_file.strip() == "":
                        continue
                    color = int(slot_colors[slot])
                    Rb, tb = apply_root_pose_to_part(root_R, root_t, pr.R, pr.t, pivot=pivot)
                    baked_lines.append(ldraw_type1_line(color, Rb, tb, pr.part_file))
                write_flat_ldr(out_path=out_path, lines=baked_lines, comment_lines=comment_lines)

            # 5) Write manifest rows (slot audit)
            outfile_name = os.path.basename(out_path)
            for slot in SLOT_ORDER:
                if slot in chosen:
                    pr = chosen[slot]
                    mw.writerow([out_base, outfile_name, slot, pr.section, pr.display_name, pr.part_file, slot_colors[slot]])
                else:
                    mw.writerow([out_base, outfile_name, slot, "", "(skipped/missing)", "", slot_colors[slot]])

            # 6) Write wide CSV row (one line per minifig, all slots always have a color)
            wide_row: List[str] = [out_base]
            for slot in SLOT_ORDER:
                part = chosen[slot].part_file if slot in chosen else ""
                wide_row.append(part)
                wide_row.append(str(int(slot_colors[slot])))
            ww.writerow(wide_row)

            print(f"[OK] {out_base}: wrote {out_path} | parts={len(local_part_lines)} | wide_row written")

    print(f"[DONE] Models -> {models_dir}")
    print(f"[DONE] manifest.csv -> {manifest_path}")
    print(f"[DONE] minifigs_out.csv -> {wide_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
