#!/usr/bin/env python3
"""
csv_to_ldr_and_dae.py

Inputs:
  1) minifigs_out.csv (wide format):
     name,hat_part,hat_color,hat2_part,hat2_color,... (your exact columns)

  2) MLCad.ini (Minifig Wizard definition file):
     used to look up (matrix 3x3 + offset xyz) for each part in each section.

Outputs:
  - out_ldr/<name>.ldr    (one per row)
  - out_dae/<name>.dae    (COLLADA export via LeoCAD, headless)

Notes:
  - Colors in your CSV are already LDraw color IDs (e.g., 86, 19, 2, ...).
  - LeoCAD CLI export: leocad -l <library_path> <file.ldr> -dae <file.dae>
    See LeoCAD CLI docs. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, List


# ----------------------------
# LeoCAD / MLCad.ini parsing
# ----------------------------

ENTRY_RE = re.compile(r'^"([^"]+)"\s+"([^"]*)"\s+(.*)$')

def parse_mlcad_ini(path: Path) -> Tuple[
    Dict[str, Dict[str, Tuple[List[str], List[str]]]],  # by_section[SECTION][part_lower] = (mat9, off3)
    Dict[str, Tuple[List[str], List[str], str]]         # by_any[part_lower] = (mat9, off3, section)
]:
    """
    Parses MLCad.ini and returns:
      - by_section: nested dict keyed by section then part filename
      - by_any: first-seen mapping for fallback if section lookup fails

    Expected line format per entry (after "Display" "File"):
      <Flags> <m11..m33> <offx offy offz>
    """
    by_section: Dict[str, Dict[str, Tuple[List[str], List[str]]]] = {}
    by_any: Dict[str, Tuple[List[str], List[str], str]] = {}

    cur_section: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                cur_section = line[1:-1].strip().upper()
                by_section.setdefault(cur_section, {})
                continue

            m = ENTRY_RE.match(line)
            if not m or cur_section is None:
                continue

            _display, filename, rest = m.groups()
            filename = filename.strip()
            if not filename:
                continue  # "None"/hidden entry

            tokens = rest.split()
            if len(tokens) < 1 + 9 + 3:
                continue

            # tokens: flags + 9 matrix + 3 offset
            mat = tokens[1:10]
            off = tokens[10:13]

            key = filename.lower()
            by_section[cur_section][key] = (mat, off)
            if key not in by_any:
                by_any[key] = (mat, off, cur_section)

    return by_section, by_any


# ----------------------------
# Slot mapping (your sections)
# ----------------------------

SLOT_TO_SECTION = {
    "hat":   "HATS",
    "hat2":  "HATS2",
    "head":  "HEAD",
    "body":  "BODY",
    "body2": "BODY2",
    "body3": "BODY3",
    "neck":  "NECK",
    "larm":  "LARM",
    "rarm":  "RARM",
    "lhand": "LHAND",
    "rhand": "RHAND",
    "lhanda":"LHANDA",
    "rhanda":"RHANDA",
    "lleg":  "LLEG",
    "rleg":  "RLEG",
    "llega": "LLEGA",
    "rlega": "RLEGA",
}

# The script discovers which slots exist by scanning CSV headers for "<slot>_part".


# ----------------------------
# LDraw writing
# ----------------------------

def write_ldr(out_path: Path, name: str, pieces: List[Tuple[int, str, List[str], List[str]]]) -> None:
    """
    pieces: list of (color_id, part_filename, mat9_strs, off3_strs)
    """
    lines = []
    lines.append(f"0 Name: {name}")
    lines.append("0 Author: csv_to_ldr_and_dae")
    lines.append("0 !LEOCAD MODEL")
    for color, part, mat, off in pieces:
        x, y, z = off
        a11,a12,a13,a21,a22,a23,a31,a32,a33 = mat
        # LDraw "type 1" line:
        lines.append(f"1 {color} {x} {y} {z} {a11} {a12} {a13} {a21} {a22} {a23} {a31} {a32} {a33} {part}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_part_filename(part: str, force_lower: bool) -> str:
    part = part.strip().strip('"')
    return part.lower() if force_lower else part


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlcad-ini", required=True, help="Path to MLCad.ini")
    ap.add_argument("--csv", required=True, help="Path to minifigs_out.csv (wide format)")
    ap.add_argument("--out-ldr", required=True, help="Output folder for generated .ldr files")
    ap.add_argument("--out-dae", required=True, help="Output folder for exported .dae files")

    ap.add_argument("--leocad", default="leocad", help="Path to leocad executable (or 'leocad' if in PATH)")
    ap.add_argument("--ldraw-lib", default="", help="LDraw library path (same as LeoCAD -l / LEOCAD_LIB)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--skip-dae", action="store_true", help="Only generate .ldr, do not run LeoCAD export")

    ap.add_argument(
        "--force-part-lower",
        action="store_true",
        help="Write part filenames lowercase in LDR (useful on case-sensitive filesystems)."
    )

    args = ap.parse_args()

    ini_path = Path(args.mlcad_ini).resolve()
    csv_path = Path(args.csv).resolve()
    out_ldr = Path(args.out_ldr).resolve()
    out_dae = Path(args.out_dae).resolve()

    out_ldr.mkdir(parents=True, exist_ok=True)
    out_dae.mkdir(parents=True, exist_ok=True)

    by_section, by_any = parse_mlcad_ini(ini_path)

    # LeoCAD library path: -l <path> takes priority, otherwise LEOCAD_LIB env.
    lib = (args.ldraw_lib or os.environ.get("LEOCAD_LIB", "")).strip()
    if not args.skip_dae and not lib:
        raise SystemExit(
            "Missing LDraw library path. Provide --ldraw-lib or set LEOCAD_LIB.\n"
            "LeoCAD uses -l <path> / LEOCAD_LIB to resolve part .dat files. "
        )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header row.")

        # Discover available slots from headers:
        part_cols = [h for h in reader.fieldnames if h.endswith("_part")]
        slots = [h[:-5] for h in part_cols]  # strip "_part"
        missing_map = [s for s in slots if s not in SLOT_TO_SECTION]
        if missing_map:
            raise SystemExit(f"CSV contains slots not in SLOT_TO_SECTION mapping: {missing_map}")

        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue

            ldr_path = out_ldr / f"{name}.ldr"
            dae_path = out_dae / f"{name}.dae"

            if ldr_path.exists() and not args.overwrite:
                print(f"[SKIP] LDR exists: {ldr_path.name}")
            else:
                pieces: List[Tuple[int, str, List[str], List[str]]] = []

                for slot in slots:
                    part_raw = (row.get(f"{slot}_part") or "").strip()
                    color_raw = (row.get(f"{slot}_color") or "").strip()

                    # Your CSV guarantees a color column even if part is empty.
                    # If part is empty, we simply skip emitting geometry.
                    if not part_raw:
                        continue

                    part = normalize_part_filename(part_raw, force_lower=args.force_part_lower)

                    try:
                        color = int(color_raw) if color_raw else 16
                    except Exception:
                        color = 16

                    section = SLOT_TO_SECTION[slot]
                    key = part.lower()

                    entry = by_section.get(section, {}).get(key)
                    if entry is None:
                        # fallback: sometimes a part appears in a different section or case mismatch
                        fallback = by_any.get(key)
                        if fallback is None:
                            raise SystemExit(f"Part not found in MLCad.ini: {part_raw} (slot={slot}, name={name})")
                        mat, off, found_section = fallback
                        # still proceed, but log where it was found
                        print(f"[WARN] Using fallback section {found_section} for {part_raw} (expected {section})")
                    else:
                        mat, off = entry

                    pieces.append((color, part, mat, off))

                write_ldr(ldr_path, name, pieces)
                print(f"[OK] Wrote {ldr_path.name} ({len(pieces)} parts)")

            if args.skip_dae:
                continue

            if dae_path.exists() and not args.overwrite:
                print(f"[SKIP] DAE exists: {dae_path.name}")
                continue

            # LeoCAD CLI export:
            #   leocad -l <lib> <file.ldr> -dae <outfile.dae>
            # See CLI docs. :contentReference[oaicite:2]{index=2}
            cmd = [args.leocad, "-l", lib, str(ldr_path), "-dae", str(dae_path)]
            print("[RUN]", " ".join(cmd))
            subprocess.run(cmd, check=True)
            print(f"[OK] Exported {dae_path.name}")

    print(f"[DONE] LDR -> {out_ldr}")
    if not args.skip_dae:
        print(f"[DONE] DAE -> {out_dae}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# use $env:LEOCAD_LIB = "C:\Program Files\LDraw (64 Bit)\LeoCAD"

#python csv_to_ldr_and_dae.py `
#  --mlcad-ini "C:\path\to\MLCad.ini" `
#  --csv "C:\path\to\minifigs_out.csv" `
#  --out-ldr "C:\path\to\out_ldr" `
#  --out-dae ".dae" `
#  --leocad "C:\Program Files\LDraw (64 Bit)\LeoCAD\LeoCAD.exe" `
#  --ldraw-lib "C:\Program Files\LDraw (64 Bit)\LeoCAD\library.bin" `
#  --overwrite

# python csv_to_ldr_and_dae.py   --mlcad-ini ".\MLCad.ini"  --csv ".\out_minifigs\minifigs_out.csv" --out-ldr ".\ldr_minifigs" --out-dae ".\dae_minifigs"  --leocad "C:\Program Files\LDraw (64 Bit)\LeoCAD\LeoCAD.exe"   --ldraw-lib "C:\Program Files\LDraw (64 Bit)\LeoCAD\library.bin"  --force-part-lower --overwrite