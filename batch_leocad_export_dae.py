#!/usr/bin/env python3
"""
batch_leocad_export_dae.py

Batch convert .ldr files -> .dae using LeoCAD CLI.

Requirements:
- LeoCAD installed and accessible as 'leocad' on PATH (or pass --leocad).
- LDraw parts library path provided via --ldraw-lib OR env var LEOCAD_LIB.

LeoCAD references:
- --export-collada / -dae option exists in LeoCAD CLI docs. :contentReference[oaicite:7]{index=7}
- --libpath / -l and LEOCAD_LIB environment variable. :contentReference[oaicite:8]{index=8}
"""

from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Folder containing .ldr files")
    ap.add_argument("--out", dest="outdir", required=True, help="Folder to write .dae files")
    ap.add_argument("--leocad", default="leocad", help="LeoCAD executable (default: leocad)")
    ap.add_argument("--ldraw-lib", default="", help="Path to LDraw parts library root (optional if LEOCAD_LIB set)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .dae")
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ldraw_lib = args.ldraw_lib.strip()
    if not ldraw_lib:
        ldraw_lib = os.environ.get("LEOCAD_LIB", "").strip()

    if not ldraw_lib:
        raise SystemExit(
            "Missing LDraw library path. Provide --ldraw-lib or set env var LEOCAD_LIB."
        )

    # LeoCAD expects a library root containing parts/p folders.
    ldraw_lib_path = Path(ldraw_lib).resolve()
    if not ldraw_lib_path.exists():
        raise SystemExit(f"LDraw library path does not exist: {ldraw_lib_path}")

    ldr_files = sorted(indir.glob("*.ldr"))
    if not ldr_files:
        print(f"No .ldr files found in: {indir}")
        return 0

    failures = 0
    for ldr in ldr_files:
        dae = outdir / (ldr.stem + ".dae")
        if dae.exists() and not args.overwrite:
            print(f"[SKIP] {dae.name} exists")
            continue

        # Command:
        #   leocad input.ldr --export-collada out.dae --libpath <ldraw_lib>
        cmd = [
            args.leocad,
            str(ldr),
            "--export-collada", str(dae),
            "--libpath", str(ldraw_lib_path),
        ]

        print("[RUN]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print(f"[OK]  {ldr.name} -> {dae.name}")
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"[FAIL] {ldr.name} (exit={e.returncode})")

    if failures:
        raise SystemExit(f"Finished with {failures} failures.")
    print(f"Done. Exported {len(ldr_files)} file(s) to: {outdir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
