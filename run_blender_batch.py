# run_blender_batch.py
# Launches Blender headless to run dae_to_svg_folder.py using your venv Python as a controller.

# python "C:\Users\Jeffr\OneDrive\Desktop\Drawing-Machines\legovenv\run_blender_batch.py" ^
#   --blender  "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" ^
#   --script   "C:\Users\Jeffr\OneDrive\Desktop\Drawing-Machines\legovenv\dae_to_svg_folder.py" ^
#   --inputdir "C:\Users\Jeffr\OneDrive\Desktop\Drawing-Machines\legovenv\Input" ^
#   --outdir   "C:\Users\Jeffr\OneDrive\Desktop\Drawing-Machines\legovenv\Output" ^
#   --pngsuffix "_render.png"

import argparse
import pathlib
import shlex
import subprocess
import sys
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", required=True,
                    help=r'Path to blender.exe, e.g. "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"')
    ap.add_argument("--script", required=True,
                    help=r'Path to dae_to_svg_folder.py')
    ap.add_argument("--inputdir", required=True,
                    help=r'Folder containing *.dae')
    ap.add_argument("--outdir", required=True,
                    help=r'Output folder')
    ap.add_argument("--pngsuffix", default="_render.png",
                    help="Suffix for PNG outputs (default: _render.png)")
    ap.add_argument("--extra", default="", help="Optional extra args passed after --")
    args = ap.parse_args()

    blender = pathlib.Path(args.blender)
    script  = pathlib.Path(args.script)
    inputdir = pathlib.Path(args.inputdir)
    outdir   = pathlib.Path(args.outdir)

    if not blender.exists():
        sys.exit(f"[ERROR] Blender not found: {blender}")
    if not script.exists():
        sys.exit(f"[ERROR] Script not found: {script}")
    if not inputdir.exists():
        sys.exit(f"[ERROR] Input dir not found: {inputdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Build the Blender command (headless)
    cmd = [
        str(blender),
        "-b", "-noaudio",
        "-P", str(script),
        "--",
        "--inputdir", str(inputdir),
        "--outdir", str(outdir),
        "--pngsuffix", args.pngsuffix,
    ]

    if args.extra:
        # Allow passing extra key/values to your Blender script
        cmd.extend(shlex.split(args.extra))

    print("[INFO] Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    # Capture Blender logs so your venv run sees them:
    result = subprocess.run(cmd, check=False, text=True)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
