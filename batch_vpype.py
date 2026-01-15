import argparse
import subprocess
import sys
from pathlib import Path


def run_vpype(vpype_exe: str, in_svg: Path, out_svg: Path, show: bool) -> None:
    cmd = [
        vpype_exe,
        "read", str(in_svg),
        "layout","--fit-to-margins", "1cm", "4inx2.5in",
        "linesimplify",
        "linemerge",
        "reloop",
        "linesort",
        "write", str(out_svg),
    ]
    if show:
        cmd.append("show")

    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch process SVGs with vpype.")
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder containing .svg files")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder")
    ap.add_argument("--vpype", default="vpype", help="vpype executable (default: vpype)")
    ap.add_argument("--show", action="store_true", help="Open vpype viewer for each output (slow)")
    ap.add_argument("--pattern", default="*.svg", help="Glob pattern (default: *.svg)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] Input folder does not exist: {in_dir}", file=sys.stderr)
        return 2

    svgs = sorted(in_dir.glob(args.pattern))
    if not svgs:
        print(f"[WARN] No files matched {args.pattern} in {in_dir}")
        return 0

    failures = 0
    for in_svg in svgs:
        if not in_svg.is_file():
            continue

        out_svg = out_dir / f"{in_svg.stem}_vpype.svg"
        try:
            run_vpype(args.vpype, in_svg, out_svg, args.show)
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"[ERROR] vpype failed for {in_svg.name} (exit={e.returncode})", file=sys.stderr)

    if failures:
        print(f"[DONE] Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print(f"[DONE] Processed {len(svgs)} file(s) into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())