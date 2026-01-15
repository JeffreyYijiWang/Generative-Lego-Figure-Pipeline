import argparse
import subprocess
import sys
from pathlib import Path


def run_vpype(vpype_exe: str, in_svg: Path, out_svg: Path, label: str,
              font: str, text_size: float, show: bool) -> None:
    # Keep the eval expression self-contained (no undefined variables).
    # Place label at bottom-right inside the 1cm margin, with an extra 2mm up-shift.
    eval_expr = "m=1*cm; w,h=prop.vp_page_size; x=w-m; y=h-m-2*mm"

    cmd = [
        vpype_exe, 
        "read", str(in_svg),

        # Force 4in x 2.5in landscape and fit artwork to 1cm margins:
        "layout", "--landscape", "--fit-to-margins", "1cm", "4inx2.5in",

        "linesimplify",
        "linemerge",
        "reloop",
        "linesort",

        "text",
            "--layer", "new",
            "--font", font,
            "--size", str(text_size),
            "--align", "right",
            "--position", "%x%,%y%",
            label,

        "write", str(out_svg),
    ]

    if show:
        cmd.append("show")

    # IMPORTANT: list2cmdline shows quoting as Windows will pass it to vpype.
    print("[INFO] Running:", subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True, shell=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--vpype", default="vpype", help="Path to vpype executable, or 'vpype' if on PATH")
    ap.add_argument("--pattern", default="*.svg")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--font", default="futural")
    ap.add_argument("--text-size", type=float, default=18.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    svgs = sorted(p for p in in_dir.glob(args.pattern) if p.is_file())
    if not svgs:
        print(f"[WARN] No SVG files matched {args.pattern} in {in_dir}")
        return 0

    failures = 0
    idx = args.start_index

    for in_svg in svgs:
        out_svg = out_dir / f"{in_svg.stem}_vpype.svg"
        try:
            run_vpype(
                vpype_exe=args.vpype,
                in_svg=in_svg,
                out_svg=out_svg,
                label=str(idx),
                font=args.font,
                text_size=args.text_size,
                show=args.show,
            )
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"[ERROR] vpype failed for {in_svg.name} (exit={e.returncode})", file=sys.stderr)
        idx += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())