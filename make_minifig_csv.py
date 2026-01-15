#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

EXCLUDE_SECTIONS_DEFAULT = {"SCAN_ORDER", "LSYNTH"}

def parse_quoted(s: str, i: int):
    """
    Parse a double-quoted string starting at s[i] == '"'.
    Supports doubled quotes ("") inside the quoted string.
    Returns (value, next_index).
    """
    if i >= len(s) or s[i] != '"':
        raise ValueError("parse_quoted: expected opening quote")

    i += 1
    out = []
    while i < len(s):
        c = s[i]
        if c == '"':
            # doubled quote -> literal quote
            if i + 1 < len(s) and s[i + 1] == '"':
                out.append('"')
                i += 2
                continue
            # end quote
            i += 1
            break
        out.append(c)
        i += 1
    return "".join(out), i

def skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i

def parse_minifig_line(line: str):
    """
    Parses:
      "<Display name>" "<DAT/LDR file name>" <Flags> <Matrix 9 nums> <Offset 3 nums> [anything else]

    Returns dict with parsed fields.
    """
    i = 0
    i = skip_ws(line, i)
    if i >= len(line) or line[i] != '"':
        return None  # not a minifig entry line

    display, i = parse_quoted(line, i)
    i = skip_ws(line, i)

    if i >= len(line) or line[i] != '"':
        return None

    part_file, i = parse_quoted(line, i)
    i = skip_ws(line, i)

    tail = line[i:].strip()
    if not tail:
        return None

    tokens = tail.split()
    # Expect at least: flags + 9 matrix + 3 offset = 13 numeric tokens
    # But we keep this tolerant: parse what we can, store remainder in raw_tail.
    nums = []
    raw_tail = []

    for t in tokens:
        try:
            # allow ints or floats
            nums.append(float(t))
        except ValueError:
            raw_tail.append(t)

    flags = nums[0] if len(nums) >= 1 else None
    matrix = nums[1:10] if len(nums) >= 10 else []
    offset = nums[10:13] if len(nums) >= 13 else []

    # Keep any numeric extras too (rare, but safe):
    extra_nums = nums[13:] if len(nums) > 13 else []
    if extra_nums:
        raw_tail = [*raw_tail, *map(str, extra_nums)]

    return {
        "display_name": display,
        "part_file": part_file,
        "flags": flags,
        "matrix": matrix,
        "offset": offset,
        "raw_tail": " ".join(raw_tail).strip(),
    }

def iter_sections(ini_path: Path):
    """
    Yields (section_name, line_no, line_text) for all lines in file.
    """
    section = None
    with ini_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                continue
            yield section, idx, line

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ini", required=True, help="Path to MLCad.ini")
    ap.add_argument("--out", default="minifigs.csv", help="Output CSV path")
    ap.add_argument(
        "--include-empty",
        action="store_true",
        help="Include entries with empty part_file (e.g., 'None', separators). Default: exclude.",
    )
    ap.add_argument(
        "--exclude-sections",
        default=",".join(sorted(EXCLUDE_SECTIONS_DEFAULT)),
        help="Comma-separated section names to ignore (default: SCAN_ORDER,LSYNTH)",
    )
    ap.add_argument(
        "--only-sections",
        default="",
        help="Optional comma-separated whitelist of sections to include (e.g. HATS,HEADS,BODY2).",
    )
    args = ap.parse_args()

    ini_path = Path(args.ini).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not ini_path.exists():
        raise SystemExit(f"[ERROR] ini not found: {ini_path}")

    exclude = {s.strip() for s in args.exclude_sections.split(",") if s.strip()}
    only = {s.strip() for s in args.only_sections.split(",") if s.strip()}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "section",
        "display_name",
        "part_file",
        "flags",
        "m11","m12","m13","m21","m22","m23","m31","m32","m33",
        "offx","offy","offz",
        "raw_tail",
        "source_line",
    ]

    rows = []
    for section, line_no, line in iter_sections(ini_path):
        if section is None:
            continue

        if section in exclude:
            continue

        if only and section not in only:
            continue

        parsed = parse_minifig_line(line)
        if not parsed:
            continue

        part_file = parsed["part_file"].strip()
        if not args.include_empty and part_file == "":
            continue

        matrix = parsed["matrix"]
        offset = parsed["offset"]

        # Pad missing numeric fields for consistent CSV shape:
        matrix = (matrix + [None]*9)[:9]
        offset = (offset + [None]*3)[:3]

        row = {
            "section": section,
            "display_name": parsed["display_name"],
            "part_file": part_file,
            "flags": parsed["flags"],
            "m11": matrix[0], "m12": matrix[1], "m13": matrix[2],
            "m21": matrix[3], "m22": matrix[4], "m23": matrix[5],
            "m31": matrix[6], "m32": matrix[7], "m33": matrix[8],
            "offx": offset[0], "offy": offset[1], "offz": offset[2],
            "raw_tail": parsed["raw_tail"],
            "source_line": line_no,
        }
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to: {out_path}")
    if not rows:
        print("[WARN] No rows found. If this is unexpected, try --include-empty or check --exclude-sections/--only-sections.")

if __name__ == "__main__":
    main()


#python make_minifigs_from_csv_with_colors.py `
#  --csv ".\minifigs.csv" `
#  --outdir "C:\path\to\out_minifigs" `
#  --count 200 `
#  --seed 1 `
#  --p-hat2 0.42 `
#  --p-lhanda 0.82 `
#  --p-rhanda 0.36 `
#  --p-llega 0.76 `
#  --p-rlega 0.6