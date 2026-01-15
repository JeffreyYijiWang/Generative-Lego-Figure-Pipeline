#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Data model for a minifig entry
# -----------------------------

@dataclass(frozen=True)
class MinifigEntry:
    section: str
    display_name: str
    part_file: str  # e.g. "6131.DAT" (may be "" for None/hidden)
    flags: int
    matrix: Tuple[float, float, float, float, float, float, float, float, float]  # a11..a33
    offset: Tuple[float, float, float]  # x y z
    line_no: int


# -----------------------------
# Robust parsing utilities
# -----------------------------

_SECTION_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")

def _parse_quoted(s: str, i: int) -> Tuple[str, int]:
    """
    Parse a double-quoted string starting at s[i] == '"'.
    Supports doubled quotes ("") inside the string.
    Returns (value, next_index).
    """
    if i >= len(s) or s[i] != '"':
        raise ValueError("Expected opening quote")

    i += 1
    out = []
    while i < len(s):
        c = s[i]
        if c == '"':
            if i + 1 < len(s) and s[i + 1] == '"':
                out.append('"')
                i += 2
                continue
            i += 1  # consume closing quote
            break
        out.append(c)
        i += 1
    return "".join(out), i

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i

def _parse_minifig_line(line: str) -> Optional[Tuple[str, str, int, Tuple[float, ...], Tuple[float, ...]]]:
    """
    Parse:
      "<Display name>" "<DAT/LDR file name>" <Flags> <Matrix9> <Offset3> [ignored tail]

    Returns: (display, part_file, flags, matrix9, offset3) or None if not a minifig line.
    """
    i = _skip_ws(line, 0)
    if i >= len(line) or line[i] != '"':
        return None

    display, i = _parse_quoted(line, i)
    i = _skip_ws(line, i)

    if i >= len(line) or line[i] != '"':
        return None

    part_file, i = _parse_quoted(line, i)
    i = _skip_ws(line, i)

    tail = line[i:].strip()
    if not tail:
        return None

    toks = tail.split()

    # Expect: flags + 9 matrix + 3 offset = 13 numeric tokens minimum.
    # Some lines may be malformed; skip them.
    if len(toks) < 13:
        return None

    try:
        flags = int(float(toks[0]))
        nums = list(map(float, toks[1:13]))  # 12 numbers: matrix9 + offset3
        matrix = tuple(nums[:9])
        offset = tuple(nums[9:12])
    except Exception:
        return None

    return display, part_file, flags, matrix, offset


# -----------------------------
# Index building
# -----------------------------

DEFAULT_EXCLUDE_SECTIONS = {"SCAN_ORDER", "LSYNTH"}

def build_section_index(
    ini_path: str | Path,
    include_sections: Optional[set[str]] = None,
    exclude_sections: Optional[set[str]] = None,
    include_none_entries: bool = True,
) -> Dict[str, List[MinifigEntry]]:
    """
    Reads MLCad.ini once and returns:
      index[section] -> list of MinifigEntry

    include_sections:
      if provided, only these sections are indexed (fast if you only want minifig sections)

    exclude_sections:
      defaults to {"SCAN_ORDER","LSYNTH"} which are not minifig pick-lists.

    include_none_entries:
      if False, drop entries where part_file == "" (e.g. "None").
    """
    ini_path = Path(ini_path).expanduser().resolve()
    if not ini_path.exists():
        raise FileNotFoundError(str(ini_path))

    ex = set(exclude_sections) if exclude_sections is not None else set(DEFAULT_EXCLUDE_SECTIONS)

    index: Dict[str, List[MinifigEntry]] = {}
    current_section: Optional[str] = None

    with ini_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith(";"):
                continue

            m = _SECTION_RE.match(line)
            if m:
                current_section = m.group("name").strip()
                continue

            if current_section is None:
                continue
            if current_section in ex:
                continue
            if include_sections is not None and current_section not in include_sections:
                continue

            parsed = _parse_minifig_line(line)
            if not parsed:
                continue

            display, part_file, flags, matrix, offset = parsed

            if not include_none_entries and part_file.strip() == "":
                continue

            entry = MinifigEntry(
                section=current_section,
                display_name=display,
                part_file=part_file,
                flags=flags,
                matrix=matrix,  # a11..a33
                offset=offset,  # x y z
                line_no=line_no,
            )
            index.setdefault(current_section, []).append(entry)

    return index


# -----------------------------
# Cache to avoid re-parsing each run
# -----------------------------

def _file_fingerprint(path: Path) -> str:
    """
    Cheap-but-safe fingerprint: size + mtime + sha1(first+last 64KB).
    This makes cached loads very reliable even if a tool rewrites timestamps oddly.
    """
    st = path.stat()
    size = st.st_size
    mtime_ns = st.st_mtime_ns

    h = hashlib.sha1()
    h.update(f"{size}:{mtime_ns}".encode("utf-8"))

    with path.open("rb") as f:
        head = f.read(65536)
        if size > 65536:
            f.seek(max(0, size - 65536))
        tail = f.read(65536)
    h.update(head)
    h.update(tail)
    return h.hexdigest()

def load_or_build_index(
    ini_path: str | Path,
    cache_path: str | Path,
    include_sections: Optional[set[str]] = None,
    exclude_sections: Optional[set[str]] = None,
    include_none_entries: bool = True,
) -> Dict[str, List[MinifigEntry]]:
    ini_path = Path(ini_path).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    fp = _file_fingerprint(ini_path)

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("fingerprint") == fp:
                return payload["index"]
        except Exception:
            pass  # fall back to rebuild

    index = build_section_index(
        ini_path=ini_path,
        include_sections=include_sections,
        exclude_sections=exclude_sections,
        include_none_entries=include_none_entries,
    )

    with cache_path.open("wb") as f:
        pickle.dump({"fingerprint": fp, "index": index}, f, protocol=pickle.HIGHEST_PROTOCOL)

    return index


# -----------------------------
# Example usage / quick test
# -----------------------------

if __name__ == "__main__":
    # Adjust paths for your setup:
    ini = "MLCad.ini"
    cache = "mlcad_index.pkl"

    # The exact sections you listed:
    want = {
        "HATS","HATS2","HEAD","BODY","BODY2","BODY3","NECK","LARM","RARM",
        "LHAND","RHAND","LHANDA","RHANDA","LLEG","RLEG","LLEGA","RLEGA",
    }

    idx = load_or_build_index(ini, cache_path=cache, include_sections=want, include_none_entries=True)

    # Fast lookups:
    print("Sections indexed:", sorted(idx.keys()))
    for s in sorted(want):
        print(f"{s}: {len(idx.get(s, []))} entries")
