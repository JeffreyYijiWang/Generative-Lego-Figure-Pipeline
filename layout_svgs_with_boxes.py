import argparse
import math
import os
import re
import subprocess
from pathlib import Path

from lxml import etree

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {"svg": SVG_NS, "xlink": XLINK_NS}

PX_PER_IN = 96.0  # CSS px per inch

URL_REF_RE = re.compile(r"url\(#([^)]+)\)")
HREF_HASH_RE = re.compile(r"^#(.+)$")


def parse_length_to_px(s: str) -> float:
    """Parse an SVG length like '4in', '100mm', '300px', '12pt' into px (96/in)."""
    if s is None:
        raise ValueError("Missing length")
    s = str(s).strip()
    m = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z%]*)$", s)
    if not m:
        raise ValueError(f"Unrecognized length: {s}")
    val = float(m.group(1))
    unit = (m.group(2) or "px").lower()

    if unit == "px":
        return val
    if unit == "in":
        return val * PX_PER_IN
    if unit == "cm":
        return val * PX_PER_IN / 2.54
    if unit == "mm":
        return val * PX_PER_IN / 25.4
    if unit == "pt":
        return val * PX_PER_IN / 72.0
    if unit == "pc":
        return val * PX_PER_IN / 6.0
    if unit == "%":
        raise ValueError("Percent lengths not supported for width/height parsing.")
    return val


def get_viewbox(svg_root: etree._Element):
    vb = svg_root.get("viewBox")
    if vb:
        parts = [float(x) for x in re.split(r"[,\s]+", vb.strip()) if x]
        if len(parts) != 4:
            raise ValueError(f"Invalid viewBox: {vb}")
        return parts  # (minx, miny, w, h)

    w_attr = svg_root.get("width")
    h_attr = svg_root.get("height")
    if not w_attr or not h_attr:
        return [0.0, 0.0, 4.0 * PX_PER_IN, 2.5 * PX_PER_IN]

    w_px = parse_length_to_px(w_attr)
    h_px = parse_length_to_px(h_attr)
    return [0.0, 0.0, w_px, h_px]


def prefix_ids_and_rewrite_refs(node: etree._Element, prefix: str):
    id_map = {}
    for el in node.iter():
        old = el.get("id")
        if old:
            new = f"{prefix}_{old}"
            id_map[old] = new
            el.set("id", new)

    if not id_map:
        return

    def rewrite_value(val: str) -> str:
        if not val:
            return val

        def repl(m):
            oid = m.group(1)
            return f"url(#{id_map.get(oid, oid)})"

        val2 = URL_REF_RE.sub(repl, val)

        m2 = HREF_HASH_RE.match(val2.strip())
        if m2:
            oid = m2.group(1)
            return f"#{id_map.get(oid, oid)}"

        return val2

    for el in node.iter():
        for attr, val in list(el.attrib.items()):
            if attr == "id":
                continue
            if attr.endswith("href") or ("url(#" in val) or val.strip().startswith("#"):
                el.set(attr, rewrite_value(val))

        style = el.get("style")
        if style and "url(#" in style:
            el.set("style", rewrite_value(style))


def svg_el(tag: str, **attrs):
    el = etree.Element(f"{{{SVG_NS}}}{tag}", nsmap=NSMAP)
    for k, v in attrs.items():
        el.set(k, str(v))
    return el


def run_vpype_optimize(
    vpype_exe: str,
    in_svg: Path,
    out_svg: Path,
    linemerge_tol: str,
    linesimplify_tol: str,
    page_size: str,
    landscape: bool,
):
    """
    vpype read in.svg linemerge ... linesimplify ... linesort pagesize [--landscape] SIZE write out.svg

    NOTE: pagesize only changes metadata / page size. It does NOT rotate geometry.
    """
    cmd = [
        vpype_exe,
        "read", str(in_svg),
        "linemerge", "--tolerance", linemerge_tol,
        "linesimplify", "--tolerance", linesimplify_tol,
        "linesort",
    ]

    if page_size:
        cmd += ["pagesize"]
        if landscape:
            cmd += ["--landscape"]
        cmd += [page_size]

    cmd += ["write", str(out_svg)]

    print("[INFO] vpype:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputdir", required=True, help="Folder of input SVGs")
    ap.add_argument("--outdir", required=True, help="Output folder")

    ap.add_argument("--paper", choices=["9x12","12x12", "letter"], default="9x12")
    ap.add_argument("--cellw", type=float, default=4.0, help="Cell width (inches)")
    ap.add_argument("--cellh", type=float, default=2.5, help="Cell height (inches)")
    ap.add_argument("--margin", type=float, default=0.5, help="Page margin (inches)")
    ap.add_argument("--gutter", type=float, default=0.25, help="Space between cells (inches)")
    ap.add_argument("--clip", action="store_true", help="Clip artwork to each cell")

    # ---- vpype options ----
    ap.add_argument("--vpype", default="vpype", help="Path to vpype executable (or 'vpype' if on PATH)")
    ap.add_argument("--no-vpype", action="store_true", help="Disable vpype optimization step")
    ap.add_argument("--linemerge-tol", default="0.5mm", help="vpype linemerge tolerance (e.g. 0.5mm)")
    ap.add_argument("--linesimplify-tol", default="0.1mm", help="vpype linesimplify tolerance (e.g. 0.1mm)")
    ap.add_argument("--vpype-suffix", default="_vpype", help="Suffix for optimized output (default: _vpype)")

    # ---- vpype page size / landscape metadata ----
    ap.add_argument("--vpype-page-size", default="", help="vpype page size string (e.g. 9x12in, letter). If empty, inferred from --paper")
    ap.add_argument("--vpype-landscape", action="store_true", help="vpype pagesize --landscape (metadata only)")

    args = ap.parse_args()

    in_dir = Path(args.inputdir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.paper == "9x12":
        page_w_in, page_h_in = 9.0, 12.0
        out_base = "layout_9x12"
    elif args.paper == "12x12":
        page_w_in, page_h_in = 12.0, 12.0
        out_base = "layout_9x12"
    else:
        page_w_in, page_h_in = 8.5, 11.0
        out_base = "layout_letter"
   

    page_w = page_w_in * PX_PER_IN
    page_h = page_h_in * PX_PER_IN
    margin = args.margin * PX_PER_IN
    gutter = args.gutter * PX_PER_IN
    cell_w = args.cellw * PX_PER_IN
    cell_h = args.cellh * PX_PER_IN

    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    cols = max(1, int((usable_w + gutter) // (cell_w + gutter)))
    rows = max(1, int((usable_h + gutter) // (cell_h + gutter)))
    per_page = cols * rows

    svg_files = sorted([p for p in in_dir.glob("*.svg")])
    if not svg_files:
        raise SystemExit(f"No .svg files found in {in_dir}")

    total_pages = math.ceil(len(svg_files) / per_page)

    for page_idx in range(total_pages):
        start = page_idx * per_page
        end = min(len(svg_files), (page_idx + 1) * per_page)
        batch = svg_files[start:end]

        root = svg_el(
            "svg",
            width=f"{page_w_in}in",
            height=f"{page_h_in}in",
            viewBox=f"0 0 {page_w} {page_h}",
            version="1.1",
        )
        defs = svg_el("defs")
        root.append(defs)

        for i, svg_path in enumerate(batch):
            r = i // cols
            c = i % cols

            cell_x = margin + c * (cell_w + gutter)
            cell_y = margin + r * (cell_h + gutter)

            # Fixed-size border: ALWAYS cell_w x cell_h
            root.append(svg_el(
                "rect",
                x=cell_x, y=cell_y,
                width=cell_w, height=cell_h,
                fill="none",
                stroke="black",
                **{"stroke-width": "1"}
            ))

            svg_bytes = svg_path.read_bytes()
            src_root = etree.fromstring(svg_bytes)

            vb_minx, vb_miny, vb_w, vb_h = get_viewbox(src_root)

            # Fit entire viewBox into the cell (preserve aspect ratio), centered
            s = min(cell_w / vb_w, cell_h / vb_h)
            off_x = (cell_w - vb_w * s) * 0.5
            off_y = (cell_h - vb_h * s) * 0.5

            g_attrs = {
                "transform": (
                    f"translate({cell_x + off_x},{cell_y + off_y}) "
                    f"scale({s}) translate({-vb_minx},{-vb_miny})"
                )
            }

            # Optional clip to the cell rectangle (prevents any spill outside the box)
            if args.clip:
                clip_id = f"clip_p{page_idx+1}_i{start+i}"
                cp = svg_el("clipPath", id=clip_id)
                cp.append(svg_el("rect", x=str(cell_x), y=str(cell_y), width=str(cell_w), height=str(cell_h)))
                defs.append(cp)
                g_attrs["clip-path"] = f"url(#{clip_id})"

            g = svg_el("g", **g_attrs)

            # Copy defs + children (prefix IDs so pages don't collide)
            prefix = f"p{page_idx+1}_i{start+i}_{svg_path.stem}"
            src_defs = src_root.find(f"{{{SVG_NS}}}defs")
            if src_defs is not None and len(src_defs):
                defs_copy = etree.fromstring(etree.tostring(src_defs))
                prefix_ids_and_rewrite_refs(defs_copy, prefix)
                for child in list(defs_copy):
                    defs.append(child)

            for child in list(src_root):
                if child.tag == f"{{{SVG_NS}}}defs":
                    continue
                child_copy = etree.fromstring(etree.tostring(child))
                prefix_ids_and_rewrite_refs(child_copy, prefix)
                g.append(child_copy)

            root.append(g)

        # Write raw output
        out_name = f"{out_base}_p{page_idx+1:02d}.svg" if total_pages > 1 else f"{out_base}.svg"
        out_path = out_dir / out_name
        out_path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True))
        print(f"[OK] Wrote {out_path}  (cols={cols}, rows={rows}, items={len(batch)})")

        # Run vpype optimization (linesort + linemerge + linesimplify + optional pagesize landscape metadata)
        if not args.no_vpype:
            vpype_out = out_path.with_name( args.vpype_suffix + out_path.stem + out_path.suffix)

            # Decide vpype page size string
            if args.vpype_page_size:
                vpype_size = args.vpype_page_size
            else:
                vpype_size = "9x12in" if args.paper == "9x12" else "letter"

            try:
                run_vpype_optimize(
                    vpype_exe=args.vpype,
                    in_svg=out_path,
                    out_svg=vpype_out,
                    linemerge_tol=args.linemerge_tol,
                    linesimplify_tol=args.linesimplify_tol,
                    page_size=vpype_size,
                    landscape=args.vpype_landscape,
                )
                print(f"[OK] Wrote optimized {vpype_out}")
            except FileNotFoundError:
                raise SystemExit(
                    f"Could not run vpype ('{args.vpype}'). "
                    f"Install with: pip install vpype  OR pass --vpype C:\\path\\to\\vpype.exe"
                )


if __name__ == "__main__":
    main()
