import os
import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.measure import find_contours
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
import svgwrite

def edge_png_to_svg(
    png_path: str,
    svg_path: str,
    sigma: float = 16.0,
    low_threshold: float = 0.08,
    high_threshold: float = 0.20,
    simplify_tol: float = 3.0,
    make_polygons: bool = False,
):
    img = Image.open(png_path).convert("L")  # grayscale
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Pixel-based edge detection (Canny)
    edges = canny(arr, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

    # Trace contours from the binary edge image
    contours = find_contours(edges.astype(np.uint8), level=0.5)

    h, w = edges.shape
    dwg = svgwrite.Drawing(svg_path, size=(w, h), profile="tiny")

    lines = []
    for c in contours:
        # c is Nx2 in (row, col) == (y, x)
        pts = [(float(p[1]), float(p[0])) for p in c]  # (x, y)
        if len(pts) < 2:
            continue
        ls = LineString(pts)
        if simplify_tol and simplify_tol > 0:
            ls = ls.simplify(simplify_tol, preserve_topology=False)
        if ls.length > 5:  # ignore tiny fragments
            lines.append(ls)

    if not make_polygons:
        # Export polylines
        for ls in lines:
            coords = list(ls.coords)
            dwg.add(dwg.polyline(points=coords, fill="none", stroke="black", stroke_width=1))
    else:
        # Convert edge network to polygons (optional; depends on how “closed” your edges are)
        merged = unary_union(lines)
        polys = list(polygonize(merged))
        for p in polys:
            dwg.add(dwg.polygon(points=list(p.exterior.coords), fill="none", stroke="black", stroke_width=1))

    dwg.save()
    print(f"[OK] Wrote SVG: {svg_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", required=True)
    ap.add_argument("--svg", required=True)
    ap.add_argument("--polygons", action="store_true")
    args = ap.parse_args()

    edge_png_to_svg(
        png_path=args.png,
        svg_path=args.svg,
        make_polygons=args.polygons,
    )