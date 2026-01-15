import bpy
import sys
import os
import math
import argparse
import addon_utils
import subprocess
import textwrap
from mathutils import Vector, Euler
import os
from contextlib import contextmanager

# ---------- parse args after '--' ----------
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []
ap = argparse.ArgumentParser()
ap.add_argument("--inputdir", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--pngsuffix", default="_render.svg")
args = ap.parse_args(argv)

INPUT_DIR = os.path.abspath(args.inputdir)
OUT_DIR = os.path.abspath(args.outdir)
PNG_SUFFIX = args.pngsuffix

#print(f"[INFO] Input dir: {INPUT_DIR}")
#print(f"[INFO] Output dir: {OUT_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# CONSTANTS & ADDON HELPERS
# ----------------------------------------------------------------------
FREESTYLE_SVG_ADDON_IDS = (
    "bl_ext.blender_org.freestyle_svg_exporter", 
    "render_freestyle_svg", 
    "freestyle_svg",
    "io_export_freestyle_svg",
)

from pathlib import Path

STAGES = {
    "plain_png": Path(OUT_DIR) / "plain_png",
    "edges_svg": Path(OUT_DIR) / "edges_svg",
    "freestyle_png": Path(OUT_DIR) / "freestyle_png",
    "freestyle_svg": Path(OUT_DIR) / "freestyle_svg",
    "vpype_edges_svg": Path(OUT_DIR) / "vpype_edges_svg",
    "vpype_freestyle_svg": Path(OUT_DIR) / "vpype_freestyle_svg",
    "combined_svg": Path(OUT_DIR) / "combined_svg",
    "combined2_svg": Path(OUT_DIR) / "combined2_svg",
    "scripts": Path(OUT_DIR) / "_scripts",
}




def enable_freestyle_svg_addon():
    """Tries to enable the Freestyle SVG exporter add-on."""
    installed = {m.__name__ for m in addon_utils.modules()}
    found = next((aid for aid in FREESTYLE_SVG_ADDON_IDS if aid in installed), None)
    if not found:
        #print("[INFO] Freestyle SVG exporter not installed.")
        return False
    try:
        en, ld = addon_utils.check(found)
    except Exception:
        en = ld = False
    
    if not (en and ld):
        try:
            # enable without setting persistent=True in case running outside of UI
            addon_utils.enable(found, default_set=True) 
            #print(f"[INFO] Enabled add-on: {found}")
        except Exception as e:
            print(f"[WARN] Could not enable {found}: {e}")
            return False
    else:
        print(f"[INFO] Add-on already enabled: {found}")
    return True

def ensure_freestyle_core():
    """Enables the main Freestyle feature on the scene and view layer."""
    scn = bpy.context.scene
    vl  = bpy.context.view_layer
    if hasattr(scn.render, "use_freestyle"):
        scn.render.use_freestyle = True
    if hasattr(vl, "use_freestyle"):
        vl.use_freestyle = True
    #print(f"[INFO] use_freestyle set. scene={getattr(scn.render,'use_freestyle',None)}")


def get_imported_objects():
    # ignore anything you create later (camera/light) and any Blender defaults
    return [o for o in bpy.data.objects if o.type not in {'CAMERA','LIGHT'}]

def make_root_empty(name="ROOT"):
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    return root

def parent_keep_world(child, parent):
    # preserve world transform while parenting
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    child.parent = parent

def group_under_root(objs):
    root = make_root_empty()
    for o in objs:
        if o is root:
            continue
        parent_keep_world(o, root)
    return root

def bounds_of_objects(objs):
    mn = Vector(( float("inf"), float("inf"), float("inf")))
    mx = Vector((float("-inf"),float("-inf"),float("-inf")))
    for o in objs:
        if o.type != 'MESH':
            continue
        for v in o.bound_box:
            w = o.matrix_world @ Vector(v)
            mn.x = min(mn.x, w.x); mn.y = min(mn.y, w.y); mn.z = min(mn.z, w.z)
            mx.x = max(mx.x, w.x); mx.y = max(mx.y, w.y); mx.z = max(mx.z, w.z)
    return mn, mx

def center_and_scale_root(root, target_max=45.0):
    # children meshes only
    children = [o for o in root.children_recursive]
    mn, mx = bounds_of_objects(children)
    size = mx - mn
    m = max(size.x, size.y, size.z)
    if m <= 0:
        #print("[WARN] Zero bounds; skipping center/scale")
        return

    center = (mn + mx) * 0.5

    # move root so the model is centered at origin:
    root.location -= center

    # apply uniform scale:
    factor = target_max / m
    root.scale *= factor

    # IMPORTANT: do NOT apply transforms on children
    # Optionally apply on root only (usually unnecessary for rendering)
    #print(f"[INFO] Root-centered and scaled. factor={factor:.4f}")


def apply_root_rotation_deg(root, rx, ry, rz):
    root.rotation_mode = 'XYZ'
    root.rotation_euler = Euler((math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ')
# ----------------------------------------------------------------------
# CORE HELPER FUNCTIONS (Simplified and merged)
# ----------------------------------------------------------------------

def new_clean_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0

def scene_bounds(objs):
    mnx=mny=mnz=float("inf")
    mxx=mxy=mxz=float("-inf")
    for o in objs:
        for v in o.bound_box:
            w = o.matrix_world @ Vector(v)
            mnx=min(mnx,w.x); mny=min(mny,w.y); mnz=min(mnz,w.z)
            mxx=max(mxx,w.x); mxy=max(mxy,w.y); mxz=max(mxz,w.z)
    return (mnx,mny,mnz),(mxx,mxy,mxz)

def scale_scene_to_max(objs, target_max=45.0):
    if not objs:
        #print("[WARN] No objects to scale.")
        return []
    (mnx, mny, mnz), (mxx, mxy, mxz) = scene_bounds(objs)
    dx, dy, dz = (mxx - mnx, mxy - mny, mxz - mnz)
    m = max(dx, dy, dz)
    if m <= 0:
        #print("[WARN] Scene has zero dimension; skipped scaling.")
        return objs

    factor = target_max / m
    vl = bpy.context.view_layer
    for o in objs:
        if o.type == 'MESH' and o.data and o.data.users > 1:
            o.data = o.data.copy()
        o.scale = (o.scale[0] * factor, o.scale[1] * factor, o.scale[2] * factor)
        try:
            for sel in bpy.context.selected_objects: sel.select_set(False)
            vl.objects.active = o
            o.select_set(True)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        except Exception as e:
            print(f"[WARN] transform_apply failed for {o.name}: {e}")
        finally:
            o.select_set(False)
            
    #print(f"[INFO] Applied scene-wide scale factor: {factor:.4f}")
    return objs

def setup_camera_and_light(focus_objs):
    scn = bpy.context.scene
    if focus_objs:
        (mnx,mny,mnz),(mxx,mxy,mxz) = scene_bounds(focus_objs)
        cx,cy,cz = ( (mnx+mxx)/2, (mny+mxy)/2, (mnz+mxz)/2 )
        dx,dy,dz = (mxx-mnx, mxy-mny, mxz-mnz)
        diag = (dx*dx+dy*dy+dz*dz)**0.5
    else:
        cx,cy,cz = 0,0,0
        diag = 10.0

    target = Vector((cx, cy, cz))

    # ---- Camera ----
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    dist = max(10.0, diag * 1.8)
    cam.location = (cx + dist, cy - dist, cz + dist * 0.6)

    cam_dir = target - cam.location
    cam.rotation_euler = cam_dir.to_track_quat('-Z', 'Y').to_euler()
    scn.camera = cam

    # ---- Headlight (SPOT at camera, aimed at target) ----
    light_data = bpy.data.lights.new(name="HeadLight", type='SPOT')
    light = bpy.data.objects.new(name="HeadLight", object_data=light_data)
    bpy.context.collection.objects.link(light)

    # put light basically at the camera (tiny offset so it’s not coincident)
    light.location = cam.location.copy()

    # aim light at the model center
    light_dir = target - light.location
    light.rotation_euler = light_dir.to_track_quat('-Z', 'Y').to_euler()

    # tune “flashlight” feel
    light_data.energy = 3000000.0
    light_data.spot_size = math.radians(80.0)   # wider cone so you don’t miss the subject
    light_data.spot_blend = 0.10

    # optional: soften hotspot a bit
    if hasattr(light_data, "shadow_soft_size"):
        light_data.shadow_soft_size = 0.25

def ensure_freestyle_lineset():
    vl = bpy.context.view_layer
    try:
        fs = vl.freestyle_settings
    except Exception:
        #print("[WARN] View layer has no freestyle_settings; skipping line set setup.")
        return

    if not fs.linesets:
        try:
            fs.linesets.new("LineSet")
        except Exception:
            return
    ls = fs.linesets[0]

    if ls.linestyle is None:
        style_name = "LineStyle"
        linestyle = bpy.data.linestyles.get(style_name)
        if linestyle is None:
            try:
                linestyle = bpy.data.linestyles.new(style_name)
            except Exception:
                return
        ls.linestyle = linestyle

    # Edge Type checkboxes (matching your intent)
    for flag in (
        "select_silhouette", "select_crease", "select_border", 
        "select_edge_mark", "select_contour", "select_by_image_border", 
    ):
        if hasattr(ls, flag):
            try:
                setattr(ls, flag, True)
            except Exception:
                pass
    
    # Line Thickness
    try:
        ls.linestyle.thickness = 2.0
    except Exception:
        pass
# --- NEW FUNCTION TO FIX EXPLODED MESHES ---
# --- REVISED FUNCTION TO FIX EXPLODED MESHES USING MATRIX MATH ---
def fix_object_origins(objs):
    """
    Applies current object world transform (position, rotation) directly 
    to the mesh geometry data and resets object transforms.
    This fixes objects that are 'exploded' due to origins being reset to (0,0,0) 
    by leveraging direct matrix math, which is more robust in headless mode.
    """
    vl = bpy.context.view_layer
    
    for obj in objs:
        # We only need to process objects with geometry data
        if obj.type == 'MESH' and obj.data:
            #print(f"[INFO] Fixing transforms for mesh: {obj.name}")
            
            # 1. Get the object's current transformation matrix (World Matrix)
            matrix_world = obj.matrix_world.copy()
            
            # 2. Apply this world matrix to every vertex in the mesh
            try:
                # Enter Object Mode to access geometry data (important for reliability)
                if bpy.context.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                
                # Check for multiple users and copy mesh data if necessary (good practice)
                if obj.data.users > 1:
                    obj.data = obj.data.copy()
                    
                # Transform the mesh data using the world matrix
                obj.data.transform(matrix_world)
                
                # 3. Reset the object's transform to identity (Location: 0, Rotation: 0, Scale: 1)
                # The world position is now baked into the mesh vertices.
                obj.matrix_world.identity()
                
                # 4. Set the object's origin to the center of its new, correct geometry
                # This requires selection, making it active, and an operator call.
                # We minimize the use of ops here for better stability.
                vl.objects.active = obj
                obj.select_set(True)
                
                # Use bpy.ops.object.origin_set, as there's no direct matrix equivalent
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                
            except Exception as e:
                print(f"[WARN] Failed matrix-based fix for {obj.name}: {e}")
            finally:
                obj.select_set(False)
        
        elif obj.type in {'EMPTY', 'ARMATURE'}:
            # For non-mesh objects that define transforms (like armatures or parents), 
            # we simply reset their transforms to identity after the children are baked.
            obj.matrix_world.identity()

    # Deselect all after the loop
    bpy.ops.object.select_all(action='DESELECT')
# ----------------------------------------------------------------------
# RENDER CONFIG
# ----------------------------------------------------------------------

def render_plain_png(png_path):
    scn = bpy.context.scene

    # Disable freestyle everywhere:
    if hasattr(scn.render, "use_freestyle"):
        scn.render.use_freestyle = False
    for vl in scn.view_layers:
        if hasattr(vl, "use_freestyle"):
            vl.use_freestyle = False

    # Render settings:
    scn.render.engine = 'BLENDER_EEVEE_NEXT'
    scn.render.image_settings.file_format = 'PNG'
    scn.render.filepath = png_path

    #print(f"[INFO] Rendering plain PNG (no Freestyle): {png_path}.png")
    bpy.ops.render.render(write_still=True)

def configure_render(out_path_prefix, svg_path):
    scn = bpy.context.scene

    # --- Engine and PNG Config ---
    scn.render.engine = 'BLENDER_EEVEE_NEXT' 
    scn.render.resolution_x = 1920
    scn.render.resolution_y = 1080
    scn.render.film_transparent = False
    scn.render.image_settings.file_format = 'PNG'
    # Set the render path prefix. The SVG exporter uses this to build the final SVG path.
    scn.render.filepath = out_path_prefix 

    # --- Freestyle Core and Addon ---
    enable_freestyle_svg_addon()
    ensure_freestyle_core()
    
    # --- CRITICAL FIX: ENABLE AUTO-EXPORT HANDLER ---
    # The add-on exports the SVG file automatically as a handler during render. 
    # This property enables the handler.
    if hasattr(scn, "svg_export"):
        scn.svg_export.use_svg_export = True
        #print("[INFO] CRITICAL: Set scene.svg_export.use_svg_export = True to enable handler.")
        
        # Set to 'FRAME' mode to ensure a single file is outputted and named predictably
        scn.svg_export.mode = 'FRAME'
    # --------------------------------------------------

    # --- View Layer/Lineset Setup ---
    for vl in scn.view_layers:
        vl.use_freestyle = True
        ensure_freestyle_lineset()

EDGE_TO_SVG_PY = r"""
import argparse
import os
import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import thin
from shapely.geometry import LineString
import svgwrite

# 8-neighborhood offsets
N8 = [(-1,-1),(-1,0),(-1,1),
      ( 0,-1),        ( 0,1),
      ( 1,-1),( 1,0),( 1,1)]

def _neighbors(y, x, H, W, skel):
    for dy, dx in N8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
            yield (ny, nx)

def _degree(y, x, H, W, skel):
    return sum(1 for _ in _neighbors(y, x, H, W, skel))

def trace_polylines_from_skeleton(skel):
    H, W = skel.shape
    skel = skel.astype(bool)

    # Represent edges as undirected pixel adjacency; mark visited edges not nodes.
    visited = set()

    def edge_key(a, b):
        # canonical undirected edge key
        return (a, b) if a <= b else (b, a)

    endpoints = [(y, x) for y, x in zip(*np.where(skel)) if _degree(y, x, H, W, skel) == 1]
    junctions = [(y, x) for y, x in zip(*np.where(skel)) if _degree(y, x, H, W, skel) >= 3]

    # For fast membership:
    is_junction = set(junctions)

    polylines = []

    def walk(start, nxt):
        path = [(start[1], start[0])]  # (x,y)
        prev = start
        curr = nxt

        while True:
            visited.add(edge_key(prev, curr))
            path.append((curr[1], curr[0]))

            deg = _degree(curr[0], curr[1], H, W, skel)
            if deg == 1 or curr in is_junction:
                break

            # continue along the only unvisited edge if possible
            next_candidates = []
            for nb in _neighbors(curr[0], curr[1], H, W, skel):
                if nb == prev:
                    continue
                if edge_key(curr, nb) in visited:
                    continue
                next_candidates.append(nb)

            if not next_candidates:
                break
            # if multiple candidates, treat as junction-like stop
            if len(next_candidates) > 1:
                break

            prev, curr = curr, next_candidates[0]

        return path

    # 1) Trace from endpoints outward
    for ep in endpoints:
        for nb in _neighbors(ep[0], ep[1], H, W, skel):
            ek = edge_key(ep, nb)
            if ek in visited:
                continue
            poly = walk(ep, nb)
            if len(poly) >= 2:
                polylines.append(poly)

    # 2) Trace remaining edges (cycles / leftover segments)
    ys, xs = np.where(skel)
    pixels = list(zip(ys.tolist(), xs.tolist()))
    for p in pixels:
        for nb in _neighbors(p[0], p[1], H, W, skel):
            ek = edge_key(p, nb)
            if ek in visited:
                continue
            # walk cycle-like
            poly = walk(p, nb)
            if len(poly) >= 2:
                polylines.append(poly)

    return polylines

def edge_png_to_svg(
    png_path,
    svg_path,
    sigma=0.2,
    low=0.02,
    high=0.05,
    simplify=3.0,
    min_length_px=5.0
):
    # tolerate missing extension
    if not os.path.exists(png_path) and os.path.exists(png_path + ".png"):
        png_path = png_path + ".png"

    img = Image.open(png_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    edges = canny(arr, sigma=sigma, low_threshold=low, high_threshold=high)
    skel = thin(edges)  # 1px skeleton

    H, W = skel.shape
    dwg = svgwrite.Drawing(svg_path, size=(W, H), profile="tiny")

    polylines = trace_polylines_from_skeleton(skel)

    for pts in polylines:
        if len(pts) < 2:
            continue
        ls = LineString(pts)
        if simplify and simplify > 0:
            ls = ls.simplify(simplify, preserve_topology=False)
        if ls.length < min_length_px:
            continue
        dwg.add(dwg.polyline(points=list(ls.coords), fill="none", stroke="black", stroke_width=1))

    dwg.save()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", required=True)
    ap.add_argument("--svg", required=True)
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--low", type=float, default=0.02)
    ap.add_argument("--high", type=float, default=0.20)
    ap.add_argument("--simplify", type=float, default=1.0)
    ap.add_argument("--minlen", type=float, default=10.0)
    args = ap.parse_args()

    edge_png_to_svg(
        args.png,
        args.svg,
        sigma=args.sigma,
        low=args.low,
        high=args.high,
        simplify=args.simplify,
        min_length_px=args.minlen,
    )

if __name__ == "__main__":
    main()
"""
def ensure_edge_script(path):
    # Always overwrite each run:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(EDGE_TO_SVG_PY))
    except Exception as e:
        print(f"[WARN] Failed to write edge helper {path}: {e}")

def run_vpype_merge(vpype_exe, freestyle_svg, edges_svg, out_svg):
    # Put BOTH reads into layer 1 so linemerge can merge across both sets
    cmd = [
        vpype_exe,
        "read", "--layer", "1", freestyle_svg,
        "read", "--layer", "1", edges_svg,
        "linemerge", "--tolerance", "3mm",
        "linesimplify", "--tolerance", "0.5mm",
        "linesort",
        "write", out_svg,
    ]
    #print("[INFO] Running vpype:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_vpype_opt(python_exe, in_svg, out_svg):
    cmd = [
        python_exe,
        "read", in_svg,
        "linemerge", "--tolerance", "0.5mm",
        "linesimplify", "--tolerance", "0.1mm",
        "linesort",
        "write", out_svg
    ]
    subprocess.run(cmd, check=True)

def run_vpype_combine(python_exe, svg_a, svg_b, out_svg):
    cmd = [
        python_exe,
        "read", "--layer", "1", svg_a,
        "read", "--layer", "1", svg_b,
        "linemerge", "--tolerance", "2.0mm",
        "linesimplify", "--tolerance", "0.5mm",
        "linesort",
        "write", out_svg
    ]
    subprocess.run(cmd, check=True)


def run_edge_vectorize(python_exe, script_path, plain_png_path, out_svg_path, polygons=False):
    cmd = [
        python_exe,
        script_path,
        "--png", plain_png_path,
        "--svg", out_svg_path,
        "--sigma", "1.2",
        "--low", "0.08",
        "--high", "0.20",
        "--simplify", "1.0",
    ]
    if polygons:
        cmd.append("--polygons")

    #print(f"[INFO] Edge vectorizing: {os.path.basename(plain_png_path)} -> {os.path.basename(out_svg_path)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[WARN] Edge vectorization failed: {e}")

# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------
for p in STAGES.values():
    p.mkdir(parents=True, exist_ok=True)

# Use ONE shared edge helper script for all assets:
edge_script = str(STAGES["scripts"] / "_edge_to_svg.py")
ensure_edge_script(edge_script)
dae_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".dae")]
dae_files.sort()
if not dae_files:
    #print("[WARN] No .dae files found.")
    sys.exit(0)

#print(f"[INFO] Found {len(dae_files)} .dae files.")

for fname in dae_files:
    dae_path = os.path.join(INPUT_DIR, fname)
    base = os.path.splitext(fname)[0]
    
    # 🌟 FIX 1: Correct output paths
    png_prefix = os.path.join(OUT_DIR, base + PNG_SUFFIX).rstrip(".png")
    svg_path = os.path.join(OUT_DIR, base + ".svg") # <-- CORRECTED EXTENSION
    
    #print(f"\n[===] Processing: {fname}")
    new_clean_scene()

    # import
    res = bpy.ops.wm.collada_import(
        filepath=dae_path, 
        import_units=True, # Explicitly use the DAE file's internal unit scale and transforms
        fix_orientation=False
    )
    #print(f"[INFO] Import result: {res}")

    # sanitize IORs
    for m in bpy.data.materials:
        if hasattr(m, "ior"):
            try:
                if m.ior is None or m.ior <= 0:
                    m.ior = 1.45
            except Exception:
                pass

    # scale to 45m
    mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    all_imported_objs = [o for o in bpy.data.objects]
    mesh_objs = scale_scene_to_max(all_imported_objs, target_max=45.0)

    base = os.path.splitext(fname)[0]

    # ---------- Plain PNG (no freestyle) ----------
    plain_prefix = str(STAGES["plain_png"] / f"{base}_plain")  # Blender adds ".png"
    plain_png = plain_prefix + ".png"

    # ---------- Edges SVG (raw edge-detect) ----------
    edges_svg = str(STAGES["edges_svg"] / f"{base}_edges.svg")

    # ---------- Freestyle PNG ----------
    # Blender uses render.filepath as a prefix; it adds ".png"
    freestyle_prefix = str(STAGES["freestyle_png"] / f"{base}_freestyle")
    freestyle_png = freestyle_prefix + ".png"

    # ---------- Freestyle SVG (addon output) ----------
    # We want it to land in freestyle_svg/, so set render.filepath prefix there too.
    # We do this by pointing configure_render at the svg folder prefix.
    freestyle_svg_prefix = str(STAGES["freestyle_svg"] / f"{base}_freestyle")

    # Your add-on naming logic: <render.filepath basename><frame:04d>.svg
    expected_svg_suffix = f"{bpy.context.scene.frame_current:04d}.svg"
    final_svg_path = str(STAGES["freestyle_svg"] / (os.path.basename(freestyle_svg_prefix) + expected_svg_suffix))

    # ---------- vpype outputs ----------
    edges_opt = str(STAGES["vpype_edges_svg"] / f"{base}_edges_opt.svg")
    free_opt  = str(STAGES["vpype_freestyle_svg"] / f"{base}_freestyle_opt.svg")

    # ---------- combined outputs ----------
    combined  = str(STAGES["combined_svg"] / f"{base}_combined.svg")
    combined2 = str(STAGES["combined2_svg"] / f"{base}_combined2.svg")

    # # camera/light
    setup_camera_and_light(mesh_objs)

    # # 1) Plain render (no freestyle)
    # plain_prefix = os.path.join(OUT_DIR, base + "_plain.png")
    # render_plain_png(plain_prefix)

    #     # ---- 2) Edge-detect + vectorize to SVG (one per asset) ----
    # edge_script = os.path.join(OUT_DIR, "_edge_to_svg.py")
    # ensure_edge_script(edge_script)

    # # Use your system python. If needed, replace "python" with an absolute path.
    # python_exe = "python"
    # edges_svg = os.path.join(OUT_DIR, base + "_edges.svg")
    # run_edge_vectorize(python_exe, edge_script, plain_prefix, edges_svg, polygons=False)

    # # 2) Freestyle render (your existing pipeline)
    # configure_render(png_prefix, svg_path)
    # #print("[INFO] Rendering PNG (Freestyle)...")
    # bpy.ops.render.render(write_still=True)
    # #print(f"[INFO] PNG saved: {png_prefix}.png")


    
    # # --- SVG File Check (Based on Add-on's Naming) ---
    # # The add-on's `create_path` function generates the file path using the 
    # # scene's render output path, appending the frame number.
    
    # expected_svg_path_suffix = f"{bpy.context.scene.frame_current:04d}.svg"
    # base_file_name_part = os.path.basename(png_prefix)
    
    # # Reconstruct the file path: [OUT_DIR] + [base filename] + [frame number] + .svg
    # final_svg_path = os.path.join(OUT_DIR, base_file_name_part + expected_svg_path_suffix)
    
    # #print(f"[INFO] Checking for automatically created SVG at: {final_svg_path}")
    
    # if os.path.exists(final_svg_path):
    #     #print(f"[SUCCESS] SVG successfully written automatically: {final_svg_path}")
    # else:
    #     #print(f"[WARN] SVG was NOT successfully created. Expected file not found: {final_svg_path}")

    # vp_py = "vpype"

    # edges_opt = os.path.join(OUT_DIR, base + "_edges_opt.svg")
    # free_opt  = os.path.join(OUT_DIR, base + "_freestyle_opt.svg")
    # combined  = os.path.join(OUT_DIR, base + "_combined.svg")

    # run_vpype_opt(vp_py, edges_svg, edges_opt)
    # run_vpype_opt(vp_py, final_svg_path, free_opt)
    # run_vpype_combine(vp_py, free_opt, edges_opt, combined)
    # combined_svg = os.path.join(OUT_DIR, base + "_combined2.svg")
    # run_vpype_merge("vpype", final_svg_path, edges_svg, combined_svg)

        # camera/light
        
    # 1) Plain render (no freestyle)
    render_plain_png(plain_prefix)

    # 2) Edge-detect + vectorize to SVG
    python_exe = "python"
    run_edge_vectorize(python_exe, edge_script, plain_prefix, edges_svg, polygons=False)

    # 3) Freestyle render
    # IMPORTANT:
    # - We want the PNG in freestyle_png/
    # - We want the SVG in freestyle_svg/
    # The Freestyle SVG exporter uses scene.render.filepath as its basis, so
    # we temporarily set render.filepath to the freestyle_svg_prefix when exporting SVG.

    # First: render Freestyle PNG to freestyle_png/
    configure_render(freestyle_prefix, svg_path=None)  # if your configure_render requires svg_path, keep passing a dummy
    #print("[INFO] Rendering Freestyle PNG...")
    bpy.ops.render.render(write_still=True)
    #print(f"[INFO] Freestyle PNG saved: {freestyle_png}")

    # Second: render again but point render.filepath to freestyle_svg_prefix so SVG lands in freestyle_svg/
    # (PNG output will also be generated again; if you want to avoid it, set a different file_format or overwrite; simplest is accept overwrite)
    configure_render(freestyle_svg_prefix, svg_path=None)
    #print("[INFO] Triggering Freestyle SVG export...")
    bpy.ops.render.render(write_still=True)


    # 4) vpype optimize + combine (separate folders)
    vp_py = "vpype"

    if os.path.exists(edges_svg):
        run_vpype_opt(vp_py, edges_svg, edges_opt)

    if os.path.exists(final_svg_path):
        run_vpype_opt(vp_py, final_svg_path, free_opt)

    if os.path.exists(edges_opt) and os.path.exists(free_opt):
        run_vpype_combine(vp_py, free_opt, edges_opt, combined)

    # Optional: direct merge pipeline into combined2_svg/
    if os.path.exists(final_svg_path) and os.path.exists(edges_svg):
        run_vpype_merge(vp_py, final_svg_path, edges_svg, combined2)


#print("\n[DONE] All files processed.")

#python generate_minifig.py --csv minifigs.csv --outdir out_minifigs --count 1000 --seed 10             