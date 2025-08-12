#!/usr/bin/env python3
import argparse, json, math
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional
from collections import defaultdict

# --- SVG parsing / sampling ---
from lxml import etree
from svgpathtools import parse_path, Path, Line, QuadraticBezier, CubicBezier, Arc
from shapely.geometry import Polygon as SPolygon, MultiPolygon, LinearRing, Point as SPoint
from shapely.ops import unary_union, triangulate
from shapely import affinity

# ----------------------------
# Utility: numeric helpers
# ----------------------------
def length_of_segment(seg) -> float:
    try:
        return seg.length(error=1e-4)
    except Exception:
        # fallback: chord length
        return abs(seg.end - seg.start)

def adaptive_sample_segment(seg, max_seg_len: float, min_points: int = 2) -> List[complex]:
    """Sample a segment with approximately-uniform chord length ≤ max_seg_len."""
    L = length_of_segment(seg)
    n = max(min_points, int(math.ceil(max(L / max_seg_len, 1))))
    ts = [i / n for i in range(n)]  # include end with next seg to avoid dup
    pts = [seg.point(t) for t in ts]
    return pts

def adaptive_sample_path(path: Path, max_seg_len: float) -> List[List[Tuple[float,float]]]:
    """Return list of subpaths (each a list of (x,y)), closed if the path is closed."""
    subpaths = []
    for sub in path.continuous_subpaths():
        pts: List[Tuple[float,float]] = []
        for i, seg in enumerate(sub):
            seg_pts = adaptive_sample_segment(seg, max_seg_len)
            for p in seg_pts:
                pts.append((p.real, p.imag))
        # add final endpoint
        if len(sub) > 0:
            p_end = sub[-1].end
            pts.append((p_end.real, p_end.imag))
        # ensure closure if start/end near
        if pts and _dist2(pts[0], pts[-1]) > 1e-16:
            # if original subpath was closed, svgpathtools ensures last equals first;
            # we only close if the distance is tiny; otherwise we leave open.
            pass
        subpaths.append(pts)
    return subpaths

def _dist2(a,b): 
    dx,dy = a[0]-b[0], a[1]-b[1]
    return dx*dx+dy*dy

# ----------------------------
# Transforms (SVG 'transform' attribute)
# ----------------------------
def parse_transform(transform_str: str):
    """Return an affine matrix [a,b,c,d,e,f] like SVG (x' = a*x + c*y + e, y' = b*x + d*y + f)."""
    # Default identity
    a,b,c,d,e,f = 1,0,0,1,0,0
    if not transform_str:
        return a,b,c,d,e,f
    import re
    # Parse commands in order
    for cmd, args in re.findall(r'([a-zA-Z]+)\(([^)]+)\)', transform_str):
        vals = [float(v) for v in re.split(r'[ ,]+', args.strip()) if v]
        cmd = cmd.lower()
        if cmd == 'matrix' and len(vals) == 6:
            ma,mb,mc,md,me,mf = vals
            a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, ma,mb,mc,md,me,mf)
        elif cmd == 'translate':
            tx, ty = (vals + [0.0])[:2]
            a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, 1,0,0,1,tx,ty)
        elif cmd == 'scale':
            sx, sy = (vals + [vals[0]])[:2]
            a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, sx,0,0,sy,0,0)
        elif cmd == 'rotate':
            angle = math.radians(vals[0])
            cosA, sinA = math.cos(angle), math.sin(angle)
            if len(vals) == 3:
                cx, cy = vals[1], vals[2]
                # translate to origin, rotate, translate back
                a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, 1,0,0,1,cx,cy)
                a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, cosA,sinA,-sinA,cosA,0,0)
                a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, 1,0,0,1,-cx,-cy)
            else:
                a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, cosA,sinA,-sinA,cosA,0,0)
        elif cmd == 'skewx':
            ang = math.radians(vals[0])
            a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, 1,0,math.tan(ang),1,0,0)
        elif cmd == 'skewy':
            ang = math.radians(vals[0])
            a,b,c,d,e,f = _mat_mul(a,b,c,d,e,f, 1,math.tan(ang),0,1,0,0)
    return a,b,c,d,e,f

def _mat_mul(a,b,c,d,e,f, A,B,C,D,E,F):
    # Compose two SVG affine transforms
    return (
        a*A + c*B,  b*A + d*B,
        a*C + c*D,  b*C + d*D,
        a*E + c*F + e,  b*E + d*F + f
    )

def apply_transform(points: Iterable[Tuple[float,float]], mat) -> List[Tuple[float,float]]:
    a,b,c,d,e,f = mat
    return [(a*x + c*y + e, b*x + d*y + f) for (x,y) in points]

# ----------------------------
# Geometry helpers
# ----------------------------
def coords_to_shapely_polygon(coords: List[Tuple[float,float]]) -> Optional[SPolygon]:
    if len(coords) < 3: 
        return None
    # Ensure closure & remove duplicate trailing vertex
    if _dist2(coords[0], coords[-1]) > 0:
        coords = coords + [coords[0]]
    try:
        poly = SPolygon(coords)
        if not poly.is_valid or poly.area <= 0:
            poly = poly.buffer(0)  # attempt fix
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None

def polygon_to_coords(poly: SPolygon) -> List[Tuple[float,float]]:
    ring = list(poly.exterior.coords)[:-1]  # drop repeated last
    return [(float(x), float(y)) for x,y in ring]

def is_convex(poly: SPolygon) -> bool:
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n < 4:  # triangle always convex
        return True
    sign = 0
    for i in range(n):
        x1,y1 = coords[i]
        x2,y2 = coords[(i+1)%n]
        x3,y3 = coords[(i+2)%n]
        cross = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
        if abs(cross) < 1e-12:
            continue
        if sign == 0:
            sign = 1 if cross > 0 else -1
        elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
            return False
    return True

def quantize(pt, q=1e-8):
    return (round(pt[0]/q)*q, round(pt[1]/q)*q)

# ----------------------------
# Triangle merge into convex parts
# ----------------------------
def build_adjacency(polys: List[SPolygon]) -> Dict[int, List[int]]:
    # Two polygons are adjacent if they share an entire edge (two equal vertices)
    edge_map: Dict[Tuple[Tuple[float,float], Tuple[float,float]], List[int]] = defaultdict(list)
    for i, p in enumerate(polys):
        coords = list(p.exterior.coords)
        for a, b in zip(coords, coords[1:]):
            qa, qb = quantize(a), quantize(b)
            key = (qa, qb) if qa <= qb else (qb, qa)
            edge_map[key].append(i)
    adj = defaultdict(list)
    for _, ids in edge_map.items():
        if len(ids) == 2:
            i, j = ids
            adj[i].append(j)
            adj[j].append(i)
    return adj

def greedy_convex_merge(tris: List[SPolygon]) -> List[SPolygon]:
    polys = tris[:]
    changed = True
    while changed:
        changed = False
        adj = build_adjacency(polys)
        merged = set()
        new_polys: List[SPolygon] = []
        used = [False]*len(polys)

        for i in range(len(polys)):
            if used[i]: 
                continue
            best_merge = None
            for j in adj.get(i, []):
                if used[j] or i == j:
                    continue
                u = polys[i].union(polys[j])
                if u.geom_type == 'Polygon' and u.is_valid and is_convex(u):
                    # Prefer the largest area gain to reduce piece count quickly
                    area_gain = u.area - polys[i].area
                    best_merge = (area_gain, j, u)
            if best_merge:
                _, j, u = best_merge
                used[i] = used[j] = True
                new_polys.append(u)
                merged.add(i); merged.add(j)
                changed = True
            else:
                # keep as-is
                if not used[i]:
                    used[i] = True
                    new_polys.append(polys[i])
        polys = new_polys
    return polys

# ----------------------------
# SVG element extraction
# ----------------------------
def element_to_paths(elem) -> List[Tuple[Path, Tuple[float,float,float,float,float,float]]]:
    tag = etree.QName(elem.tag).localname.lower()
    tfm = parse_transform(elem.get('transform', ''))
    paths = []
    if tag == 'path':
        d = elem.get('d', '')
        if not d.strip():
            return paths
        try:
            p = parse_path(d)
            paths.append((p, tfm))
        except Exception:
            pass
    elif tag == 'rect':
        try:
            x = float(elem.get('x', 0)); y = float(elem.get('y', 0))
            w = float(elem.get('width', 0)); h = float(elem.get('height', 0))
            rx = float(elem.get('rx', 0) or 0); ry = float(elem.get('ry', 0) or 0)
            if rx == 0 and ry == 0:
                d = f"M{x},{y} h{w} v{h} h{-w} Z"
            else:
                # rounded rect: approximate with path arcs (simple)
                rxx, ryy = rx or ry, ry or rx
                rxx = min(rxx, w/2); ryy = min(ryy, h/2)
                d = (
                    f"M{x+rxx},{y} "
                    f"H{x+w-rxx} A{rxx},{ryy} 0 0 1 {x+w},{y+ryy} "
                    f"V{y+h-ryy} A{rxx},{ryy} 0 0 1 {x+w-rxx},{y+h} "
                    f"H{x+rxx} A{rxx},{ryy} 0 0 1 {x},{y+h-ryy} "
                    f"V{y+ryy} A{rxx},{ryy} 0 0 1 {x+rxx},{y} Z"
                )
            paths.append((parse_path(d), tfm))
        except Exception:
            pass
    elif tag in ('polygon','polyline'):
        pts = elem.get('points','')
        import re
        nums = [float(x) for x in re.split(r'[, \t\r\n]+', pts.strip()) if x]
        coords = list(zip(nums[::2], nums[1::2]))
        d = "M" + " L".join(f"{x},{y}" for x,y in coords)
        if tag == 'polygon':
            d += " Z"
        paths.append((parse_path(d), tfm))
    elif tag in ('circle','ellipse'):
        try:
            if tag == 'circle':
                cx = float(elem.get('cx', 0)); cy = float(elem.get('cy', 0)); r = float(elem.get('r', 0))
                rx = ry = r
            else:
                cx = float(elem.get('cx', 0)); cy = float(elem.get('cy', 0))
                rx = float(elem.get('rx', 0)); ry = float(elem.get('ry', 0))
            # approximate by path with arcs
            d = f"M {cx-rx},{cy} " \
                f"A {rx},{ry} 0 1 0 {cx+rx},{cy} " \
                f"A {rx},{ry} 0 1 0 {cx-rx},{cy} Z"
            paths.append((parse_path(d), tfm))
        except Exception:
            pass
    return paths

def svg_to_unioned_geometry(svg_path: str, max_seg_len: float, simplify_tol: float):
    tree = etree.parse(svg_path)
    root = tree.getroot()

    all_polys: List[SPolygon] = []

    for elem in root.iter():
        tag = etree.QName(elem.tag).localname.lower()
        if tag not in ('path','rect','polygon','polyline','circle','ellipse'):
            continue
        paths = element_to_paths(elem)
        for p, tfm in paths:
            subpaths = adaptive_sample_path(p, max_seg_len=max_seg_len)
            if not subpaths:
                continue
            for sp in subpaths:
                if len(sp) < 3: 
                    continue
                sp_t = apply_transform(sp, tfm)
                poly = coords_to_shapely_polygon(sp_t)
                if poly is not None and poly.area > 0:
                    all_polys.append(poly)

    if not all_polys:
        return None

    unioned = unary_union(all_polys)
    # Simplify topology-preserving
    unioned = unioned.simplify(simplify_tol, preserve_topology=True)
    # Ensure polygons only
    if isinstance(unioned, SPolygon):
        return unioned
    elif isinstance(unioned, MultiPolygon):
        return unioned
    else:
        # sometimes union can return GeometryCollection; filter polygons
        polys = [g for g in getattr(unioned, 'geoms', []) if isinstance(g, SPolygon)]
        return MultiPolygon(polys) if len(polys) > 1 else (polys[0] if polys else None)

# ----------------------------
# Main pipeline
# ----------------------------
def decompose_to_convex(unioned_geom, target_edge: float) -> List[SPolygon]:
    """Triangulate then greedily merge into convex polygons."""
    polys: List[SPolygon] = []
    if isinstance(unioned_geom, SPolygon):
        targets = [unioned_geom]
    else:
        targets = list(unioned_geom.geoms)

    for poly in targets:
        # Shapely triangulate does constrained triangulation inside polygon (incl. holes).
        tris = [t for t in triangulate(poly) if poly.contains(t.representative_point())]
        merged = greedy_convex_merge(tris)
        polys.extend(merged)

    # Optional small-piece cleanup (merge tiny slivers to neighbors if convex)
    # Skipped for simplicity; could be added if needed.

    return polys

def save_json(polys: List[SPolygon], out_json: str):
    out = []
    for p in polys:
        out.append(polygon_to_coords(p))
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

def save_debug_svg(polys: List[SPolygon], out_svg: str, stroke_width=1.0):
    # Compute bounds
    allx = []
    ally = []
    for p in polys:
        xs, ys = zip(*polygon_to_coords(p))
        allx.extend(xs); ally.extend(ys)
    if not allx:
        allx = [0,100]; ally = [0,100]
    minx, maxx = min(allx), max(allx)
    miny, maxy = min(ally), max(ally)
    w = max(1.0, maxx - minx)
    h = max(1.0, maxy - miny)

    # SVG with each convex polygon as <polygon>
    root = etree.Element('svg', xmlns="http://www.w3.org/2000/svg",
                         width=str(w), height=str(h),
                         viewBox=f"{minx} {miny} {w} {h}")
    for p in polys:
        pts = polygon_to_coords(p)
        pts_str = " ".join(f"{x},{y}" for x,y in pts)
        poly_el = etree.SubElement(root, 'polygon', points=pts_str,
                                   fill="none", stroke="black", 
                                   **{'stroke-width': str(stroke_width)})
    etree.ElementTree(root).write(out_svg, pretty_print=True, xml_declaration=True, encoding='utf-8')

def main():
    ap = argparse.ArgumentParser(description="Approximate an SVG with a set of convex polygons.")
    ap.add_argument("input_svg")
    ap.add_argument("--max-seg-len", type=float, default=2.5,
                    help="Max chord length when sampling curves (smaller = more accurate).")
    ap.add_argument("--simplify", type=float, default=0.5,
                    help="Topology-preserving simplification tolerance (SVG units).")
    ap.add_argument("--out-json", default="convex_polygons.json",
                    help="Output JSON file with [[[x,y],...], ...].")
    ap.add_argument("--debug-svg", default="convex_overlay.svg",
                    help="Optional SVG overlay drawing the convex pieces.")
    args = ap.parse_args()

    unioned = svg_to_unioned_geometry(args.input_svg, args.max_seg_len, args.simplify)
    if unioned is None or unioned.is_empty:
        print("No polygons extracted from SVG.")
        return

    convex_polys = decompose_to_convex(unioned, target_edge=args.max_seg_len)
    save_json(convex_polys, args.out_json)
    if args.debug_svg:
        save_debug_svg(convex_polys, args.debug_svg)

    total = len(convex_polys)
    area_sum = sum(p.area for p in convex_polys)
    print(f"Convex pieces: {total}, total area: {area_sum:.3f}")
    print(f"Wrote: {args.out_json}")
    if args.debug_svg:
        print(f"Wrote: {args.debug_svg}")

if __name__ == "__main__":
    main()
