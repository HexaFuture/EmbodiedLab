"""Render-only looks for the newer benches (truss_bridge first): procedural meshes with normals and UVs,
UsdPreviewSurface materials and the PIL textures they sample.

Nothing here collides, has mass or moves by itself. A look is authored at build time beside the collider it dresses;
the collider is made invisible and keeps colliding (PhysX ignores visibility, and the planner's adapter measures each
collider's own bound, which visibility does not change). Bounds of a body (BBoxCache skips invisible prims) then come
from its looks, so a look stays within the collider it dresses (tests check 1 mm).

Meshes: Geo holds points and polygon faces (optionally face-varying normals); define() writes a UsdGeom.Mesh with
face-varying normals (crease-angle smoothed unless given) and planar-projected UVs (primvars:st, metres / tile).
Textures: written once per texture directory (EMBODIEDLAB_LOOKS_DIR, config['looks_dir'] or the temp directory),
greyscale where a material tints them (UsdUVTexture scale = the material colour).
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

TEXTURE_VERSION = 'v1'
TEXTURE_PX = 512


# --- geometry ------------------------------------------------------------------------------------------------------
class Geo:
    """Points (n, 3), faces (index tuples, counter-clockwise seen from outside), optional face-varying normals."""

    def __init__(self, points, faces, normals=None):
        self.P = np.asarray(points, float).reshape(-1, 3)
        self.faces = [tuple(int(i) for i in f) for f in faces]
        self.N = None if normals is None else np.asarray(normals, float).reshape(-1, 3)

    def moved(self, rotation=None, translate=(0.0, 0.0, 0.0)):
        R = np.eye(3) if rotation is None else np.asarray(rotation, float)
        return Geo(self.P @ R.T + np.asarray(translate, float), self.faces, None if self.N is None else self.N @ R.T)


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-15, n, 1.0)


def _padded(faces):
    """Faces as an (n, w) index array, short faces padded with their last index; and the corner counts."""
    counts = np.array([len(f) for f in faces])
    F = np.empty((len(faces), counts.max()), int)
    for k, f in enumerate(faces):
        F[k, :len(f)] = f
        F[k, len(f):] = f[-1]
    return F, counts


def _face_normals(P, faces):
    """Area-weighted (Newell) normal of each face (a padded corner adds nothing)."""
    F, _ = _padded(faces)
    p, q = P[F], P[np.roll(F, -1, axis=1)]
    return 0.5 * np.stack([np.sum((p[..., 1] - q[..., 1]) * (p[..., 2] + q[..., 2]), axis=1),
                           np.sum((p[..., 2] - q[..., 2]) * (p[..., 0] + q[..., 0]), axis=1),
                           np.sum((p[..., 0] - q[..., 0]) * (p[..., 1] + q[..., 1]), axis=1)], axis=1)


def _corners(faces):
    counts = np.array([len(f) for f in faces])
    return np.repeat(np.arange(len(faces)), counts), np.fromiter((i for f in faces for i in f), int, counts.sum())


def smooth_normals(geo, crease_deg=35.0):
    """Face-varying normals: each corner averages the faces at its point within crease_deg of its own face."""
    if geo.N is not None:
        return geo.N
    fn = _face_normals(geo.P, geo.faces)
    un = _unit(fn)
    cf, cv = _corners(geo.faces)
    order = np.argsort(cv, kind='stable')
    sv, sf = cv[order], cf[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sv)) + 1]
    sizes = np.diff(np.r_[starts, len(sv)])
    c = math.cos(math.radians(crease_deg))
    acc = np.zeros((len(sv), 3))
    pos = np.arange(len(sv))
    for d in range(-int(sizes.max()) + 1, int(sizes.max())):
        j = pos + d
        ok = (j >= 0) & (j < len(sv))
        ok[ok] &= sv[j[ok]] == sv[ok]
        i, j = pos[ok], j[ok]
        near = np.sum(un[sf[j]] * un[sf[i]], axis=1) >= c
        np.add.at(acc, i[near], fn[sf[j[near]]])
    out = np.empty_like(acc)
    out[order] = acc
    return _unit(out)


def planar_uv(geo, normals, tile=0.05):
    """Face-varying UVs: each face projected along its dominant normal axis, in metres / tile."""
    fn = _face_normals(geo.P, geo.faces)
    cf, cv = _corners(geo.faces)
    axis = np.argmax(np.abs(fn), axis=1)[cf]
    a = np.choose(axis, [1, 0, 0])
    b = np.choose(axis, [2, 2, 1])
    pts = geo.P[cv]
    return np.stack([pts[np.arange(len(cv)), a], pts[np.arange(len(cv)), b]], axis=1) / tile


def _drop_degenerate(geo, eps=1e-14):
    """The faces with an area (zero-area faces would get no normal)."""
    keep = np.linalg.norm(_face_normals(geo.P, geo.faces), axis=1) > eps
    if keep.all():
        return geo
    N = None
    if geo.N is not None:
        counts = np.array([len(f) for f in geo.faces])
        N = geo.N[np.repeat(keep, counts)]
    return Geo(geo.P, [f for f, k in zip(geo.faces, keep) if k], N)


def baked(geo, crease_deg=35.0):
    """geo without zero-area faces, its normals computed once (to define many times)."""
    geo = _drop_degenerate(geo)
    return Geo(geo.P, geo.faces, smooth_normals(geo, crease_deg))


def merge(*geos):
    """One Geo of several (each keeps its own normals; computed where missing)."""
    P, faces, N, base = [], [], [], 0
    for g in geos:
        P.append(g.P)
        faces += [tuple(i + base for i in f) for f in g.faces]
        N.append(smooth_normals(g))
        base += len(g.P)
    return Geo(np.vstack(P), faces, np.vstack(N))


def define(stage, path, geo, material=None, *, tile=0.05, crease_deg=35.0, translate=None, rotate_xyz=None, uv=None):
    """A render mesh at path from geo (local frame), bound to material."""
    if uv is None:
        geo = _drop_degenerate(geo)
    normals = smooth_normals(geo, crease_deg)
    st = planar_uv(geo, normals, tile) if uv is None else np.asarray(uv, float)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in geo.P.tolist()])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in geo.faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in geo.faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals.tolist()])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    primvar.Set([Gf.Vec2f(*t) for t in st.tolist()])
    lo, hi = geo.P.min(axis=0), geo.P.max(axis=0)
    mesh.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    if translate is not None or rotate_xyz is not None:
        api = UsdGeom.XformCommonAPI(mesh)
        if translate is not None:
            api.SetTranslate(Gf.Vec3d(*translate))
        if rotate_xyz is not None:
            api.SetRotate(Gf.Vec3f(*rotate_xyz), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


# --- shapes ----------------------------------------------------------------------------------------------------------
def _axis_samples(h, r, arc):
    """Coordinates along one axis of a rounded box: the rounds' arc samples at both ends."""
    inner = h - r
    if r <= 0:
        return [-h, h]
    a = [inner + r * math.sin(math.pi / 2 * k / arc) for k in range(arc + 1)]
    return sorted({-v for v in a} | set(a))


def rounded_box(size, radius, arc=3):
    """Box of size (x, y, z) about the origin, edges and corners rounded to radius (analytic normals)."""
    h = np.asarray(size, float) / 2
    r = float(min(radius, *(h * 0.999)))
    inner = h - r
    P, faces, N = [], [], []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        ua, ub = _axis_samples(h[a], r, arc), _axis_samples(h[b], r, arc)
        for sign in (1.0, -1.0):
            base = len(P)
            for va in ua:
                for vb in ub:
                    v = np.zeros(3)
                    v[axis], v[a], v[b] = sign * h[axis], va, vb
                    core = np.clip(v, -inner, inner)
                    d = v - core
                    n = d / np.linalg.norm(d)
                    P.append(core + r * n)
            na, nb = len(ua), len(ub)
            for i in range(na - 1):
                for j in range(nb - 1):
                    q = [base + i * nb + j, base + (i + 1) * nb + j, base + (i + 1) * nb + j + 1, base + i * nb + j + 1]
                    # (axis, a, b) right-handed for axis 0 and 2, so +sign keeps this order outward; axis 1 flips
                    if (sign > 0) != (axis == 1):
                        faces.append(tuple(q))
                    else:
                        faces.append(tuple(reversed(q)))
    P = np.array(P)
    core = np.clip(P, -inner, inner)
    normals = _unit(P - core)
    N = [normals[i] for f in faces for i in f]
    return Geo(P, faces, N)


def _signed_area(poly):
    p = np.asarray(poly, float)
    q = np.roll(p, -1, axis=0)
    return 0.5 * float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))


def earclip(poly):
    """Triangles (index triples, counter-clockwise) of a simple polygon."""
    p = np.asarray(poly, float)
    idx = list(range(len(p)))
    if _signed_area(p) < 0:
        idx.reverse()
    tris = []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

    def inside(pt, a, b, c):
        return cross(a, b, pt) >= -1e-14 and cross(b, c, pt) >= -1e-14 and cross(c, a, pt) >= -1e-14

    guard = 0
    while len(idx) > 3 and guard < 4 * len(p) ** 2:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            c = cross(p[i0], p[i1], p[i2])
            if c < -1e-14:
                continue
            if c <= 1e-14:                      # collinear: drop the middle point, no triangle
                idx.pop(k)
                break
            if any(inside(p[j], p[i0], p[i1], p[i2]) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise ValueError('polygon is not simple')
    if len(idx) == 3 and abs(cross(*(p[i] for i in idx))) > 1e-14:
        tris.append(tuple(idx))
    return tris


def prism(poly, z0, z1):
    """A simple polygon (x, y) extruded from z0 to z1 (flat faces, crease normals)."""
    p = np.asarray(poly, float)
    if _signed_area(p) < 0:
        p = p[::-1]
    n = len(p)
    P = np.vstack([np.c_[p, np.full(n, z0)], np.c_[p, np.full(n, z1)]])
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    tris = earclip(p)
    faces += [(n + a, n + b, n + c) for a, b, c in tris]
    faces += [(c, b, a) for a, b, c in tris]
    return Geo(P, faces)


def lathe(profile, segments=48, closed=False, a0=0.0, a1=2 * math.pi):
    """Profile (r, z) revolved about z. The profile runs counter-clockwise in the (r, z) plane (outer wall up, top
    inward), so faces point outward. closed: the profile is a loop; a partial sweep (a1 - a0 < 2 pi) caps both ends."""
    prof = [(float(r), float(z)) for r, z in profile]
    m = len(prof)
    full = abs((a1 - a0) - 2 * math.pi) < 1e-9
    rings = segments if full else segments + 1
    P = []
    for j in range(rings):
        a = a0 + (a1 - a0) * j / segments
        P += [(r * math.cos(a), r * math.sin(a), z) for r, z in prof]
    faces = []
    edges = [(i, i + 1) for i in range(m - 1)] + ([(m - 1, 0)] if closed else [])
    for j in range(segments):
        j1 = (j + 1) % rings
        for i, k in edges:
            q = [j * m + i, j1 * m + i, j1 * m + k, j * m + k]
            if prof[i][0] < 1e-12:
                q = [q[0], q[2], q[3]]
            elif prof[k][0] < 1e-12:
                q = [q[0], q[1], q[2]]
            faces.append(tuple(q))
    if closed and not full:
        tris = earclip(prof)
        faces += tris                                                          # at a0, facing -theta
        last = (rings - 1) * m
        faces += [(last + c, last + b, last + a) for a, b, c in tris]          # at a1, facing +theta
    return Geo(P, faces)


def tube(points, radius, sides=12, closed=False):
    """A round tube along a polyline (parallel-transported rings); open ends uncapped."""
    pts = np.asarray(points, float)
    n = len(pts)
    tangents = []
    for i in range(n):
        if closed:
            d = pts[(i + 1) % n] - pts[i - 1]
        else:
            d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        tangents.append(d / np.linalg.norm(d))
    ref = np.array([0.0, 0.0, 1.0]) if abs(tangents[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(tangents[0], ref))
    P = []
    for i in range(n):
        if i:
            u = u - tangents[i] * float(u @ tangents[i])
            u = u / np.linalg.norm(u)
        v = np.cross(tangents[i], u)
        P += [pts[i] + radius * (math.cos(2 * math.pi * k / sides) * u + math.sin(2 * math.pi * k / sides) * v)
              for k in range(sides)]
    faces = []
    for i in range(n if closed else n - 1):
        i1 = (i + 1) % n
        for k in range(sides):
            k1 = (k + 1) % sides
            faces.append((i * sides + k, i1 * sides + k, i1 * sides + k1, i * sides + k1))
    return Geo(P, faces)


def torus(major, minor, segments=32, sides=12):
    return lathe([(major + minor * math.cos(2 * math.pi * k / sides), minor * math.sin(2 * math.pi * k / sides))
                  for k in range(sides)], segments, closed=True)


def hex_prism(across_flats, height, z0=0.0):
    r = across_flats / 2 / math.cos(math.pi / 6)
    return prism([(r * math.cos(math.pi / 6 + k * math.pi / 3), r * math.sin(math.pi / 6 + k * math.pi / 3)) for k in range(6)],
                 z0, z0 + height)


def cap_screw(head_r, head_h, z0=0.0, segments=32):
    """Socket head cap screw head standing on z0 (a chamfered drum; the socket is a separate dark hexagon)."""
    c = min(head_r * 0.12, head_h * 0.3)
    return lathe([(0.0, z0), (head_r, z0), (head_r, z0 + head_h - c), (head_r - c, z0 + head_h), (0.0, z0 + head_h)],
                 segments)


def round_rect(w, h, r, arc=4):
    """Rounded rectangle outline (counter-clockwise) about the origin."""
    r = min(r, w / 2 * 0.999, h / 2 * 0.999)
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180),
                       (w / 2 - r, -h / 2 + r, 270)):
        for k in range(arc + 1):
            a = math.radians(a0 + 90 * k / arc)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def frame_ring(outer_wh, inner_wh, z0, z1, r_out=0.002, r_in=0.001):
    """A flat rectangular frame (a bezel): outer and inner rounded outlines, walls, top and bottom."""
    o = round_rect(*outer_wh, r_out)
    i = round_rect(*inner_wh, r_in)
    n = len(o)
    P = [(x, y, z0) for x, y in o] + [(x, y, z1) for x, y in o] + [(x, y, z0) for x, y in i] + [(x, y, z1) for x, y in i]
    ob, ot, ib, it = 0, n, 2 * n, 3 * n
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces += [(ob + k, ob + k1, ot + k1, ot + k), (ib + k1, ib + k, it + k, it + k1),
                  (ot + k, ot + k1, it + k1, it + k), (ob + k1, ob + k, ib + k, ib + k1)]
    return Geo(P, faces)


def tslot_profile(size=0.020, slot=0.006, neck=0.0018, cavity=0.0084, depth=0.0055, corner=0.001):
    """Outline (counter-clockwise) of a 20-series T-slot extrusion: a square with a T-slot in each side."""
    h = size / 2
    pts = []
    for side in range(4):
        R = rot_z(90 * side)[:2, :2]
        local = [(h, -h + corner), (h, -slot / 2), (h - neck, -slot / 2), (h - neck, -cavity / 2),
                 (h - depth + 0.0015, -cavity / 2), (h - depth, -slot / 2 + 0.0005), (h - depth, slot / 2 - 0.0005),
                 (h - depth + 0.0015, cavity / 2), (h - neck, cavity / 2), (h - neck, slot / 2), (h, slot / 2),
                 (h, h - corner), (h - corner * (1 - math.cos(math.pi / 4)), h - corner * (1 - math.sin(math.pi / 4)))]
        pts += [tuple(R @ np.array(p)) for p in local]
    return pts


def tslot(length, size=0.020):
    """T-slot extrusion along z from 0 to length."""
    return prism(tslot_profile(size), 0.0, length)


def eye_link(eye_x, cut_x, half_w, eye_r, hole_r, t, chamfer, n=120):
    """A flat bar body in its own frame: the strip from x = cut_x to the eye at (eye_x, 0) (half width half_w), the
    eye boss (radius eye_r) with its hole (hole_r), thickness t from z = 0. Its outline is star-shaped about the eye
    centre (union of a rectangle and a disc that both hold it), so the caps are rings about it. The top edge is
    chamfered by chamfer except along the cut face (where the other body continues the bar)."""
    sign = 1.0 if cut_x > eye_x else -1.0
    jx = math.sqrt(max(eye_r ** 2 - half_w ** 2, 0.0))
    angles = {2 * math.pi * k / n for k in range(n)}
    for s in (1, -1):
        angles.add(math.atan2(s * half_w, cut_x - eye_x) % (2 * math.pi))
        angles.add(math.atan2(s * half_w, sign * jx) % (2 * math.pi))
    angles = sorted(angles)
    lo, hi = min(cut_x, eye_x), max(cut_x, eye_x)

    def exit_rect(d):
        ts = []
        if abs(d[0]) > 1e-12:
            ts.append(((hi if d[0] > 0 else lo) - eye_x) / d[0])
        if abs(d[1]) > 1e-12:
            ts.append((half_w if d[1] > 0 else -half_w) / d[1])
        return min(t for t in ts if t >= -1e-12) if ts else 0.0

    outline, on_cut = [], []
    for a in angles:
        d = np.array([math.cos(a), math.sin(a)])
        tr = exit_rect(d)
        rho = max(tr, eye_r)
        p = np.array([eye_x, 0.0]) + rho * d
        outline.append(p)
        on_cut.append(tr > eye_r + 1e-9 and abs(p[0] - cut_x) < 1e-9)
    outline = np.array(outline)
    m = len(outline)
    # inset of the top outline: each edge moved inward by the chamfer (none on the cut face), corners mitred
    off = []
    for k in range(m):
        a, b = outline[k], outline[(k + 1) % m]
        e = b - a
        L = np.linalg.norm(e)
        nrm = np.array([e[1], -e[0]]) / L if L > 1e-12 else np.zeros(2)         # outward (counter-clockwise)
        c = 0.0 if (on_cut[k] and on_cut[(k + 1) % m]) else chamfer
        off.append((a - nrm * c, e, c))
    inset = []
    for k in range(m):
        p0, e0, _ = off[k - 1]
        p1, e1, _ = off[k]
        A = np.array([e0, -e1]).T
        if abs(np.linalg.det(A)) < 1e-14 * max(np.linalg.norm(e0) * np.linalg.norm(e1), 1e-30):
            inset.append(p1)
        else:
            s = np.linalg.solve(A, p1 - p0)
            q = p0 + s[0] * e0
            inset.append(q if np.linalg.norm(q - outline[k]) < 3 * chamfer + 1e-9 else p1)
    inset = np.array(inset)
    hole = np.array([[eye_x + hole_r * math.cos(a), hole_r * math.sin(a)] for a in angles])
    zc = t - chamfer
    rings = [np.c_[outline, np.zeros(m)], np.c_[outline, np.full(m, zc)], np.c_[inset, np.full(m, t)],
             np.c_[hole, np.full(m, t)], np.c_[hole, np.zeros(m)]]
    # the cut face has no chamfer: its inner points go straight up to the top (its corners keep the sides' chamfer,
    # so the chamfer runs on to the cut and continues on the next piece)
    rings[1][[k for k in range(m) if on_cut[k] and on_cut[k - 1] and on_cut[(k + 1) % m]], 2] = t
    P = np.vstack(rings)
    faces = []
    for r in range(4):
        for k in range(m):
            k1 = (k + 1) % m
            a, b, c, d = r * m + k, r * m + k1, (r + 1) * m + k1, (r + 1) * m + k
            if r == 2:              # top ring (inset outline -> hole): triangles, the ring can be long and thin
                faces += [(a, b, c), (a, c, d)]
            else:
                faces.append((a, b, c, d))
    for k in range(m):              # bottom ring (hole -> outline at z = 0)
        k1 = (k + 1) % m
        faces += [(4 * m + k, 4 * m + k1, k1), (4 * m + k, k1, k)]
    return Geo(P, faces)


def web(xa, xb, half_w, t, chamfer, slots=(), ends=(True, True), arc=5):
    """The strip of a flat bar from x = xa to xb (half width half_w, thickness t from z = 0, top edges chamfered),
    with through slots (a, b, half_height, corner_radius): rounded rectangles about y = 0 between x = a and b. Built
    of quads: full-width blocks between the slots, bands above and below each. ends: end faces at xa, xb."""
    wi = half_w - chamfer
    zc = t - chamfer
    loops, tops = [], []                    # each slot's outline (counter-clockwise) and its upper edge by x
    for a, b, hh, rr in slots:
        loop = [(x + (a + b) / 2, y) for x, y in round_rect(b - a, 2 * hh, rr, arc)]
        loops.append(loop)
        tops.append(sorted({(round(x, 9), y) for x, y in loop if y >= -1e-12}))
    X = sorted({xa, xb} | {x for top in tops for x, _ in top})
    P, faces = [], []

    def add(points):
        base = len(P)
        P.extend(points)
        return list(range(base, base + len(points)))

    def caps(z, w, top_face):
        """Cap at z (half width w): block polygons between the slots, bands beside them."""
        bounds = [xa] + [x for top in tops for x in (top[0][0], top[-1][0])] + [xb]
        for k in range(0, len(bounds), 2):
            x0, x1 = bounds[k], bounds[k + 1]
            left = tops[k // 2 - 1][0][1] if k else None             # the slot that ends at x0: its end half height
            right = tops[k // 2][0][1] if k // 2 < len(tops) else None
            poly = [(x0, -w), (x1, -w)]
            if right is not None:
                poly += [(x1, -right), (x1, right)]
            poly += [(x1, w), (x0, w)]
            if left is not None:
                poly += [(x0, left), (x0, -left)]
            ids = add([(x, y, z) for x, y in poly])
            faces.append(tuple(ids) if top_face else tuple(reversed(ids)))
        for top in tops:
            for (x0, y0), (x1, y1) in zip(top, top[1:]):
                for sign in (1, -1):
                    q = add([(x0, sign * y0, z), (x1, sign * y1, z), (x1, sign * w, z), (x0, sign * w, z)])
                    up = (sign > 0) == top_face
                    faces.append(tuple(q) if up else tuple(reversed(q)))

    caps(t, wi, True)
    caps(0.0, half_w, False)
    # sides: wall and chamfer along y = +-half_w, sampled at X (no T-junctions with the caps)
    for sign in (1, -1):
        rows = [add([(x, sign * half_w, 0.0) for x in X]), add([(x, sign * half_w, zc) for x in X]),
                add([(x, sign * wi, t) for x in X])]
        for r0, r1 in zip(rows, rows[1:]):
            for i in range(len(X) - 1):
                q = (r0[i], r0[i + 1], r1[i + 1], r1[i])
                faces.append(q if sign < 0 else tuple(reversed(q)))
    # the slots' walls (facing into the slot)
    for loop in loops:
        bottom, top = add([(x, y, 0.0) for x, y in loop]), add([(x, y, t) for x, y in loop])
        n = len(loop)
        for k in range(n):
            k1 = (k + 1) % n
            faces.append((bottom[k1], bottom[k], top[k], top[k1]))
    profile = [(-half_w, 0.0), (half_w, 0.0), (half_w, zc), (wi, t), (-wi, t), (-half_w, zc)]
    for x, on, sign in ((xa, ends[0], -1), (xb, ends[1], 1)):
        if on:
            ids = add([(x, y, z) for y, z in profile])
            faces.append(tuple(ids) if sign > 0 else tuple(reversed(ids)))
    return Geo(P, faces)


def split(geo, keep):
    """Two Geos of a baked geo (normals given): the faces where keep(face normal) holds and the others, each with its
    own points."""
    fn = _unit(_face_normals(geo.P, geo.faces))
    counts = np.array([len(f) for f in geo.faces])
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    out = []
    for want in (True, False):
        sel = [k for k in range(len(geo.faces)) if bool(keep(fn[k])) == want]
        used = sorted({i for k in sel for i in geo.faces[k]})
        remap = {i: j for j, i in enumerate(used)}
        N = np.vstack([geo.N[starts[k]:starts[k] + counts[k]] for k in sel]) if sel else np.zeros((0, 3))
        out.append(Geo(geo.P[used], [tuple(remap[i] for i in geo.faces[k]) for k in sel], N))
    return out


# --- materials ---------------------------------------------------------------------------------------------------------
def material(stage, path, colour, *, metallic=0.0, roughness=0.5, clearcoat=0.0, clearcoat_roughness=0.05,
             emission=0.0, diffuse=None, rough_map=None, rough_scale=1.0, normal=None, cutout=None, specular=None):
    """UsdPreviewSurface. diffuse: a greyscale texture tinted by colour; rough_map: roughness texture (its red
    channel x rough_scale); normal: a tangent-space normal map; cutout: an RGBA decal (colour from it, opacity from
    its alpha, threshold 0.5)."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(metallic))
    if specular is not None:
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*specular))
    if clearcoat:
        shader.CreateInput('clearcoat', Sdf.ValueTypeNames.Float).Set(float(clearcoat))
        shader.CreateInput('clearcoatRoughness', Sdf.ValueTypeNames.Float).Set(float(clearcoat_roughness))
    if emission:
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * emission for c in colour]))
    reader = None

    def texture(name, file, colour_space, scale=(1, 1, 1, 1), bias=(0, 0, 0, 0)):
        nonlocal reader
        if reader is None:
            reader = UsdShade.Shader.Define(stage, path + '/St')
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, f'{path}/{name}')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(colour_space)
        tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        return tex

    if cutout is not None:
        tex = texture('Decal', cutout, 'sRGB')
        tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        tex.CreateOutput('a', Sdf.ValueTypeNames.Float)
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), 'a')
        shader.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(0.5)
    elif diffuse is not None:
        tex = texture('Diffuse', diffuse, 'sRGB', scale=(*colour, 1.0))
        tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    else:
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    if rough_map is not None:
        tex = texture('Roughness', rough_map, 'raw', scale=(rough_scale,) * 4)
        tex.CreateOutput('r', Sdf.ValueTypeNames.Float)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), 'r')
    else:
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
    if normal is not None:
        tex = texture('Normal', normal, 'raw', scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0))
        tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


# --- textures (PIL) ----------------------------------------------------------------------------------------------------
def texture_dir(config=None):
    root = os.environ.get('EMBODIEDLAB_LOOKS_DIR') or (config or {}).get('looks_dir') or \
        os.path.join(tempfile.gettempdir(), 'embodiedlab_looks')
    path = os.path.join(root, TEXTURE_VERSION)
    os.makedirs(path, exist_ok=True)
    return path


def _blur(a, r, axis):
    """Box blur of radius r along axis, wrapping (keeps textures tileable)."""
    if r < 1:
        return a
    out = np.zeros_like(a)
    for k in range(-r, r + 1):
        out += np.roll(a, k, axis=axis)
    return out / (2 * r + 1)


def _smooth_noise(rng, n, r):
    a = rng.standard_normal((n, n))
    for _ in range(2):
        a = _blur(_blur(a, r, 0), r, 1)
    a -= a.min()
    return a / max(a.max(), 1e-12)


def _normal_map(height, strength):
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) / 2
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) / 2
    n = _unit(np.dstack([-gx * strength, gy * strength, np.ones_like(height)]))
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def _save(folder, name, array):
    from PIL import Image
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        tmp = path + f'.{os.getpid()}.tmp.png'
        Image.fromarray(array).save(tmp)
        os.replace(tmp, path)
    return path


def _grey(v):
    return (np.clip(v, 0, 1) * 255).astype(np.uint8)


def brushed(folder, name='brushed', streak=0.10, px=TEXTURE_PX, seed=1):
    """Brushed metal along u: (diffuse grey, roughness) paths."""
    if os.path.exists(os.path.join(folder, name + '_d.png')):
        return os.path.join(folder, name + '_d.png'), os.path.join(folder, name + '_r.png')
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((px, 1)) * 0.6 + _blur(rng.standard_normal((px, 1)), 3, 0) * 1.5
    fine = _blur(rng.standard_normal((px, px)), 24, 1)
    s = np.tanh(rows + fine * 1.2)
    d = _save(folder, name + '_d.png', _grey(0.88 + streak * s))
    r = _save(folder, name + '_r.png', _grey(0.32 + 0.10 * s))
    return d, r


def grain(folder, name, amplitude, roughness, spread, px=TEXTURE_PX, seed=2, blur=2):
    """Fine mottled grain (anodizing, black oxide, powder coat): (diffuse grey, roughness) paths."""
    if os.path.exists(os.path.join(folder, name + '_d.png')):
        return os.path.join(folder, name + '_d.png'), os.path.join(folder, name + '_r.png')
    rng = np.random.default_rng(seed)
    s = _smooth_noise(rng, px, 16) - 0.5 + 0.35 * (_blur(_blur(rng.standard_normal((px, px)), blur, 0), blur, 1))
    d = _save(folder, name + '_d.png', _grey(1.0 - amplitude + amplitude * (0.5 + s)))
    r = _save(folder, name + '_r.png', _grey(roughness + spread * s))
    return d, r


def orange_peel(folder, name='peel', px=TEXTURE_PX, seed=3, strength=6.0):
    """Normal map of a powder-coat / cast paint surface."""
    path = os.path.join(folder, name + '_n.png')
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(seed)
    h = _smooth_noise(rng, px, 3) * 0.6 + _smooth_noise(rng, px, 8) * 0.4
    return _save(folder, name + '_n.png', _normal_map(h, strength))


def spangle(folder, name='galvanized', px=TEXTURE_PX, seeds=90, seed=4):
    """Galvanized zinc spangle: Voronoi cells of differing brightness and roughness."""
    if os.path.exists(os.path.join(folder, name + '_d.png')):
        return os.path.join(folder, name + '_d.png'), os.path.join(folder, name + '_r.png')
    rng = np.random.default_rng(seed)
    pts = rng.random((seeds, 2)) * px
    yy, xx = np.mgrid[0:px, 0:px]
    best = np.full((px, px), np.inf)
    cell = np.zeros((px, px), int)
    for k, (x, y) in enumerate(pts):
        dx = np.minimum(np.abs(xx - x), px - np.abs(xx - x))
        dy = np.minimum(np.abs(yy - y), px - np.abs(yy - y))
        d = dx * dx + dy * dy
        better = d < best
        best[better], cell[better] = d[better], k
    shade = rng.random(seeds)[cell]
    feather = _blur(_blur(rng.standard_normal((px, px)), 6, 1), 1, 0) * 0.08
    d = _save(folder, name + '_d.png', _grey(0.78 + 0.16 * shade + feather))
    r = _save(folder, name + '_r.png', _grey(0.22 + 0.25 * (1 - shade)))
    return d, r


def wood(folder, name='beech', px=TEXTURE_PX, seed=5):
    """Beech: fine straight grain along u with pores and rays; (diffuse colour, roughness) paths (diffuse in colour,
    not tinted: use colour (1, 1, 1))."""
    if os.path.exists(os.path.join(folder, name + '_d.png')):
        return os.path.join(folder, name + '_d.png'), os.path.join(folder, name + '_r.png')
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:px, 0:px] / px
    warp = _smooth_noise(rng, px, 24) * 0.08
    rings = np.sin(2 * math.pi * (yy * 9 + warp * 6)) * 0.5 + 0.5
    pores = _blur(rng.random((px, px)) < 0.015, 6, 1).astype(float)
    rays = (_blur((rng.random((px, px)) < 0.002).astype(float), 1, 0) > 0.2) * 0.5
    v = 0.78 + 0.10 * rings ** 3 - 0.25 * pores - 0.04 * _smooth_noise(rng, px, 10) + 0.05 * rays
    base = np.array([0.86, 0.66, 0.45])
    rgb = np.clip(base[None, None, :] * v[..., None], 0, 1)
    d = _save(folder, name + '_d.png', (rgb * 255).astype(np.uint8))
    r = _save(folder, name + '_r.png', _grey(0.55 + 0.15 * pores))
    return d, r


def knurl(folder, name='knurl', px=TEXTURE_PX, pitch=24, strength=10.0):
    """Normal map of a diamond knurl (two crossed sets of grooves)."""
    path = os.path.join(folder, name + '_n.png')
    if os.path.exists(path):
        return path
    yy, xx = np.mgrid[0:px, 0:px]
    a = np.abs(((xx + yy) % pitch) / pitch - 0.5)
    b = np.abs(((xx - yy) % pitch) / pitch - 0.5)
    h = np.minimum(a, b)
    return _save(folder, name + '_n.png', _normal_map(h, strength))


def layer_lines(folder, name='layers', px=TEXTURE_PX, lines=64, seed=9):
    """FDM layer lines: ridges along u, `lines` per texture height (tile the UVs at lines x layer height). Returns
    (diffuse grey, normal map) paths."""
    if os.path.exists(os.path.join(folder, name + '_d.png')):
        return os.path.join(folder, name + '_d.png'), os.path.join(folder, name + '_n.png')
    rng = np.random.default_rng(seed)
    period = px // lines
    v = np.arange(px)[:, None] + 0.6 * _blur(rng.standard_normal((1, px)), 40, 1) * period   # slight wobble
    phase = (v % period) / period
    amp = 1.0 + 0.25 * np.repeat(rng.standard_normal(lines), period)[:, None]
    h = amp * np.sin(np.pi * phase) ** 0.6
    d = _save(folder, name + '_d.png', _grey(0.90 + 0.10 * h / max(h.max(), 1e-9)))
    n = _save(folder, name + '_n.png', _normal_map(h * period / 6.0, 2.5))
    return d, n


def raster(folder, name='raster', px=TEXTURE_PX, period=16, seed=10):
    """A printed top surface: faint diagonal extrusion lines (period px apart, +45 degrees). Returns (diffuse grey,
    normal map) paths."""
    if os.path.exists(os.path.join(folder, name + '_d.png')):
        return os.path.join(folder, name + '_d.png'), os.path.join(folder, name + '_n.png')
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:px, 0:px]
    phase = ((xx + yy) % period) / period
    h = np.sin(np.pi * phase) ** 0.8 + 0.08 * _blur(_blur(rng.standard_normal((px, px)), 2, 0), 2, 1)
    d = _save(folder, name + '_d.png', _grey(0.95 + 0.05 * h))
    n = _save(folder, name + '_n.png', _normal_map(h, 1.2))
    return d, n


def decal(folder, name, lines, px_h=128, ink=(20, 20, 20), background=None, pad=0.12, bold=False):
    """A text decal: lines of text centred, ink on a transparent (or given RGBA) background. Returns (path, aspect
    width / height)."""
    from PIL import Image, ImageDraw, ImageFont
    path = os.path.join(folder, name + '.png')
    font = ImageFont.load_default(size=int(px_h * (1 - 2 * pad) / max(len(lines), 1) * 0.92))
    probe = ImageDraw.Draw(Image.new('L', (8, 8)))
    widths = [probe.textlength(line, font=font) for line in lines]
    w = int(max(widths) + 2 * pad * px_h) + 1
    if not os.path.exists(path):
        img = Image.new('RGBA', (w, px_h), background or (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        step = px_h * (1 - 2 * pad) / len(lines)
        for k, line in enumerate(lines):
            y = pad * px_h + step * (k + 0.5)
            draw.text((w / 2, y), line, font=font, fill=(*ink, 255), anchor='mm',
                      stroke_width=1 if bold else 0, stroke_fill=(*ink, 255))
        tmp = path + f'.{os.getpid()}.tmp.png'
        img.save(tmp)
        os.replace(tmp, path)
    return path, w / px_h


def quad(width, height, z=0.0):
    """A flat rectangle about the origin facing +z, with 0..1 UVs (u along x, v along y)."""
    w, h = width / 2, height / 2
    geo = Geo([(-w, -h, z), (w, -h, z), (w, h, z), (-w, h, z)], [(0, 1, 2, 3)], [(0, 0, 1)] * 4)
    return geo, [(0, 0), (1, 0), (1, 1), (0, 1)]
