"""Flat printed text and seven-segment readouts (render only; no physics), in a 2D frame on a horizontal plane.

A Frame maps (u, v) - u the reading direction, v the text's up - to (x, y, z) in the parent prim's coordinates. Every
segment is its own one-quad mesh; a readout is updated by toggling visibility (an unlit segment is not drawn), so lit
segments are bright on a dark window and nothing else changes.
"""
from __future__ import annotations

import numpy as np

SEGMENT_NAMES = 'abcdefg'
GLYPHS = {'0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg', '5': 'acdfg', '6': 'acdefg', '7': 'abc',
          '8': 'abcdefg', '9': 'abcdfg', '-': 'g', ' ': '', 'O': 'abcdef', 'L': 'def', 'E': 'adefg', 'r': 'eg'}
LIT = (0.55, 1.0, 0.55)
INK = (0.02, 0.02, 0.02)
WHITE = (0.93, 0.93, 0.90)


class Frame:
    def __init__(self, origin, reading, up, z):
        self.o = np.asarray(origin, float)[:2]
        self.r = np.asarray(reading, float)[:2]
        self.up = np.asarray(up, float)[:2]
        self.z = float(z)

    def at(self, u, v, dz=0.0):
        p = self.o + u * self.r + v * self.up
        return (float(p[0]), float(p[1]), self.z + dz)


SEGMENT_GAP = 0.35          # gap between neighbouring segments, in strokes (separate bars, as on a real display)


def segment_rects(u0, v0, w, h, t):
    """Segment rectangles (u0, v0, u1, v1) of a digit cell with its lower-left corner at (u0, v0): three horizontal
    bars and four vertical ones, separated by small gaps."""
    j = SEGMENT_GAP * t
    return {'a': (u0 + j, v0 + h - t, u0 + w - j, v0 + h), 'd': (u0 + j, v0, u0 + w - j, v0 + t),
            'g': (u0 + j, v0 + h / 2 - t / 2, u0 + w - j, v0 + h / 2 + t / 2),
            'f': (u0, v0 + h / 2 + t / 2 + j, u0 + t, v0 + h - t - j),
            'b': (u0 + w - t, v0 + h / 2 + t / 2 + j, u0 + w, v0 + h - t - j),
            'e': (u0, v0 + t + j, u0 + t, v0 + h / 2 - t / 2 - j),
            'c': (u0 + w - t, v0 + t + j, u0 + w, v0 + h / 2 - t / 2 - j)}


def segment_centres(u0, v0, w, h, t):
    return {k: ((a + c) / 2, (b + d) / 2) for k, (a, b, c, d) in segment_rects(u0, v0, w, h, t).items()}


def quad_mesh(stage, path, frame, rects, colour, dz=0.0, visible=True, material=None):
    """One mesh of rectangles (u0, v0, u1, v1) in `frame`; `material` (UsdShade) is bound when given."""
    from pxr import Gf, UsdGeom, UsdShade
    points, counts, indices = [], [], []
    for u0, v0, u1, v1 in rects:
        base = len(points)
        for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            points.append(Gf.Vec3f(*frame.at(u, v, dz)))
        counts.append(4)
        indices += [base, base + 1, base + 2, base + 3]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    arr = np.asarray([[p[0], p[1], p[2]] for p in points])
    mesh.CreateExtentAttr([Gf.Vec3f(*arr.min(0).tolist()), Gf.Vec3f(*arr.max(0).tolist())])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if not visible:
        mesh.MakeInvisible()
    return mesh


def text_rects(text, height):
    """Pixel-font rectangles (u0, v0, u1, v1) of a one-line label, origin at its left end, v centred; and its width."""
    import cv2
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    pad = 2
    image = np.zeros((h + base + 2 * pad, w + 2 * pad), np.uint8)
    cv2.putText(image, text, (pad, pad + h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_8)
    pixel = height / h
    rows = image > 127
    rects = []
    for r in range(rows.shape[0]):
        c = 0
        while c < rows.shape[1]:
            if rows[r, c]:
                start = c
                while c < rows.shape[1] and rows[r, c]:
                    c += 1
                top = (pad + h - r) * pixel - height / 2
                rects.append(((start - pad) * pixel, top - pixel, (c - pad) * pixel, top))
            else:
                c += 1
    return rects, w * pixel


def text(stage, path, label, frame, u, v, height, colour, align='left', dz=0.0004, visible=True, material=None):
    rects, width = text_rects(label, height)
    shift = {'left': 0.0, 'centre': -width / 2, 'right': -width}[align] + u
    return quad_mesh(stage, path, frame, [(a + shift, b + v, c + shift, d + v) for a, b, c, d in rects], colour,
                     dz, visible, material)


class Readout:
    """Sign cell, digit cells and decimal points; `show(sign, digits, point)` lights the matching segments."""

    def __init__(self, stage, path, frame, *, u0, v0, digit_w, digit_h, stroke, pitch, count, sign=True,
                 points=(), colour=LIT, dz=0.0008, material=None):
        from pxr import UsdGeom
        UsdGeom.Xform.Define(stage, path)
        self.stage, self.path, self.count = stage, path, count
        self.cells = []
        self.prims = {}
        if sign:
            g = segment_rects(u0, v0, digit_w * 0.7, digit_h, stroke)['g']
            self.prims['sign'] = quad_mesh(stage, f'{path}/Sign', frame, [g], colour, dz,
                                           material=material).GetPrim().GetPath()
            u0 += digit_w * 0.7 + (pitch - digit_w)
        for k in range(count):
            cu = u0 + k * pitch
            self.cells.append((cu, v0))
            for name, rect in segment_rects(cu, v0, digit_w, digit_h, stroke).items():
                self.prims[(k, name)] = quad_mesh(stage, f'{path}/D{k}{name}', frame, [rect], colour,
                                                  dz, material=material).GetPrim().GetPath()
        p = stroke * 1.4
        self.point_centres = {}
        for k in points:
            cu = self.cells[k][0] + digit_w + (pitch - digit_w) / 2
            self.point_centres[k] = (cu, v0 + p / 2)
            self.prims[('dp', k)] = quad_mesh(stage, f'{path}/P{k}', frame,
                                              [(cu - p / 2, v0, cu + p / 2, v0 + p)], colour, dz,
                                              material=material).GetPrim().GetPath()
        self.geometry = dict(digit_w=digit_w, digit_h=digit_h, stroke=stroke, pitch=pitch)
        self.state = None

    def show(self, sign, digits, point):
        from pxr import UsdGeom
        state = (sign, digits, point)
        if state == self.state:
            return
        lit = set()
        if sign == '-':
            lit.add('sign')
        for k, ch in enumerate(digits.rjust(self.count)[:self.count]):
            for s in GLYPHS.get(ch, ''):
                lit.add((k, s))
        if point is not None:
            lit.add(('dp', point))
        for key, prim_path in self.prims.items():
            prim = UsdGeom.Imageable(self.stage.GetPrimAtPath(prim_path))
            if key in lit:
                prim.MakeVisible()
            else:
                prim.MakeInvisible()
        self.state = state


def set_visible(stage, path, on):
    from pxr import UsdGeom
    prim = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    prim.MakeVisible() if on else prim.MakeInvisible()
