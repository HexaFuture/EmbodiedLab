"""Colliders that match what is drawn, for every fixed instrument and scene part of a bench.

Every bench is built from many visual parts; some carried no collider, so the arm or a held part could move through
a drawn housing, frame, stand, cable or panel. complete runs once on the composed stage, before the simulation
starts, and gives each such part a collider of exactly its drawn shape: the visible gprim itself becomes the collider
(a cube, cylinder, sphere, capsule or cone collides as that primitive; a mesh as its static triangle mesh), with a
small contact offset and zero rest offset so contact happens at the drawn surface.

Scope:
  - only fixed parts (no rigid body of their own, or a part of the static scene): moving measurement bodies, the
    gripper, grasped parts and their colliders are left exactly as they are;
  - parts that are not solid are not colliders: liquids and water surfaces, light (beams, spots, glows, lamp lenses
    drawn as light), screen images and display segments drawn on a screen, text and decals (drawn thinner than
    DECAL_M on a surface);
  - a new collider never touches a body that moves on a joint (pendulum, glider, rider, turntable, cart, button
    cap, lever...): the pair is filtered, so the measurement physics of those bodies is unchanged. They still
    collide with the robot and with every collider they collided with before.
The result is written to the private log (scene_colliders.json): every prim given a collider and every filtered pair.
"""
import json
import os
import re

# Off by default until the collider pass is verified on every bench; EMBODIEDLAB_SCENE_COLLIDERS=1 turns it on.
ENABLED = os.environ.get('EMBODIEDLAB_SCENE_COLLIDERS', '0') == '1'

ROBOT = '/World/Franka'
CONTACT_OFFSET_M = 0.002
REST_OFFSET_M = 0.0
DECAL_M = 0.0012            # drawn thinner than this: text, labels, segments, film on a surface
SUPPORT_M = 0.005           # a thin part within this of a fixed collider lies on that surface (a decal); a thin part
                            # that does not (a label overhanging an edge, a strip on two posts) collides as drawn
# Not solid: what is drawn is liquid, light, an image or an indicator drawn on a surface.
NOT_SOLID = re.compile(r'(liquid|^waterline\d+$|^water(surface|level)\d*$|meniscus|(laser|light)(beam|line|spot|cone|patch)|'
                       r'glow|halo|fringe|pattern|image|shadow|segment|^seg\d|pixel|caret|cursor|highlight|reflection)',
                       re.IGNORECASE)
SKIP_ROOTS = ('/Render', '/OmniverseKit', '/World/Looks', '/World/PhysicsMaterials')
SOLID_TYPES = ('Cube', 'Cylinder', 'Sphere', 'Capsule', 'Cone', 'Mesh')   # curves and points are drawn lines
# Fixed parts the runtime moves or reshapes (a static collider must not move): drawn springs, the catch tongues,
# the strip indicator plungers and the stirrer lamp lens that rises when lit.
ANIMATED = re.compile(r'(/HookeApparatus/Spring/|/ProjectileLab/SpringCoil$|/CatchTongue[^/]*/|/Plunger_[^/]*(/|$)|'
                      r'/StirLamp/)')
# Kinematic bodies that never move (the anchor of a joint): their drawn parts get colliders too.
FIXED_KINEMATIC = ('/World/HookeApparatus/Stand',)


def _owner(prim):
    from pxr import UsdPhysics
    anc = prim
    while anc and not anc.IsPseudoRoot():
        if anc.HasAPI(UsdPhysics.RigidBodyAPI):
            return anc
        if anc.HasAPI(UsdPhysics.ArticulationRootAPI):
            return anc
        anc = anc.GetParent()
    return None


def _thin(prim, cache):
    bound = cache.ComputeWorldBound(prim)
    rng = bound.GetRange()
    if rng.IsEmpty():
        return True
    import numpy as np
    m = np.asarray(bound.GetMatrix()).T
    size = (np.array(rng.GetMax()) - np.array(rng.GetMin())) * np.linalg.norm(m[:3, :3], axis=0)
    return float(min(size)) < DECAL_M


def _jointed_bodies(stage):
    """Bodies that move on a joint (measurement bodies, controls): every non-robot dynamic body of a non-fixed
    joint."""
    from pxr import UsdPhysics
    out = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.FixedJoint):
            continue
        path = str(prim.GetPath())
        if path.startswith(ROBOT):
            continue
        joint = UsdPhysics.Joint(prim)
        for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
            for target in rel.GetTargets():
                t = str(target)
                body = stage.GetPrimAtPath(t)
                if not t or t.startswith(ROBOT) or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                if UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get():
                    continue                              # a fixed anchor (a stand), not a moving body
                out.add(t)
    return sorted(out)


def candidates(stage, thin=False):
    """Visible fixed gprims without a collider that complete would make colliders (thin=True: the thin ones)."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    out = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path.startswith(ROBOT) or path.startswith(SKIP_ROOTS) or prim.GetTypeName() not in SOLID_TYPES:
            continue
        if prim.IsInstanceProxy() or ANIMATED.search(path):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):      # deliberately disabled by its builder: left as authored
            continue
        if prim.GetName() == 'Visual' and any(
                c.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(c).GetCollisionEnabledAttr().Get() is not False
                for c in prim.GetParent().GetChildren()):
            continue                                  # a part drawn as Visual with its own exact Collider beside it
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() not in ('default', 'render'):
            continue
        owner = _owner(prim)
        if owner is not None and str(owner.GetPath()) not in FIXED_KINEMATIC:   # moving body: out of scope
            continue
        if any(NOT_SOLID.search(part) for part in path.split('/')[2:]):
            continue
        if _thin(prim, cache) is not thin:
            continue
        out.append(prim)
    return out


def _fixed_collider_ranges(stage):
    """World-aligned boxes of every enabled collider of the fixed scene (drawn or not)."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], useExtentsHint=False,
                              ignoreVisibility=True)
    out = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path.startswith(ROBOT) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        owner = _owner(prim)
        if owner is not None and str(owner.GetPath()) not in FIXED_KINEMATIC:
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not rng.IsEmpty():
            out.append(rng)
    return out


def _on_a_surface(rng, ranges):
    lo, hi = rng.GetMin(), rng.GetMax()
    return any(all(r.GetMin()[k] - SUPPORT_M <= lo[k] and hi[k] <= r.GetMax()[k] + SUPPORT_M for k in range(3))
               for r in ranges)


def _thin_collider(stage, prim):
    """A thin drawn part that lies on no surface: the part itself collides if it has a thickness; a flat mesh (text)
    gets a box of its drawn outline, DECAL_M thick about its plane."""
    from pxr import Gf, Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    bound = cache.ComputeWorldBound(prim)
    if prim.GetTypeName() != 'Mesh':
        return prim
    rng, m = bound.GetRange(), bound.GetMatrix()
    axes = [Gf.Vec3d(m.GetRow3(k)).GetLength() or 1.0 for k in range(3)]
    size = [max(rng.GetMax()[k] - rng.GetMin()[k], DECAL_M / axes[k]) for k in range(3)]
    centre = (rng.GetMin() + rng.GetMax()) / 2
    world = Gf.Matrix4d().SetScale(Gf.Vec3d(*size)) * Gf.Matrix4d().SetTranslate(centre) * m
    parent = prim.GetParent()
    to_parent = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) \
        if parent.IsA(UsdGeom.Xformable) else Gf.Matrix4d(1.0)
    box = UsdGeom.Cube.Define(stage, str(prim.GetPath()) + '_Collider')
    box.CreateSizeAttr(1.0)
    box.AddTransformOp().Set(world * to_parent.GetInverse())
    box.MakeInvisible()
    return box.GetPrim()


def complete(stage, private_dir=None):
    """Give every visible fixed part a collider of its drawn shape; returns the private record (None when off)."""
    if not ENABLED:
        return None
    from pxr import PhysxSchema, Sdf, UsdPhysics
    added = []
    jointed = _jointed_bodies(stage)
    solid = candidates(stage)
    thin = candidates(stage, thin=True)

    def _collide(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.GetTypeName() == 'Mesh':
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')   # static triangle mesh
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr(CONTACT_OFFSET_M)
        contact.CreateRestOffsetAttr(REST_OFFSET_M)
        if jointed:
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets([Sdf.Path(p) for p in jointed])
        added.append(str(prim.GetPath()))

    for prim in solid:
        _collide(prim)
    # Thin parts: decals on a surface stay as drawn; the others collide (thickest first, so a strip's ticks and
    # numbers then lie on the strip).
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    ranges = _fixed_collider_ranges(stage)
    sized = []
    for prim in thin:
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not rng.IsEmpty():
            sized.append((min(rng.GetMax() - rng.GetMin()), str(prim.GetPath()), prim, rng))
    thin_added = []
    for _, _, prim, rng in sorted(sized, key=lambda s: (-s[0], s[1])):
        if _on_a_surface(rng, ranges):
            continue
        box = _thin_collider(stage, prim)
        _collide(box)
        thin_added.append(str(box.GetPath()))
        ranges.append(cache.ComputeWorldBound(prim).ComputeAlignedRange())
    record = {'colliders_added': added, 'thin_parts_off_any_surface': thin_added,
              'filtered_against_jointed_bodies': jointed,
              'contact_offset_m': CONTACT_OFFSET_M, 'rest_offset_m': REST_OFFSET_M}
    if private_dir is not None:
        from pathlib import Path
        (Path(private_dir) / 'scene_colliders.json').write_text(json.dumps(record, indent=1))
    return record
