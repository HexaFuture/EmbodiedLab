"""EmbodiedLab-owned Isaac geometry adapter, not a benchmark observation/evaluator.

Read exclusively on the simulation thread. No target choice, experiment logic,
parameter truth, evidence or score is provided. This reference system uses
simulator collision geometry; report this tool condition when comparing it.
"""
import itertools
import numpy as np


def sphere_world_corners(radius, transform):
    """Exact enclosing world box of an affine sphere, independent of spin."""
    transform=np.asarray(transform,float)
    centre=transform[:3,3]
    half=float(radius)*np.linalg.norm(transform[:3,:3],axis=1)
    return np.array(list(itertools.product(*zip(centre-half,centre+half))))


class NativeGeometry:
    def __init__(self, backend):
        self.backend = backend
        from collections import deque
        self.probe_contacts = deque(maxlen=18000)
        self.support_samples = deque(maxlen=round(1/backend.dt/30) if hasattr(backend, 'dt') else 8)
        self.probe_contact_count = 0
        self.scene = getattr(backend, 'scene', backend)

    def after_step(self, tick):
        if getattr(self.backend, 'held', None) is not None:
            sensor = self.backend.support_sensors[self.backend.held]
            force = np.asarray(sensor.get_contact_force_matrix(dt=self.backend.dt), float)
            if not np.isfinite(force).all():
                raise RuntimeError('invalid contact sensor')
            self.support_samples.append((self.backend.held, float(force[0, 0, 2])))
        else:
            self.support_samples.clear()
        counts = []
        for sensor in getattr(self.scene, 'probe_contacts', []):
            value = sensor.get_contact_force_data(dt=self.scene.dt)
            count = int(np.asarray(value[4]).sum())
            if count < 0 or count >= 512:
                raise RuntimeError('invalid or full probe contact buffer')
            counts.append(count)
        self.probe_contact_count = sum(counts)
        if self.probe_contact_count:
            self.probe_contacts.append(int(tick))

    def snapshot(self, contact_only=False):
        from pxr import Usd, UsdGeom, UsdPhysics
        backend, scene = self.backend, self.scene
        stage = scene.stage
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
        held = None
        if getattr(backend, 'held', None) is not None:
            held = backend.paths[backend.held]
        elif backend.robot_state().get('magnet_attached'):
            held = scene.tool['body_path']
        boxes = []
        static_boxes = []
        oriented = []      # centre, size and axes of every collider in its own frame (benches with tilting parts)
        # Contact-controlled descent needs only the held footprint and support
        # sensors, not an O(N^2) rebuild of every planning obstacle each tick.
        for prim in (() if contact_only else Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())):
            path = str(prim.GetPath())
            if path == '/World/Franka' or path.startswith('/World/Franka/'):
                continue
            if held and (path == held or path.startswith(held + '/')):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = np.array(extent.GetMin()), np.array(extent.GetMax())
            if not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
                raise RuntimeError('invalid simulator collider bound')
            box = {'lo': lo.tolist(), 'hi': hi.tolist()}
            boxes.append(box)
            if getattr(backend, 'oriented_planning_boxes', False):
                bound = cache.ComputeWorldBound(prim)
                local, matrix = bound.GetRange(), np.asarray(bound.GetMatrix()).T
                scale = np.linalg.norm(matrix[:3, :3], axis=0)
                middle = (np.array(local.GetMin()) + np.array(local.GetMax())) / 2
                oriented.append({'centre': (matrix @ np.r_[middle, 1.])[:3].tolist(),
                                 'size': ((np.array(local.GetMax()) - np.array(local.GetMin())) * scale).tolist(),
                                 'axes': (matrix[:3, :3] / scale).tolist()})
            # Fixed apparatus is an obstacle even during an intentional contact
            # skill. Classify physical mobility only, never object/task identity.
            ancestor = prim
            dynamic = False
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = UsdPhysics.RigidBodyAPI(ancestor)
                    dynamic = (bool(body.GetRigidBodyEnabledAttr().Get()) and
                               not bool(body.GetKinematicEnabledAttr().Get()))
                    break
                ancestor = ancestor.GetParent()
            if not dynamic:
                static_boxes.append(box)
        retained = []
        for i, box in enumerate(boxes):
            lo, hi = np.array(box['lo']), np.array(box['hi'])
            if not any(j != i and np.all(np.array(other['lo']) <= lo)
                       and np.all(np.array(other['hi']) >= hi)
                       and (j < i or np.any(np.array(other['lo']) < lo) or np.any(np.array(other['hi']) > hi))
                       for j, other in enumerate(boxes)):
                retained.append(box)
        if not contact_only and not 1 <= len(retained) <= 128:
            raise RuntimeError('collision geometry exceeds planner capacity')
        result = {'world_boxes': retained, 'static_world_boxes': static_boxes,
                  'payload_corners_world': None,
                  'physics_tick': int(getattr(backend, 'tick_index', getattr(scene, 'tick_index', 0))),
                  'source': 'EmbodiedLab native simulator collision adapter',
                  'probe_contact_count': self.probe_contact_count, 'probe_contact_events': list(self.probe_contacts)}
        if oriented:
            result['oriented_boxes'] = oriented
        if held:
            result['world_from_payload'] = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(held))).T.tolist()
            bound = cache.ComputeWorldBound(stage.GetPrimAtPath(held))
            extent = bound.GetRange()
            corners = np.array(list(itertools.product(*zip(extent.GetMin(), extent.GetMax()))))
            transform = np.asarray(bound.GetMatrix()).T
            result['payload_corners_world'] = ((transform @ np.c_[corners, np.ones(8)].T).T[:, :3]).tolist()
            prim=stage.GetPrimAtPath(held)
            if prim.IsA(UsdGeom.Sphere):
                # Rotating a sphere's local cube extent inflates its apparent
                # radius after recovery, although the physical sphere is the
                # same. Placement must use the sphere's actual world bounds.
                body=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
                result['payload_corners_world']=sphere_world_corners(UsdGeom.Sphere(prim).GetRadiusAttr().Get(),body).tolist()
        # Basic contact instrumentation for the SDK's support-controlled release;
        # this is instantaneous contact, not the benchmark's success predicate.
        if getattr(backend, 'held', None) is not None:
            sensor = backend.support_sensors[backend.held]
            force = np.asarray(sensor.get_contact_force_matrix(dt=backend.dt), float)
            if not np.isfinite(force).all():
                raise RuntimeError('invalid contact sensor')
            result['held_support_force_n'] = abs(float(force[0, 0, 2]))
            # list copies the deque in one C call; iterating it directly fails with "deque mutated during
            # iteration" when a physics tick appends a sample at the same time.
            samples = [v for key, v in list(self.support_samples) if key == backend.held]
            result['held_support_force_mean_n'] = float(np.mean(samples)) if samples else None
        else:
            result['held_support_force_n'] = None
        return result
