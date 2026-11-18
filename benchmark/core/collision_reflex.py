"""Franka-like collision reflex and private robot-contact log, shared by every bench.

External force estimate. Each physics tick PhysX reports every contact of a robot link (fingers included) with any
other body; the robot links carry a contact report with no force threshold (prepare_robot, applied before the
simulation starts). A contact's force on the link is its contact impulse over the tick. The links are grouped as a
real arm's force/torque estimate sees them: the hand and both fingers form one group (the end effector), every other
link is its own group, and the base link (panda_link0) is left out because forces on the mounted base never reach a
joint torque sensor. The estimate is the sum over groups of the magnitude of each group's vector sum. Opposing finger
contacts on one part (the squeeze of a grasp) therefore cancel inside the end-effector group, as a gripper's internal
force does on a real arm.

A grasped part is a known payload, as a Franka with a configured load: its weight and inertia reach the hand through
the rigid hold (grip_contact) and are not contacts, and its contacts with the hand and fingers are filtered while it
is held. What the payload touches does count: the contacts of the held part with every body other than the robot are
added to the end-effector group, because they are transmitted to the hand through the hold (to within the part's own
inertia, which a configured load also accounts for). Contacts between the held part and other robot links are internal
to the robot and its load and are left out. The held part carries a contact report too (prepare_payloads).

Gripper squeeze. The real gripper's two fingers are driven by one motor through a rack: whatever part they close on,
the forces of the two fingers along the closing axis are equal and opposite and reach the hand as an internal force,
even when the part is held in a seat off the jaw centre. The simulated fingers are two independent position drives,
so closing on a part that cannot move would push the hand with the difference of the two finger drives, a force the
real arm never feels. The components along the closing axis of the fingers' contacts with a part the gripper may hold
(prepare_payloads) are therefore left out; every other contact component of the fingers counts.

Filter. The estimate is filtered by the median of the last FILTER_TICKS ticks (21 ms at 240 Hz): a contact
impulse spike of one or two ticks (an impact transient of the solver) does not trip the reflex, a contact force held
for three ticks does. The filter is deterministic.

Reflex. When the filtered estimate exceeds FORCE_LIMIT_N the reflex trips in that tick: the backend sets the arm's
drive target to its measured pose (the arm holds where it is), and the running command ends with a CollisionReflex
error; the episode continues and the next command runs normally. While a contact that tripped the reflex is still
pressing (the filtered force has not fallen below REARM_FRACTION of the limit) a new command trips again only if the
filtered force rises RETRIP_MARGIN_N above the lowest value since the trip, so the arm can always move away.

Log (private, never agent-visible). Every robot contact is written to private/robot_contacts.jsonl, merged per
(robot link, body) contact episode: first and last tick, peak force and its tick, deepest penetration and the peak
estimate during the episode. Each reflex is one row with the force, the groups and the contacts that made it.
"""
from collections import deque
import json
import math
import os

# Off by default until verified on every bench; EMBODIEDLAB_COLLISION_REFLEX=1 turns it on.
ENABLED = os.environ.get('EMBODIEDLAB_COLLISION_REFLEX', '0') == '1'
FORCE_LIMIT_N = 20.0            # public: the external force limit of the collision reflex
FILTER_TICKS = 5
REARM_FRACTION = 0.75
RETRIP_MARGIN_N = 5.0
END_EFFECTOR = ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')
BASE_LINK = 'panda_link0'
PAYLOAD = 'payload'
MAX_LOG_ROWS = 200000           # bound on the private log; later episodes are counted, not written
LOG_MIN_FORCE_N = 0.0           # every touching contact (any force, or zero separation) is logged


def public_collision_behaviour():
    """The collision reflex as a public robot constant (device_calibration.collision_reflex)."""
    return {
        'external_force_limit_n': FORCE_LIMIT_N,
        'description': COLLISION_REFLEX_TEXT,
    }


COLLISION_REFLEX_TEXT = (
    f'The robot has a collision reflex, like a real Franka arm: when the external force on the arm, hand or fingers '
    f'(including a held part pressing on something) exceeds {FORCE_LIMIT_N:g} N, the running motion command stops at '
    f'once with a collision reflex error and the arm holds its current pose. The episode continues and the next '
    f'command runs normally.')


class CollisionReflex(RuntimeError):
    """The filtered external force exceeded FORCE_LIMIT_N; the arm holds its measured pose."""


def reflex_error(force_n):
    return CollisionReflex(f'collision reflex: external force {force_n:.1f} N exceeded the {FORCE_LIMIT_N:g} N limit; '
                           'the arm holds its current pose and the rest of this command was not executed; the episode '
                           'continues')


def group_of(side):
    if side == PAYLOAD or side in END_EFFECTOR:
        return 'end_effector'
    return side


def estimate(contacts):
    """(estimate N, {group: |vector sum| N}) of one tick's contacts [(side, body, (fx, fy, fz), ...)]."""
    sums = {}
    for side, _body, force, *_ in contacts:
        if side == BASE_LINK:
            continue
        g = group_of(side)
        s = sums.setdefault(g, [0.0, 0.0, 0.0])
        s[0] += force[0]; s[1] += force[1]; s[2] += force[2]
    groups = {g: math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) for g, v in sums.items()}
    return sum(groups.values()), groups


class CollisionMonitor:
    """Filter, reflex decision and private contact log. Pure Python: contacts come from a reader (PhysxContacts)."""

    def __init__(self, log_path=None, *, limit_n=FORCE_LIMIT_N, filter_ticks=FILTER_TICKS):
        self.limit = float(limit_n)
        self.window = deque([0.0] * filter_ticks, maxlen=filter_ticks)
        self.latched = False
        self.floor = 0.0
        self.filtered = 0.0
        self.reflexes = 0
        self.open = {}                 # (side, body) -> contact episode being merged
        self.rows = 0
        self.dropped = 0
        self.log = open(log_path, 'w') if log_path is not None else None

    # --- log ----------------------------------------------------------------------------------------------------
    def _write(self, row):
        if self.log is None:
            return
        if self.rows >= MAX_LOG_ROWS:
            self.dropped += 1
            return
        self.log.write(json.dumps(row, separators=(',', ':')) + '\n')
        self.rows += 1

    def _merge(self, tick, contacts, est):
        seen = set()
        for side, body, force, *rest in contacts:
            magnitude = math.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2)
            separation = rest[0] if rest else None
            if magnitude <= LOG_MIN_FORCE_N and (separation is None or separation > 0.0):
                continue                     # a speculative pair inside the contact offsets: no touch, no force
            key = (side, body)
            seen.add(key)
            e = self.open.get(key)
            if e is None:
                e = self.open[key] = {'event': 'robot_contact', 'link': side, 'body': body, 'first_tick': tick,
                                      'last_tick': tick, 'ticks': 0, 'peak_n': 0.0, 'peak_tick': tick,
                                      'min_separation_m': None, 'peak_estimate_n': 0.0}
            e['last_tick'] = tick
            e['ticks'] += 1
            if magnitude > e['peak_n']:
                e['peak_n'], e['peak_tick'] = magnitude, tick
            if separation is not None and (e['min_separation_m'] is None or separation < e['min_separation_m']):
                e['min_separation_m'] = separation
            e['peak_estimate_n'] = max(e['peak_estimate_n'], est)
        for key in [k for k in self.open if k not in seen]:
            self._close(key)

    def _close(self, key):
        e = self.open.pop(key)
        e['peak_n'] = round(e['peak_n'], 3)
        e['peak_estimate_n'] = round(e['peak_estimate_n'], 3)
        if e['min_separation_m'] is not None:
            e['min_separation_m'] = round(e['min_separation_m'], 6)
        self._write(e)

    # --- reflex -------------------------------------------------------------------------------------------------
    def update(self, tick, contacts):
        """One physics tick. Returns a CollisionReflex when the reflex trips in this tick, else None."""
        est, groups = estimate(contacts)
        self.window.append(est)
        self.filtered = f = sorted(self.window)[len(self.window) // 2]
        self._merge(tick, contacts, est)
        if self.latched:
            self.floor = min(self.floor, f)
            if f <= REARM_FRACTION * self.limit:
                self.latched = False
        threshold = self.limit if not self.latched else max(self.limit, self.floor) + RETRIP_MARGIN_N
        if f <= threshold:
            return None
        self.latched, self.floor = True, f
        self.reflexes += 1
        top = sorted(contacts, key=lambda c: -(c[2][0] ** 2 + c[2][1] ** 2 + c[2][2] ** 2))[:8]
        self._write({'event': 'collision_reflex', 'tick': tick, 'filtered_force_n': round(f, 3),
                     'raw_force_n': round(est, 3), 'limit_n': self.limit,
                     'groups_n': {g: round(v, 3) for g, v in groups.items()},
                     'contacts': [{'link': c[0], 'body': c[1],
                                   'force_n': round(math.sqrt(c[2][0] ** 2 + c[2][1] ** 2 + c[2][2] ** 2), 3)}
                                  for c in top]})
        if self.log is not None:
            self.log.flush()
        return reflex_error(f)

    def close(self):
        for key in list(self.open):
            self._close(key)
        if self.log is not None and not self.log.closed:
            self._write({'event': 'robot_contact_log_end', 'rows': self.rows, 'dropped_rows': self.dropped,
                         'reflexes': self.reflexes})
            self.log.close()


# --- Isaac / PhysX side ---------------------------------------------------------------------------------------------
def _report(prim):
    from pxr import PhysxSchema
    api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    api.CreateThresholdAttr().Set(0.0)


def prepare_robot(stage, robot_path):
    """Contact reports (no force threshold) on every rigid link of the robot, and gravity compensation. Called before
    the simulation starts (with hooke_physics.configure_arm).

    Gravity compensation: a Franka's joint torque controller compensates the arm's own weight, so the arm stands where
    it is commanded and its external force estimate is not disturbed by its weight. The simulated position servos have
    no such feed-forward: under gravity the hand hangs 0.07-0.2 mm below its target, and a command to hold the measured
    pose (a trajectory starting at the measured joints) drives it that far down. With a part latched in the gripper and
    still resting on its seat, that pushes the part into the seat with ~20-30 N (stiff arm on a rigid floor) and the
    reflex trips at the start of the lift. The robot links therefore carry no gravity, as with the real controller's
    compensation; held parts keep theirs (the payload's weight reaches the arm through the grasp)."""
    from pxr import PhysxSchema, Usd, UsdPhysics
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            _report(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)


GRASPABLE = []                  # prim paths of the parts the gripper may hold (prepare_payloads)
FINGERS = ('panda_leftfinger', 'panda_rightfinger')


def prepare_payloads(stage, paths):
    """Contact reports on every part the gripper may hold, so that what a held part touches is reported."""
    from pxr import UsdPhysics
    del GRASPABLE[:]
    for path in paths:
        prim = stage.GetPrimAtPath(str(path))
        if prim and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            _report(prim)
            GRASPABLE.append(str(path))


def graspable(body, parts):
    return any(body == p or body.startswith(p + '/') for p in parts)


def without_squeeze(contacts, axis, parts):
    """Contacts with the closing-axis component of finger contacts on graspable parts removed (see Gripper squeeze)."""
    if axis is None:
        return contacts
    ax, ay, az = axis
    out = []
    for c in contacts:
        side, body, (fx, fy, fz) = c[0], c[1], c[2]
        if side in FINGERS and graspable(body, parts):
            along = fx * ax + fy * ay + fz * az
            c = (side, body, (fx - along * ax, fy - along * ay, fz - along * az)) + tuple(c[3:])
        out.append(c)
    return out


def prepare_scene(stage, robot_path, private_dir, payloads=()):
    """Once, on the composed stage before the simulation starts: colliders of the drawn shape for every fixed part
    (scene_colliders), and the contact reports of the robot and the parts it may hold."""
    import scene_colliders
    record = scene_colliders.complete(stage, private_dir)
    if not ENABLED:
        return record
    prepare_robot(stage, robot_path)
    prepare_payloads(stage, list(payloads))
    return record


class PhysxContacts:
    """This tick's robot contacts from PhysX's contact report: [(side, collider path of the other body, force on the
    side (N), min separation (m))]; side is a robot link name or PAYLOAD (the held part)."""

    def __init__(self, robot_path, dt):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        self.interface = get_physx_simulation_interface()
        self.decode = PhysicsSchemaTools.intToSdfPath
        self.prefix = robot_path.rstrip('/') + '/'
        self.dt = float(dt)
        self.paths = {}
        self.fingers = [self.prefix + name for name in FINGERS]

    def closing_axis(self):
        """Unit vector from the left to the right finger (world), or None."""
        from omni.physx import get_physx_interface
        physx = get_physx_interface()
        try:
            a, b = (physx.get_rigidbody_transformation(p)['position'] for p in self.fingers)
        except Exception:
            return None
        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        return None if n < 1e-4 else (d[0] / n, d[1] / n, d[2] / n)

    def _path(self, code):
        path = self.paths.get(code)
        if path is None:
            path = self.paths[code] = str(self.decode(code))
        return path

    def _side(self, path, payload):
        if path.startswith(self.prefix):
            return path[len(self.prefix):].split('/', 1)[0]
        if payload is not None and (path == payload or path.startswith(payload + '/')):
            return PAYLOAD
        return None

    def read(self, payload=None):
        headers, data = self.interface.get_contact_report()[:2]
        out = []
        for h in headers:
            n = int(h.num_contact_data)
            if not n:
                continue
            # Collider paths (a robot link's collision mesh lies under the link; a body's colliders under the body).
            a0 = self._path(h.collider0) if h.collider0 else self._path(h.actor0)
            a1 = self._path(h.collider1) if h.collider1 else self._path(h.actor1)
            s0, s1 = self._side(a0, payload), self._side(a1, payload)
            if (s0 is None) == (s1 is None):        # no robot side, or internal to the robot and its payload
                continue
            fx = fy = fz = 0.0
            separation = math.inf
            offset = int(h.contact_data_offset)
            for d in data[offset:offset + n]:
                j = d.impulse
                fx += j[0]; fy += j[1]; fz += j[2]
                separation = min(separation, float(d.separation))
            # PhysX: the impulse acts on actor0 (its normal points from actor1 to actor0).
            sign = 1.0 / self.dt if s0 is not None else -1.0 / self.dt
            side, body = (s0, a1) if s0 is not None else (s1, a0)
            out.append((side, body, (fx * sign, fy * sign, fz * sign), separation if separation < math.inf else None))
        if GRASPABLE and any(c[0] in FINGERS and graspable(c[1], GRASPABLE) for c in out):
            out = without_squeeze(out, self.closing_axis(), GRASPABLE)
        return out


class Reflex:
    """What a backend owns: the PhysX reader and the monitor. check once per physics tick, after world.step."""

    def __init__(self, robot_path, dt, private_dir):
        from pathlib import Path
        self.reader = self.monitor = None
        if ENABLED:
            self.reader = PhysxContacts(robot_path, dt)
            self.monitor = CollisionMonitor(Path(private_dir) / 'robot_contacts.jsonl')

    def check(self, tick, payload=None):
        if self.monitor is None:
            return None
        return self.monitor.update(tick, self.reader.read(payload))

    def close(self):
        if self.monitor is not None:
            self.monitor.close()


def held_path(backend):
    """Prim path of the part the gripper holds now, or None (benches with the contact-conditioned grasp)."""
    held = getattr(backend, 'held', None)
    paths = getattr(backend, 'paths', None)
    if held is None or not isinstance(paths, dict):
        return None
    path = paths.get(held)
    return str(path) if path is not None else None
