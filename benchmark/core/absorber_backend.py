"""Vibration absorber physics device: the chemistry backend with the absorber apparatus. The gripper carries the brass
weights (real masses); the carrier is pushed along its blade by hand; the valve keys and SWEEP are pressed.

Every physics tick:
- the weights standing in the tray's sockets are read in the carrier's frame (as a rack seat: xy within SEAT_XY_TOL_M of
  the socket centre, the foot within SEAT_Z_TOL_M of its floor, tilt at most SEAT_TILT_DEG; not held or being released);
- the carrier's clamp point s is read from its pose in the mount frame, its speed from the change of s; within the
  index pin's reach the detent force toward the nearest detent is applied along the track for the next step;
- the keys: SWEEP's cap depth below its rest (the kit's), the valve keys' depth relative to the mount (they ride on it);
- PhysX's contact report gives the loads: a contact on a strip piece (or the stinger) loads the strip at its point, its
  force normal to the piece; a contact on the mount or a valve key loads the mount; a contact on the carrier, the yoke
  or a blade piece (its share at the carrier: the cantilever's influence) loads the absorber; a robot contact on a
  weight standing in the tray loads the absorber too. The weights standing in the tray are the absorber's mass (their
  own contacts with the tray are not loads). Whoever presses on the strip or the absorber with at least
  absorber_bench.TOUCH_N touches it (a sweep running is void);
- the bench advances one tick, and the strip, mount, blade, track, yoke and stinger get kinematic targets at the
  present state (absorber_bench poses): the carrier rides its track, the weights in the tray ride the carrier.
At each rendered frame the same poses are written to USD for the renderer, the analyzer and valve LCDs show the bench's
displays (the analyzer refreshes ANALYZER_REFRESH_S while a sweep runs) and the accelerometer's cable follows the tip.
"""
from __future__ import annotations

import math
import os

import numpy as np

from chem_backend import ChemBackend, SEAT_TILT_DEG, SEAT_XY_TOL_M, SEAT_Z_TOL_M, _tilt_deg

ROBOT_TOUCH_N = 1e-4
ROBOT_LOAD_SLEW_N = float(os.environ.get('EMBODIEDLAB_ABSORBER_SLEW_N', '0.02'))
                          # the robot's load on the absorber changes by at most this per physics step (4.8 N/s): the
                          # carrier's height is the beam's state, posed kinematically a step after the load that moved
                          # it, so a finger held against the carrier or the tray (stiff, the arm position-controlled)
                          # closes a delayed loop that pumps the absorber up at its resonance (seen: a finger left a
                          # millimetre over the tray after a hold-down push, 1 N, then 34 N, then a tray thrown off).
                          # Slewed, a finger's push still deflects the absorber, at a hand's pace. The touch damping
                          # (bench TOUCH_DAMPING_NS_M) does not replace it: at 0.2 N per step fingers closing on a weight
                          # in its socket fed the beam 60 N within a second and threw the carrier off its detent
LIFT_CLEAR_M = 0.004   # a weight lifted out of its socket stays the absorber's mass until its foot is this far above the
                       # floor: the rigid hold carries none of its weight while it still stands there, and the tray
                       # springing up the moment the fingers close would shove it out of the fingers. A weight being set
                       # in is not the absorber's mass while it is in the fingers (held 3 mm over the floor, a mass
                       # that came with the foot inside this band and went as the carrier sagged under it made a relay:
                       # the empty carrier rang at the blade's 30 Hz mode and was thrown 5-25 mm); let go, it is from
                       # the moment its foot is in the band
SETTLE_M = 0.010       # a weight let go into a socket stays the absorber's mass while it settles (bouncing in its well:
                       # unseated, its impacts on the carrier would be loads fed to the beam) unless its foot rises this
                       # far above the floor
ROBOT_TOUCH_HOLD = 12      # physics steps (50 ms) a robot contact on the absorber counts as a finger on it (see bench
                           # TOUCH_DAMPING: contacts flicker as the absorber rings against a finger)
ANALYZER_REFRESH_S = 0.25
CARRIER_V_TICKS = 120  # the carrier's speed is measured over 0.5 s (at 240 Hz): a carrier shaken about its detent by the
                       # blade at resonance (a few tenths of a mm at the drive frequency) is not moving along the track
LOAD_MAX_N = 60.0      # a part's load from one physics step at most (a Franka pushes with a few tens of newtons; PhysX can
                       # report far larger impulses when it pushes a held object into a kinematic part)


def _rotate(q_wxyz, v):
    w, x, y, z = [float(c) for c in q_wxyz]
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _relative(q_wxyz, o_wxyz):
    """The rotation o seen from the frame q: conj(q) * o (wxyz)."""
    w0, x0, y0, z0 = [float(c) for c in q_wxyz]
    w1, x1, y1, z1 = [float(c) for c in o_wxyz]
    x0, y0, z0 = -x0, -y0, -z0
    return (w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1)


class AbsorberBackend(ChemBackend):
    ASSET = 'absorber_apparatus'

    def __init__(self, args, app, configuration):
        import time
        self._t0 = time.monotonic()
        super().__init__(args, app, configuration)
        self._mark('backend ready')

    def _gripper_mechanism(self, positions, velocities):
        """The kit's contact-conditioned grasp (grip_contact: both fingers touch the part, it stops them, and it moves
        with the hand within 0.05 m/s), the part's motion judged from its positions over the fingers' stall window
        (GRIP_STALL_TICKS: the displacement over the last 6 ticks, 25 ms) instead of the solver's reported velocity.
        A light weight pinched in the fingers while its foot stands on a floor is reported moving at 0.05-0.14 m/s
        while it does not move (p80, 10 g on a rack seat: (-60, -1, +32) mm/s for 200 ticks, finger forces steady);
        judged that way the grip took hold only by chance. A part that slides in the fingers, or that the hand leaves,
        moves relative to the jaw over the window and is not held. With the bring-up trace on, why a closing grip does
        not take hold."""
        import grip_contact
        jaw0 = getattr(self, 'previous_jaw', None)
        history = getattr(self, '_grip_positions', None)
        if history is None:
            from collections import deque
            history = self._grip_positions = deque(maxlen=grip_contact.GRIP_STALL_TICKS + 1)
        history.append(np.asarray(positions, float).copy())
        if len(history) == history.maxlen and history[0].shape == history[-1].shape:
            velocities = (history[-1] - history[0]) / ((history.maxlen - 1) * self.dt)
        super()._gripper_mechanism(positions, velocities)
        if os.environ.get('EMBODIEDLAB_ABSORBER_TRACE', '0') != '1' or self.held is not None or jaw0 is None:
            return
        q = np.asarray(self.robot.get_joint_positions())[7:9]
        if self.width_target is None or not self.width_target + 0.002 < float(q[0] + q[1]) < 0.040:
            return
        left = getattr(self, '_grip_diag_left', 3000)
        if left <= 0:
            return
        import sys
        import grip_contact
        jv = (np.asarray(self.previous_jaw) - np.asarray(jaw0)) / self.dt
        qd = float(np.max(np.abs(np.asarray(self.robot.get_joint_velocities())[7:9])))
        rows = []
        for k, i in enumerate(self.ids):
            d = float(np.linalg.norm(np.asarray(positions[k], float) - np.asarray(self.previous_jaw, float)))
            if d > 0.20:
                continue
            t = grip_contact.fingers_touching(self.finger_sensors[i], self.dt)
            if not any(t):
                continue
            v = np.asarray(velocities[k], float)
            rel = float(np.linalg.norm(v - jv))
            rows.append((i, t, round(rel, 4), round(d * 1000, 1), [round(c * 1000, 1) for c in v],
                         [round(c * 1000, 1) for c in jv]))
        if not rows:                                   # only grips that touch a weight
            return
        self._grip_diag_left = left - 1
        print(f"[absorber-grip] tick {self.tick_index} width_mm {float(q[0] + q[1]) * 1000:.2f} qd {qd:.4f} "
              f"stalled {self.grip.stalled} rows {rows}", file=sys.stderr, flush=True)

    def _trace(self, tray, loads, touch, summary):
        """Bring-up trace on stderr (EMBODIEDLAB_ABSORBER_TRACE=1 turns it on): the carrier, the absorber and the loads
        every 0.5 s, every tick while the carrier is off its travel."""
        import os
        import sys
        if os.environ.get('EMBODIEDLAB_ABSORBER_TRACE', '0') != '1' or self.bench is None:
            return
        s = self.carrier['s_m']
        off = not (0.020 <= s <= 0.090)
        self._trace_off = getattr(self, '_trace_off', 0) + off
        fq, fqd = None, 0.0
        if getattr(self, '_settled', False):             # no dense trace while the bench settles (a slow start)
            try:
                fq = np.asarray(self.robot.get_joint_positions(), float)[7:9]
                fqd = float(np.max(np.abs(np.asarray(self.robot.get_joint_velocities(), float)[7:9])))
            except Exception:
                fq, fqd = None, 0.0
        slip = getattr(self, '_settled', False) and abs(s * 1000 - 5 * round(s * 200)) > 0.25   # a quarter mm off its detent
        fast = slip or abs(self.carrier['v_m_s']) > 0.030 or getattr(self, '_jit', {}).get('off_mm', 0.0) > 0.3 or fqd > 0.002 \
            or 'carrier:robot' in summary
        if fast and getattr(self, '_dense_left', 0) <= 0 and getattr(self, "_dense_total", 0) < 12000:
            self._dense_left = 48
        dense = getattr(self, '_dense_left', 0) > 0
        if dense:
            self._dense_left -= 1
            self._dense_total = getattr(self, '_dense_total', 0) + 1
        if self.tick_index % 120 and not dense and not (off and self._trace_off <= 200):
            return
        st = self.bench.strip_state()
        print(f"[absorber-trace] tick {self.tick_index} s_mm {s * 1000:.3f} v_mm_s {self.carrier['v_m_s'] * 1000:.2f} "
              f"z_mm {st['z'] * 1000:.4f} tip_mm {float(self.bench._tip()) * 1000:.4f} tray {tray} "
              f"absN {loads['absorber_N']:.4f} rtouch {int(loads['robot_on_absorber'])} zF_mm {self._static_z_mm(loads):.4f} "
              f"mountN {loads['mount_N']:.4f} "
              f"strip {[(round(x, 3), round(f, 4)) for x, f in loads['strip']][:6]} touch {touch} sum {summary} "
              f"jit {({k: (round(v, 3) if isinstance(v, float) else v) for k, v in getattr(self, '_jit', {}).items()})} holds {self.holds()}"
              f"{' dense fq ' + str(None if fq is None else np.round(fq * 1000, 2).tolist()) if dense else ''}"
              f"{' wc ' + str(getattr(self, '_wc', [])) if dense else ''}"
              f"{' hc ' + str(getattr(self, '_hc', [])) if dense else ''}",
              file=sys.stderr, flush=True)
        self._jit = {}
        self._wc = []
        self._hc = []

    def _jitter(self, positions, orientations):
        """Bring-up: the window's largest carrier error against its track (off the joint axis, mm; turn, deg) and the
        largest offset of a weight in the tray from its socket in the carrier frame (lateral and up, mm; tilt, deg)."""
        if self.poses is None:
            return
        B = self.B
        tp = self.poses[B.SEGMENTS + 1 + B.BLADE_PIECES]
        a = 2 * math.atan2(tp[4], tp[6])
        p, q = self._carrier_pose()
        d = p - np.asarray(tp[:3], float)
        c, s = math.cos(a), math.sin(a)
        off = math.hypot(d[1], s * d[0] + c * d[2])
        qt = (math.cos(a / 2), 0.0, -math.sin(a / 2), 0.0)                  # the track's inverse, wxyz
        w = qt[0] * q[0] - qt[2] * q[2]                                       # the scalar part of qt * q
        turn = math.degrees(2 * math.acos(min(1.0, abs(w))))
        j = self.__dict__.setdefault('_jit', {})
        j['off_mm'] = max(j.get('off_mm', 0.0), off * 1000)
        j['turn_deg'] = max(j.get('turn_deg', 0.0), turn)
        floor = self.asset.socket_floor_local()
        qc = (q[0], -q[1], -q[2], -q[3])
        for i, key in enumerate(self.ids):
            if key not in self.tray:
                d = np.asarray(positions[i], float) - p
                if float(np.linalg.norm(d)) < 0.12:
                    j[f'c_{key}'] = tuple(round(float(v) * 1000, 2) for v in _rotate(qc, d))
                continue
            n = int(self.tray[key].split(':')[1])
            rel = _rotate(qc, np.asarray(positions[i], float) - p) - np.array([0.0, B.SOCKET_Y[n - 1], floor])
            j['w_lat_mm'] = max(j.get('w_lat_mm', 0.0), math.hypot(rel[0], rel[1]) * 1000)
            j['w_up_mm'] = max(j.get('w_up_mm', -9.0), rel[2] * 1000)
            j['w_tilt_deg'] = max(j.get('w_tilt_deg', 0.0), _tilt_deg(_relative(q, orientations[i])))   # as the seat rule
            j[f'rel_{key}'] = tuple(round(float(v) * 1000, 2) for v in rel)

    def holds(self):
        """Each hold-down slide's place against its closed rest pose in the carrier frame: (x mm, -stroke open .. 0
        closed; lift mm)."""
        if self.hold_view is None:
            from isaacsim.core.prims import RigidPrim
            self.hold_view = RigidPrim(self.hold_paths, name='absorber_holds', reset_xform_properties=False)
            self.hold_view.initialize()
            self.hold_order = [self.hold_paths.index(str(p)) for p in self.hold_view.prim_paths]
        p, q = self._carrier_pose()
        pos = np.asarray(self.hold_view.get_world_poses()[0], float)
        qc = (q[0], -q[1], -q[2], -q[3])
        out = [None] * len(self.hold_paths)
        for k, i in enumerate(self.hold_order):
            rel = _rotate(qc, pos[k] - p) - np.asarray(self.asset.hold_origin_local(i + 1))
            out[i] = (round(float(rel[0]) * 1000, 2), round(float(rel[2]) * 1000, 2))
        return out

    def _mark(self, what):
        import sys
        import time
        print(f'[absorber] {time.monotonic() - self._t0:8.1f} s  {what}', file=sys.stderr, flush=True)
        if what.startswith('settled'):
            self._settled = True

    def _build(self, asset):
        self._mark('build start')
        bench = self.configuration['bench']
        built = asset.build(self.stage, {'weights': bench['weight_labels'], 'masses_kg': bench['masses_kg'],
                                         'weight_seats': bench['weight_seats'], 'carrier_mm': bench['carrier_mm'],
                                         'gap_m': bench['gap_m']})
        import absorber_bench as B
        self.B = B
        self.kin_paths = asset.kinematic_paths()
        self.piece_index = {asset.segment_path(i): i for i in range(B.SEGMENTS)}
        self.blade_index = {asset.blade_path(i): i for i in range(B.BLADE_PIECES)}
        self.weight_of = {path: w for w, path in built['objects'].items()}
        self.hold_paths = [asset.hold_path(n) for n in range(1, len(B.SOCKET_Y) + 1)]
        self.hold_view = None
        self.key_paths = {built['buttons'][k]: k for k in B.VALVE_KEYS}
        self.kin_view, self.kin_order = None, None
        self.carrier_view = None
        self.detent_drive, self.detent_s0, self.detent_hole = None, None, None
        self.poses, self.poses_drawn, self.cable_drawn = None, None, None
        self.frame = None                       # the mount frame of the latest targets
        self.table_z = None
        self.carrier = None                     # {'s_m', 'v_m_s'} of the last tick
        self.carrier_along = None               # the carrier's place along its track (track frame x, m) at that tick
        self.tray = {}
        self.analyzer_drawn, self.analyzer_at = None, -1.0
        self.lcd_drawn = None
        self.key_rest = None                    # valve keys' rest height above the mount origin
        self._path_cache = {}
        self._contacts_tick, self._contacts = None, None
        self.load_audit = []
        self.robot_load_N = 0.0                 # the robot's load on the absorber as applied (slewed, see contacts)
        self.robot_touch_left = 0               # steps the robot still counts as touching the absorber
        self._mark('build done')
        return {'objects': built['objects'], 'buttons': built['buttons'], 'seats': built['seats'], 'screens': {}}

    def _support_paths(self, key, built):
        A = self.asset
        return list(built['seats'].values()) + [A.ROOT + '/Rack', A.CARRIER]

    def _table_z(self):
        if self.table_z is None:
            floor = float(np.mean([f for _, f in self.seats.values()]))
            self.table_z = floor - (self.B.RACK['base'][2] + self.asset.C.SEAT_FLOOR)
        return self.table_z

    def _mount_frame(self):
        if self.frame is None:
            B = self.B
            return (B.MOUNT_X, B.BEAM_Y, self._table_z() + B.BEAM_TOP), 0.0
        return self.frame

    # --- the carrier ----------------------------------------------------------------------------------------------
    def _carrier_pose(self):
        if self.carrier_view is None:
            from isaacsim.core.prims import RigidPrim
            self.carrier_view = RigidPrim(self.asset.CARRIER, name='absorber_carrier', reset_xform_properties=False)
            self.carrier_view.initialize()
        p, q = self.carrier_view.get_world_poses()
        return np.asarray(p, float)[0], np.asarray(q, float)[0]

    def _read_carrier(self):
        p, _ = self._carrier_pose()
        s = float(self.B.carrier_s(self._mount_frame(), p))
        if self.poses is not None:                    # the track's pose in this tick's step (its targets, set after the
            tp = self.poses[self.B.SEGMENTS + 1 + self.B.BLADE_PIECES]          # last step's read)
            a = 2 * math.atan2(tp[4], tp[6])
            self.carrier_along = math.cos(a) * (float(p[0]) - tp[0]) - math.sin(a) * (float(p[2]) - tp[2])
        if self.carrier is None:
            self.carrier_track = []
        track = self.carrier_track
        track.append(s)
        del track[:-CARRIER_V_TICKS - 1]
        # the carrier's speed over the last CARRIER_V_TICKS ticks: the change of its mean position from the first half of
        # the window to the second (a one-tick difference carries PhysX's contact jitter, and an end-to-end difference
        # the carrier's rocking about its detent while the blade shakes it)
        h = len(track) // 2
        v = 0.0 if h < 1 else (float(np.mean(track[-h:])) - float(np.mean(track[:h]))) / ((len(track) - h) * self.dt)
        self.carrier = {'s_m': s, 'v_m_s': float(v), 's_avg_m': float(np.mean(track))}
        return self.carrier

    def _detent(self):
        """The index pin, a spring ball plunger: while its ball sits in a hole the carrier slide's drive is a stiff spring
        to that hole (force capped at DETENT_PEAK_N + CLAMP_FRICTION_N: pushing harder rolls the ball out); once the
        ball is DETENT_RELEASE_M out of the hole it rolls on the flat and the drive is the clamp's friction alone, until
        the carrier comes within DETENT_CAPTURE_M of a hole and the ball drops in (for the next physics step). The
        friction is static: the same stiff spring to an anchor where the slide stands, capped at CLAMP_FRICTION_N, the
        anchor dragged along to within CLAMP_SLIP_M of the slide as it is pushed (a saturated damper alone, PhysX's
        contact jitter dithering it, let the slide walk some 1 mm/s along its track)."""
        if self.carrier is None:
            self._read_carrier()
        A, B = self.asset, self.B
        if self.detent_drive is None:
            from pxr import UsdPhysics
            prim = self.stage.GetPrimAtPath(A.CARRIER_JOINT)
            self.detent_drive = UsdPhysics.DriveAPI.Get(prim, 'linear')
            self.detent_s0 = float(prim.GetAttribute(A.CARRIER_S0_ATTR).Get())
            self.detent_hole = B.nearest_detent(self.detent_s0)[0]          # authored in the starting hole
            self.detent_anchor = None
        s = self.carrier['s_m']
        hole = self.detent_hole
        if hole is not None and abs(s - hole / 1000.0) > A.DETENT_RELEASE_M:
            hole = None
        elif hole is None:
            mm, off = B.nearest_detent(s)
            if abs(off) <= A.DETENT_CAPTURE_M:
                hole = mm
        drive = self.detent_drive
        # the joint's position: the carrier's place along its track, which tilts with the blade's slope at the carrier
        # (s0 - s, s taken level in the mount frame, is short of it by s (1 - cos): 0.18 mm at 61 mm under a 3 mm sag,
        # and an anchor set there pulled the loaded carrier along at the friction's full 1.5 N)
        x = self.detent_s0 - s if self.carrier_along is None else self.detent_s0 + self.carrier_along
        if hole == self.detent_hole:
            a = self.detent_anchor
            if hole is None and abs(x - a) > A.CLAMP_SLIP_M:               # pushed past the friction: the anchor trails
                self.detent_anchor = x - math.copysign(A.CLAMP_SLIP_M, x - a)
                drive.GetTargetPositionAttr().Set(float(self.detent_anchor))
            return
        if hole is None:
            self.detent_anchor = x
            drive.GetTargetPositionAttr().Set(float(x))
            drive.GetMaxForceAttr().Set(float(A.CLAMP_FRICTION_N))
        else:
            self.detent_anchor = None
            drive.GetTargetPositionAttr().Set(float(self.detent_s0 - hole / 1000.0))
            drive.GetMaxForceAttr().Set(float(A.DETENT_PEAK_N + A.CLAMP_FRICTION_N))
        drive.GetStiffnessAttr().Set(float(A.DETENT_STIFFNESS))
        self._mark(f'index pin {"out of its hole" if hole is None else f"in the {hole} mm hole"} at s {s * 1000:.2f} mm')
        self.detent_hole = hole

    def _tray(self, positions, orientations):
        """{weight: socket} of the weights standing in the tray's sockets (upright on the tray: the tilt is taken
        against the carrier, which pitches with the blade)."""
        B = self.B
        p, q = self._carrier_pose()
        floor = self.asset.socket_floor_local()
        sockets = {n: p + _rotate(q, (0.0, y, floor)) for n, y in enumerate(B.SOCKET_Y, start=1)}
        tray = {}
        for i, key in enumerate(self.ids):
            if key in (self.held, self.releasing) or _tilt_deg(_relative(q, orientations[i])) > SEAT_TILT_DEG:
                continue
            for n, c in sockets.items():
                if np.linalg.norm(positions[i][:2] - c[:2]) <= SEAT_XY_TOL_M and abs(positions[i][2] - c[2]) <= SEAT_Z_TOL_M:
                    tray[key] = f'tray:{n}'
                    break
        return tray

    def _in_wells(self, positions, tray):
        """{weight: socket} of the weights not standing in the tray that are the absorber's mass still or already:
        one in the fingers that was its mass (it stood in the socket) while its foot is less than LIFT_CLEAR_M above
        the floor; one let go (or free) from when its foot is in a socket's well less than LIFT_CLEAR_M above the
        floor, while it stays less than SETTLE_M above it. In a well: within the seat rule's xy of its centre."""
        B = self.B
        p, q = self._carrier_pose()
        floor = self.asset.socket_floor_local()
        sockets = [(n, p + _rotate(q, (0.0, y, floor))) for n, y in enumerate(B.SOCKET_Y, start=1)]
        before = getattr(self, '_mass_before', {})
        wells = {}
        for i, key in enumerate(self.ids):
            if key in tray:
                continue
            pos = positions[i]
            for n, c in sockets:
                if np.linalg.norm(pos[:2] - c[:2]) > SEAT_XY_TOL_M:
                    continue
                up = pos[2] - c[2]
                if key == self.held:
                    mass = key in before and -SEAT_Z_TOL_M <= up <= LIFT_CLEAR_M
                else:
                    mass = -SEAT_Z_TOL_M <= up <= (SETTLE_M if key in before else LIFT_CLEAR_M)
                if mass:
                    wells[key] = f'tray:{n}'
                break
        return wells

    # --- contacts --------------------------------------------------------------------------------------------------
    def _path(self, code):
        path = self._path_cache.get(code)
        if path is None:
            from pxr import PhysicsSchemaTools
            path = self._path_cache[code] = str(PhysicsSchemaTools.intToSdfPath(code))
        return path

    def _part(self, path):
        A = self.asset
        if path in self.piece_index:
            return ('strip', self.piece_index[path])
        if path in self.blade_index:
            return ('blade', self.blade_index[path])
        if path == A.STINGER:
            return ('stinger', None)
        if path == A.MOUNT:
            return ('mount', None)
        if path in self.key_paths:
            return ('key', self.key_paths[path])
        if path == A.CARRIER or path in self.hold_paths:
            return ('carrier', None)                  # the hold-downs ride the carrier: part of the absorber
        if path == A.YOKE:
            return ('yoke', None)
        return None

    def _who(self, path):
        if path in self.weight_of:
            return self.weight_of[path]
        if path.startswith(self.lab.ROBOT_PATH):
            return 'robot'
        return 'other:' + path.rsplit('/', 1)[-1]

    def _static_z_mm(self, loads):
        """The absorber's static z under the present load (the model's rest with absorber_N held; trace only)."""
        p = self.bench.plant
        f = loads['absorber_N'] - p.m * 9.81
        z = f / p.k
        if abs(z) > p.gap:
            z = (f + math.copysign(self.B.M.K_STOP * p.gap, z)) / (p.k + self.B.M.K_STOP)
        return z * 1000

    def contacts(self, tray):
        """The loads of the last physics step for the bench ({'strip': [(x, F up)], 'mount_N', 'absorber_N'}), who
        touches the strip or the absorber, and a summary per (part, who) for the audit."""
        if self._contacts_tick == self.tick_index:
            return self._contacts
        from omni.physx import get_physx_simulation_interface
        B = self.B
        headers, data = get_physx_simulation_interface().get_contact_report()[:2]
        strip, mount_N, absorber_N, robot_N = [], 0.0, 0.0, 0.0
        robot_on_absorber = False
        pressing, summary = {}, {}
        frame = self._mount_frame()
        L = self.carrier['s_m'] if self.carrier else self.configuration['bench']['carrier_mm'] / 1000.0
        for h in headers:
            n = int(h.num_contact_data)
            if not n:
                continue
            a0, a1 = self._path(h.actor0), self._path(h.actor1)
            offset = int(h.contact_data_offset)
            points = data[offset:offset + n]
            if getattr(self, '_dense_left', 0) > 0 and (a0 in self.hold_paths or a1 in self.hold_paths):
                hp, o = (a0, a1) if a0 in self.hold_paths else (a1, a0)
                jmax = max(float(np.linalg.norm(d.impulse)) for d in points) / self.dt
                jsum = np.sum([np.asarray(d.impulse, float) for d in points], axis=0) / self.dt
                pc = np.mean([np.asarray(d.position, float) for d in points], axis=0)
                self.__dict__.setdefault('_hc', []).append(
                    ('/'.join(str(hp).split('/')[-1:]), '/'.join(str(o).split('/')[-2:]), n, round(jmax, 2),
                     [round(float(v), 2) for v in jsum], [round(float(v) * 1000, 1) for v in pc]))
            if getattr(self, '_dense_left', 0) > 0 and (a0 in self.weight_of or a1 in self.weight_of):
                w, o = (a0, a1) if a0 in self.weight_of else (a1, a0)
                jmax = max(float(np.linalg.norm(d.impulse)) for d in points) / self.dt
                pc = np.mean([np.asarray(d.position, float) for d in points], axis=0)
                self.__dict__.setdefault('_wc', []).append(
                    (self.weight_of[w], '/'.join(str(o).split('/')[-2:]), n, round(jmax, 3),
                     [round(float(v) * 1000, 1) for v in pc]))
            # the robot pressing a weight that stands in the tray: a load on the absorber. Only pressing in: a weight
            # resting in its socket cannot pull the carrier up, and fingers that grip it (their friction holding it
            # while the carrier moves under it) would otherwise feed the beam a lift that the gripped weight then
            # resists, a loop that runs away. Lifting a weight off takes its load away through its own contacts: once
            # held it is no longer the absorber's mass and its contacts with the carrier are loads (below).
            for w, o, sign in ((a0, a1, 1.0), (a1, a0, -1.0)):
                if w in self.weight_of and self.weight_of[w] in tray and o.startswith(self.lab.ROBOT_PATH):
                    robot_on_absorber = True
                    fz = sum(sign * float(d.impulse[2]) for d in points) / self.dt
                    if any(float(np.linalg.norm(d.impulse)) / self.dt > ROBOT_TOUCH_N for d in points):
                        robot_N += min(fz, 0.0)           # PhysX z up: pressing down on the weight is negative fz
                        pressing['robot'] = max(pressing.get('robot', 0.0), abs(fz))
            p0, p1 = self._part(a0), self._part(a1)
            if (p0 is None) == (p1 is None):
                continue                                  # neither, or two parts of the bench (assembled into each other)
            part, other, sign = (p0, a1, 1.0) if p0 is not None else (p1, a0, -1.0)
            who = self._who(other)
            if who in tray and part[0] == 'carrier':
                continue                                  # a weight standing in the tray: part of the absorber's mass
            # a weight in the fingers (held, or being let go) on the carrier: it touches the absorber, but its contact
            # is not fed to the beam as a load. The rigid hold makes it part of the arm, and its contact with the
            # carrier (which the beam's state moves) would close a stiff loop that runs away within a few ticks as a
            # weight is lifted out of its socket or set into it; the weight's mass leaves the absorber when its foot
            # is LIFT_CLEAR_M up and joins it once let go into a well (_in_wells).
            gripped = who in (self.held, self.releasing) and part[0] == 'carrier'
            robot = other.startswith(self.lab.ROBOT_PATH)
            kind, key = part
            if robot and kind in ('carrier', 'yoke'):
                robot_on_absorber = True
            fz_sum = 0.0
            for d in points:
                pos, j = np.asarray(d.position, float), np.asarray(d.impulse, float)
                if kind == 'strip':
                    i = key
                    pose = self.poses[i] if self.poses is not None else None
                    centre = np.asarray(pose[:3]) if pose is not None else np.array(
                        [B.CLAMP_X + B.segment_x(i), B.BEAM_Y, self._table_z() + B.BEAM_TOP - B.STRIP[2] / 2])
                    theta = 2 * math.atan2(pose[4], pose[6]) if pose is not None else 0.0
                    up = np.array([math.sin(theta), 0.0, math.cos(theta)])     # the piece's normal (beam_bench)
                    f = abs(float(j @ up)) / self.dt
                    if float((pos - centre) @ up) > 0:
                        f = -f                            # pressing on its top: down
                    strip.append((float(pos[0]) - B.CLAMP_X, f))
                    fz = f
                else:
                    fz = sign * float(j[2]) / self.dt     # PhysX: the impulse acts on actor0 (normal from 1 to 0)
                    if kind == 'stinger':
                        strip.append((B.M.X_F, fz))
                    elif kind in ('mount', 'key'):
                        mount_N += fz
                    elif kind in ('carrier', 'yoke'):
                        if robot:
                            robot_N += fz
                        else:
                            absorber_N += 0.0 if gripped else fz
                    else:                                 # a blade piece: its share at the carrier, the rest on the mount
                        s = B.blade_s(key)
                        share = 1.0 if s >= L else s ** 2 * (3 * L - s) / (2 * L ** 3)
                        if robot:
                            robot_N += share * fz
                        else:
                            absorber_N += share * fz
                        mount_N += (1 - share) * fz
                fz_sum += fz
            row = summary.setdefault((kind, who), [0.0, 0])
            row[0] += fz_sum
            row[1] += n
            if kind != 'key':
                pressing[who] = max(pressing.get(who, 0.0), abs(fz_sum))
        touch = sorted(w for w, f in pressing.items() if f >= B.TOUCH_N)
        robot_N = float(np.clip(robot_N, -LOAD_MAX_N, LOAD_MAX_N))
        self.robot_load_N += float(np.clip(robot_N - self.robot_load_N, -ROBOT_LOAD_SLEW_N, ROBOT_LOAD_SLEW_N))
        absorber_N += self.robot_load_N
        self.robot_touch_left = ROBOT_TOUCH_HOLD if robot_on_absorber else max(self.robot_touch_left - 1, 0)
        total = sum(f for _, f in strip)
        if abs(total) > LOAD_MAX_N:
            strip = [(x, f * LOAD_MAX_N / abs(total)) for x, f in strip]
        loads = {'strip': strip, 'mount_N': float(np.clip(mount_N, -LOAD_MAX_N, LOAD_MAX_N)),
                 'absorber_N': float(np.clip(absorber_N, -LOAD_MAX_N, LOAD_MAX_N)),
                 'robot_on_absorber': self.robot_touch_left > 0}
        self._contacts_tick = self.tick_index
        self._contacts = (loads, touch, {f'{k}:{w}': [round(v[0], 4), v[1]] for (k, w), v in summary.items()})
        return self._contacts

    # --- world as the bench sees it ---------------------------------------------------------------------------
    def _world(self, positions, orientations):
        A, B = self.asset, self.B
        self._table_z()
        seated = {sid: None for sid in self.seats}
        tilts = {}
        for i, key in enumerate(self.ids):
            tilts[key] = _tilt_deg(orientations[i])
            if key in (self.held, self.releasing) or tilts[key] > SEAT_TILT_DEG:
                continue
            for sid, (xy, floor) in self.seats.items():
                if np.linalg.norm(positions[i][:2] - xy) <= SEAT_XY_TOL_M and abs(positions[i][2] - floor) <= SEAT_Z_TOL_M:
                    seated[sid] = key
                    break
        self._read_carrier()
        tray = self._tray(positions, orientations)
        self.tray = tray
        mass = dict(tray)                              # the absorber's mass, for the bench: the seated weights and any
        mass.update(self._in_wells(positions, tray))   # being lifted out or settling in (LIFT_CLEAR_M, SETTLE_M)
        self._mass_before = mass
        for key, sid in tray.items():
            seated[sid] = key
        caps = np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        (_, _, mz), _ = self._mount_frame()
        if self.key_rest is None:
            self.key_rest = {b: float(caps[k] - mz) for k, b in enumerate(self.button_ids) if b in B.VALVE_KEYS}
        depth = self.cap_rest_z - caps
        for k, b in enumerate(self.button_ids):
            if b in B.VALVE_KEYS:
                depth[k] = self.key_rest[b] - (caps[k] - mz)
        for b, d in zip(self.button_ids, depth):
            if d >= A.C.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= A.C.SWITCH_UP_M:
                self.pressed.discard(b)
        loads, touch, summary = self.contacts(mass)
        self.last_summary = summary
        self._jitter(positions, orientations)
        self._trace(tray, loads, touch, summary)
        return ({'seats': seated, 'tray': mass, 'held': self.held, 'pressed': set(self.pressed),
                 'carrier': dict(self.carrier), 'loads': loads, 'touch': touch}, depth, tilts)

    def seat_audit(self):
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        where = {key: None for key in self.ids}
        for sid, key in world['seats'].items():
            if key is not None:
                where[key] = sid
        return {'weight_seat': where, 'held': self.held, 'releasing': self.releasing,
                'tilt_deg': {k: round(t, 2) for k, t in tilts.items()},
                'carrier': {'s_mm': round(self.carrier['s_m'] * 1000, 4), 'v_mm_s': round(self.carrier['v_m_s'] * 1000, 4)},
                'contacts': dict(getattr(self, 'last_summary', {}))}

    def at_rest(self):
        """At rest: the kit's rule (the weights still, no key held) and the carrier still on its track."""
        return super().at_rest() and (self.carrier is None or abs(self.carrier['v_m_s']) < 0.001)

    # --- stepping: the moving parts follow the bench's state ------------------------------------------------
    def step(self, target, velocity=None):
        self._carrier_pose()
        self._detent()
        out = super().step(target, velocity)
        if self.bench is not None:
            self._drive()
        return out

    def _poses(self):
        B, bench = self.B, self.bench
        st = bench.strip_state()
        tz = self._table_z()
        frame = B.mount_frame(tz, st['w_a'], st['s_a'])
        z, L = st['z'], max(st['L'], 0.02)
        poses = B.strip_poses(tz, st['seg_w'], st['seg_s']) + [B.in_frame(frame, 0.0, 0.0, 0.0)] + \
            B.blade_poses(frame, z, L) + [B.track_pose(frame, z, L), B.yoke_pose(frame, z), B.stinger_pose(tz, st['w_f'])]
        return frame, poses

    def _drive(self):
        frame, poses = self._poses()
        if self.kin_view is None:
            from isaacsim.core.prims import RigidPrim
            view = RigidPrim(list(self.kin_paths), name='absorber_kinematic', reset_xform_properties=False)
            view.initialize()
            self.kin_view = view
            order = [str(p) for p in view.prim_paths]
            self.kin_order = np.array([self.kin_paths.index(p) for p in order])
        self.frame, self.poses = frame, poses
        data = np.asarray(poses, np.float32)[self.kin_order]
        self.kin_view._physics_view.set_kinematic_targets(data, np.arange(len(data), dtype=np.int32))

    def observation(self):
        frames = super().observation()
        self.last_frames = frames
        return frames

    def _redraw(self):
        if self.bench is None:
            return
        A, B = self.asset, self.B
        tz = self._table_z()
        if self.poses is not None:
            key = np.round(np.asarray(self.poses)[:, [0, 2, 4]], 6).tobytes()
            if key != self.poses_drawn:
                A.set_render_poses(self.stage, self.kin_paths, self.poses, tz)
                self.poses_drawn = key
                self.history_stale = True
            last = self.poses[B.SEGMENTS - 1]
            tip = (B.ACCEL[2], B.BEAM_Y, last[2] + B.STRIP[2] / 2 + B.ACCEL[1])
            if self.cable_drawn is None or abs(tip[2] - self.cable_drawn) > 2e-4:
                end = A.C._xy(B.ANALYZER['xy'], *A.CABLE_END_UV)
                A.set_cable(self.stage, B.cable_points(tip, (end[0], end[1], tz + 0.030)), tz)
                self.cable_drawn = tip[2]
        analyzer = self.bench.analyzer_text()
        t = self.bench.t_s
        running = analyzer[2]
        if analyzer != self.analyzer_drawn and (not running or self.analyzer_drawn is None or not self.analyzer_drawn[2]
                                                or t - self.analyzer_at >= ANALYZER_REFRESH_S):
            self.analyzer_drawn, self.analyzer_at = analyzer, t
        text = (self.analyzer_drawn or analyzer, self.bench.valve_text())
        if text != getattr(self, 'text_drawn', None):
            self.lcd_drawn = A.show(self.stage, text[0], text[1], self.lcd_drawn)
            self.text_drawn = text
            self._reset_history()

    def drawn_displays(self):
        """What the analyzer and the valve LCD show now (as drawn)."""
        if getattr(self, 'text_drawn', None) is None:
            return None
        (peak, hz, run), valve = self.text_drawn
        return {'analyzer': {'peak': peak, 'hz': hz, 'run': run}, 'valve_lcd': valve}
