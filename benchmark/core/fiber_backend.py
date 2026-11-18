"""Fiber coupling physics device (fiber_coupling): the chemistry backend with the fiber apparatus. Objects the gripper
carries: the five lens cartridges, the alignment cap and the target; the five adjuster knobs are graspable bodies on
their revolute joints (fiber_knob), turned by a held hand; POWER is pressed.

Every physics tick:
- the holders' seats are read as the chemistry kit reads them (a holder standing upright on a well floor; the keyed
  wells let a cartridge in only square to the seat; one turned around is seen from its pose);
- the knobs' angles are read from the physics (fiber_knob.KnobReader);
- POWER's cap depth (the kit's push button);
- every rendered frame, PhysX raycasts along the beam's three legs (nominal path) find anything not of the optics in
  the beam: the robot, a holder that does not stand in its seat, anything set down in the beam (the seated target, cap
  and cartridge are the optics' own; the model passes the light through them as it does);
- the bench (fiber_bench) advances one tick; when the screws moved, the mirrors' front plates (kinematic) are tilted to
  the model's mirror tilts and the focus carriage (kinematic) is moved to the fiber's focus travel.
At each rendered frame the spots where the beam meets a surface (the beam itself is invisible in clean
air), the pilot lamp and the laser's exit, and the meter's LCD are drawn as the bench has them (the LCD
shows the reading of the last 0.2 s refresh).
"""
from __future__ import annotations

import json

import numpy as np

from chem_backend import ChemBackend

MIRROR_SCATTER = 0.06       # a mirror's faint scatter spot (relative brightness)
FIBER_SPOT_MIN_M = 0.25e-3  # the focused spot on the fiber port's face drawn at least this large (a glint)
RAY_START_M = 0.003          # rays start and stop this far from the mirrors (the beam meets them at their faces)
LOG_EVERY_FRAMES = 3         # private log of the bench at 10 Hz (physics.jsonl)


def _rot(q_wxyz):
    w, x, y, z = [float(v) for v in q_wxyz]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class FiberBackend(ChemBackend):
    ASSET = 'fiber_apparatus'

    def _build(self, asset):
        bench = self.configuration['bench']
        built = asset.build(self.stage, {'lens_seats': bench['lens_seats'], 'cap_seat': bench['cap_seat'],
                                         'target_seat': bench['target_seat']})
        self.knob_paths = dict(built['knobs'])
        self.kin_paths = dict(built['kinematic'])
        self.z0 = float(built['z0'])
        self.knob_reader = None
        self.kin_bodies = None
        self.screw_drawn = None
        self.blocked = None                       # (path coordinate, hit path) of the last raycasts
        self.drawn_state = {}
        self.meter_drawn = None
        self.sq = None
        self.contact_log = []
        return {'objects': built['objects'], 'buttons': built['buttons'], 'seats': built['seats']}

    def _support_paths(self, key, built):
        A = self.asset
        if str(key).startswith('knob_'):
            return [A.ROOT + '/M1/BackPlate', A.ROOT + '/M2/BackPlate', A.ROOT + '/LeadScrewHousing']
        return list(built['seats'].values()) + A.support_paths()

    # --- physics readers -------------------------------------------------------------------------------------------
    def knob_angles(self):
        if self.knob_reader is None:
            import fiber_knob as K
            self.knob_reader = K.KnobReader(self.knob_paths)
        a = self.knob_reader.update()
        return np.array([a[k] for k in self.asset.KNOB_KEYS])

    def _kinematic(self):
        if self.kin_bodies is None:
            from isaacsim.core.prims import SingleRigidPrim
            self.kin_bodies = {}
            for name, path in self.kin_paths.items():
                body = SingleRigidPrim(path, name=f'fiber_kin_{name.lower()}', reset_xform_properties=False)
                body.initialize()
                self.kin_bodies[name] = body
        return self.kin_bodies

    def _move_optics(self):
        """Front plates to the mirror tilts, the carriage to the focus travel, when the screws moved."""
        bench = self.bench
        if bench is None:
            return
        screw = tuple(round(float(v), 6) for v in bench.adj.screw)
        if screw == self.screw_drawn:
            return
        import fiber_model as M
        bodies = self._kinematic()
        px, py = bench.adj.phis()
        for k, mirror in enumerate(('M1', 'M2')):
            self.asset.set_front_plate(self.stage, mirror, float(px[k]), float(py[k]), bodies[mirror], self.z0)
        travel = float(bench.adj.screw[4]) * M.FOCUS_UM_PER_DEG * 1e-6
        self.asset.set_carriage(self.stage, travel, bodies['carriage'], self.z0)
        self.screw_drawn = screw

    def _reversed(self, positions, orientations, seated):
        oid = seated.get('coupler:lens')
        if oid is None:
            return False
        k = self.ids.index(oid)
        y_axis = _rot(orientations[k])[:, 1]
        return bool(y_axis[1] < 0.0)

    def _raycast(self, seated):
        """First thing in the beam that is not the optics' own: (path coordinate, prim path) or None."""
        import carb
        import fiber_model as M
        from omni.physx import get_physx_scene_query_interface
        A = self.asset
        if self.sq is None:
            self.sq = get_physx_scene_query_interface()
        own = [A.front_path('M1'), A.front_path('M2'), A.CARRIAGE_PATH]
        own += [A.holder_path(v) for v in seated.values() if v is not None]
        own += [str(p) for p in self.seat_paths.values()]
        legs = ((RAY_START_M, M.L1 - RAY_START_M), (M.L1 + RAY_START_M, M.L2 - RAY_START_M),
                (M.L2 + RAY_START_M, A.P_FIBER - 0.0005))
        for p0, p1 in legs:
            a, b = A.path_point(p0), A.path_point(p1)
            a[2] += self.z0
            b[2] += self.z0
            d = b - a
            L = float(np.linalg.norm(d))
            d /= L
            hits = []

            def report(hit):
                path = str(hit.rigid_body or hit.collision)
                coll = str(hit.collision)
                if not any(path.startswith(o) or coll.startswith(o) for o in own):
                    hits.append((float(hit.distance), coll))
                return True
            self.sq.raycast_all(carb.Float3(*a.tolist()), carb.Float3(*d.tolist()), L, report)
            if hits:
                dist, path = min(hits)
                return (p0 + dist, path)
        return None

    def _world(self, positions, orientations):
        world, depth, tilts = super()._world(positions, orientations)
        seated = {sid: world['seats'].get(sid) for sid in ('coupler:lens', 'coupler:cap', 'bench:target')}
        world['knob_deg'] = self.knob_angles()
        world['reversed'] = self._reversed(positions, orientations, seated)
        if self.tick_index % self.frame_ticks == 0 or self.blocked is None:
            try:
                self.blocked = self._raycast(seated) or ()
            except Exception as exc:                    # pragma: no cover - logged, the beam is taken as clear
                self.audit_file.write(json.dumps({'event': 'raycast_error', 'tick': self.tick_index,
                                                  'error': str(exc)}) + '\n')
                self.blocked = ()
        world['blocked_p'] = self.blocked[0] if self.blocked else None
        world['blocked_by'] = self.blocked[1] if self.blocked else None
        return world, depth, tilts

    def step(self, target, velocity=None):
        out = super().step(target, velocity)
        self._move_optics()
        return out

    def seat_audit(self):
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        where = {k: None for k in self.ids if not k.startswith('knob_')}
        for sid, key in world['seats'].items():
            if key is not None:
                where[key] = sid
        return {'holder_seat': where, 'held': self.held, 'reversed': world['reversed'],
                'knob_deg': [round(float(v), 4) for v in world['knob_deg']],
                'blocked_p': world['blocked_p'], 'blocked_by': world['blocked_by'],
                'tilt_deg': {k: round(t, 2) for k, t in tilts.items() if not k.startswith('knob_')}}

    # --- drawing ---------------------------------------------------------------------------------------------------
    def _redraw(self):
        bench = self.bench
        A = self.asset
        light = bench.light
        changed = False
        if self.drawn_state.get('power') != bench.power:
            A.set_power(self.stage, bench.power)
            self.drawn_state['power'] = bench.power
            changed = True
        spots = self._spots_to_draw(light)
        for name in A.SPOT_NAMES:
            drawn = spots.get(name)
            k = None if drawn is None else (tuple(round(float(v), 5) for v in drawn['xyz']), round(drawn['w'], 5),
                                            round(drawn['level'], 3), drawn.get('rot_z', 0.0))
            if k != self.drawn_state.get(name):
                A.set_spot(self.stage, name, drawn)
                self.drawn_state[name] = k
                changed = True
        shown = A.meter_text(bench.reading_uW)
        if shown != self.meter_drawn:
            self.meter_drawn = A.show_meter(self.stage, shown, self.meter_drawn)
            changed = True
        if changed:
            self._reset_history()

    def _spots_to_draw(self, light):
        """{surface: spot} where the beam lands (ROOT frame; set_spot's format): a faint scatter on each mirror it
        reaches, the target card and the cap (the bench's spots), whatever else blocks it (the raycasts), and past an
        open coupler either the lens cell it clips (beyond the clear aperture) or the fiber port's face, where the
        lens focuses it (the ray through the lens, fiber_model's geometry)."""
        import fiber_model as M
        from fiber_bench import beam_radius
        A = self.asset
        bench = self.bench
        out = {}
        end = light.get('beam_end_p')
        if not light['power'] or end is None:
            return out
        inst = bench.inst
        blocked = self.blocked[0] if self.blocked else None
        reach = end

        def on_beam(p):
            if p < M.L2:
                return A.path_point(p)
            o = bench.offset_at(p)
            return A.path_point(p) + np.array([o[0], 0.0, o[1]])

        r2 = 1.0 / np.sqrt(2.0)
        for name, p, normal, rot in (('m1', M.L1, (r2, -r2), -135.0), ('m2', M.L2, (-r2, -r2), 135.0)):
            if reach > p - 1e-6:
                xyz = on_beam(p) + 0.0008 * np.array([normal[0], normal[1], 0.0])
                out[name] = {'xyz': xyz, 'w': beam_radius(inst, p), 'level': MIRROR_SCATTER, 'rot_z': rot}
        level = 1.0
        for name, face_y in (('target', A.TARGET_FACE_Y), ('cap', A.CAP_FACE_Y)):
            spot = light.get(f'{name}_spot')
            if spot is not None:
                ox, oy = spot['offset_m']
                out[name] = {'xyz': (A.BEAM_X + ox, face_y, A.BEAM_Z + oy), 'w': spot['w_m'], 'level': spot['level']}
        level *= float(light.get('target_T', 1.0))          # 1 without the target
        if blocked is not None and blocked <= end + 1e-9:          # (the bench's end already stops at it)
            p = float(blocked)
            rot = 180.0 if p < M.L1 else (90.0 if p < M.L2 else 0.0)
            lv = level if p > bench.p_target else 1.0
            if lv > 1e-4:
                out['block'] = {'xyz': on_beam(p), 'w': beam_radius(inst, p), 'level': lv, 'rot_z': rot}
            return out
        lens = light.get('lens')
        if bench.cap_on() or level <= 1e-4 or end < A.P_FIBER - 0.05:
            return out
        if lens is None:
            p = A.P_FIBER
            out['fiber'] = {'xyz': on_beam(p) + np.array([0.0, 0.0002, 0.0]), 'w': beam_radius(inst, p),
                            'level': level}
            return out
        d = A.lens_disc_y(lens) - A.LENS_SEAT_Y
        cell_y = A.LENS_SEAT_Y + (-d if bench.reversed else d) + A.RING_T / 2
        p_cell = A.path_of_y(cell_y)
        o = bench.offset_at(p_cell)
        w_cell = beam_radius(inst, p_cell)
        if float(np.hypot(*o)) > A.LENS_CA_R:
            out['lens'] = {'xyz': (A.BEAM_X + o[0], cell_y, A.BEAM_Z + o[1]), 'w': w_cell, 'level': level}
            return out
        off, ang = np.asarray(light['ray_lens_m'], float), np.asarray(light['ray_angle_rad'], float)
        f = float(inst['f'][lens])
        dz = float(light['dz_m'] or 0.0)
        at_fiber = off + (f + dz) * (ang - off / f)
        w_focus = M.LAM * f / (np.pi * w_cell)
        zr = np.pi * w_focus ** 2 / M.LAM
        w_fiber = w_focus * np.sqrt(1.0 + (dz / zr) ** 2)
        travel = float(bench.adj.screw[4]) * M.FOCUS_UM_PER_DEG * 1e-6
        out['fiber'] = {'xyz': (A.BEAM_X + at_fiber[0], A.FIBER_Y - travel + 0.0002, A.BEAM_Z + at_fiber[1]),
                        'w': max(float(w_fiber), FIBER_SPOT_MIN_M), 'level': level}
        return out

    def after_step(self, timers, tick):
        super().after_step(timers, tick)
        bench = self.bench
        if bench is None or tick % (self.frame_ticks * LOG_EVERY_FRAMES):
            return
        light = bench.light
        row = {'event': 'fiber', 'tick': int(tick), 'power': bench.power, 'seats': dict(bench.seats),
               'reversed': bench.reversed, 'knob_deg': [round(float(v), 3) for v in bench.knob_deg],
               'screw_deg': [round(float(v), 3) for v in bench.adj.screw],
               'meter': self.meter_drawn, 'reading_uW': bench.reading_uW,
               'true_uW': round(1000.0 * light['meter_fraction'], 4), 'core': round(light['core'], 5),
               'target_T': round(light['target_T'], 4), 'blocked_p': bench.blocked_p,
               'blocked_by': self.blocked[1] if self.blocked else None,
               'beam_end_p': light['beam_end_p'], 'ray_lens_mm': [round(v * 1000, 3) for v in light['ray_lens_m']],
               'knob_tilt_deg': {k: round(v, 3) for k, v in (self.knob_reader.tilt_deg if self.knob_reader else {}).items()},
               'knob_drift_mm': {k: round(v, 3) for k, v in (self.knob_reader.drift_mm if self.knob_reader else {}).items()}}
        self.audit_file.write(json.dumps(row, default=str) + '\n')

    def drawn_meter(self):
        if self.meter_drawn is None:
            return None
        return {'text': self.meter_drawn, 'uW': self.asset.meter_value(self.meter_drawn)}

    def close(self):
        if self.closed:
            return
        bench = self.bench
        super().close()
        if bench is not None:
            (self.private / 'fiber_bench.json').write_text(json.dumps(bench.private_record(), indent=1, default=str))
