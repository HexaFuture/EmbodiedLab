"""active_filter on the circuit board: a 7 x 6 board whose far row is a ground rail (pads joined by
printed copper), the frequency-response analyzer (keypad, screen; its signal output strapped to the IN pad, its
ground to the rail), resistor pieces 3k3..33k and capacitor pieces 1n0..100n (two of each value), eight wires, two
op-amp modules (three-node blocks: IN+, IN-, OUT on three neighbouring pads in a line) and the analyzer's probe block,
in two trays. The circuit and the analyzer are filter_bench (no Isaac); the scene is filter_scene.

The profile follows circuit_grid.variants (Profile.apply sets the bench geometry on the layout module; one experiment
per bench process): besides the blocks and tray it sets the board size and position and the tray seats (two trays,
`tray_seats`), and resets board's socket cache. The backend is the variants backend with the analyzer's screen and no
levers; the episode has export_sweeps instead of the logger export.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import numpy as np

from . import layout as L
from .variants import Profile, register

import filter_model as M

# --- geometry (world metres; layout.py's frame) ---------------------------------------------------------------------
COLS, ROWS = 7, 6
NODE_X0, NODE_Y0 = 0.720, 0.050          # node (1, 1); rows run toward -y (away from the robot)
GROUND_ROW = ROWS                        # the far row: the ground rail
IN_NODE = (1, 3)
GROUND_NODE = (1, GROUND_ROW)            # where the analyzer's ground strap lands

ANALYZER_BOX = ((0.425, 0.652), (-0.312, -0.020), 0.045)       # x range, y range, height
ANALYZER_SCREEN = ((0.452, 0.636), (-0.300, -0.145))           # screen glass on the top (x range, y range)
ANALYZER_KEY_ROW_X = (0.495, 0.530, 0.565, 0.600, 0.635)       # image rows, top to bottom
ANALYZER_KEY_COL_Y = (-0.115, -0.080, -0.045)                  # image columns, left to right
ANALYZER_KEYS = (('start', 'stop', 'pts'), ('7', '8', '9'), ('4', '5', '6'), ('1', '2', '3'), ('clr', '0', 'sweep'))
ANALYZER_KEY_PRINT = {'start': 'START', 'stop': 'STOP', 'pts': 'PTS', 'clr': 'CLR', 'sweep': 'SWEEP'}
ANALYZER_JACK_Y = {'out': -0.070, 'gnd': -0.250}               # output jacks on the housing's +x face (IN, GND rows)
ANALYZER_KEY_SIZE = 0.022

# tray A (+x of the board): resistors and capacitors; tray B (-x, toward the robot): op-amps, wires, probe
TRAY_A_ROW_X = (1.170, 1.218, 1.266, 1.314, 1.362)
TRAY_A_COL_Y = (0.100, 0.044, -0.012, -0.068, -0.124, -0.180)
TRAY_B_ROW_X = (0.475, 0.523, 0.571, 0.619)
TRAY_B_COL_Y = (0.212, 0.150, 0.088, 0.026)
TRAY_MARGIN = 0.010

PIECES = M.piece_ids()
R_PIECES = [p for p in PIECES if p.startswith('R')]
C_PIECES = [p for p in PIECES if p.startswith('C')]
WIRES = [f'W{k}' for k in range(1, 9)]
OPAMPS = ('OA1', 'OA2')
OPAMP_HANDLE_X = 0.030                    # the module's handle over the middle of its IN- / OUT edge (local x)

R_COLOUR = (0.80, 0.68, 0.48)             # tan
C_COLOUR = (0.12, 0.30, 0.66)             # blue
PROBE_COLOUR = (0.95, 0.80, 0.10)         # yellow
OPAMP_COLOUR = (0.22, 0.23, 0.25)         # dark grey


def _value(p):
    return p[1:].rsplit('_', 1)[0]


BLOCKS = {**{p: ('resistor', _value(p), 'R' + _value(p), 0.040) for p in R_PIECES},
          **{p: ('capacitor', _value(p), 'C' + _value(p), 0.040) for p in C_PIECES},
          **{w: ('wire', '', 'wire', 0.035) for w in WIRES},
          'PROBE': ('probe', 'PROBE', 'probe', 0.040)}
THREE = {'OA1': ('opamp', 'OPAMP 1', 'opamp', 0.080), 'OA2': ('opamp', 'OPAMP 2', 'opamp', 0.080)}
TRAY_A = [PIECES[k:k + 6] for k in range(0, 30, 6)]
TRAY_B = [[None, None, None, None], WIRES[:4], WIRES[4:], ['PROBE', None, None, None]]
OPAMP_WELLS = {'OA1': ((TRAY_B_ROW_X[0], (TRAY_B_COL_Y[0] + TRAY_B_COL_Y[1]) / 2), L.TRAY_YAW_DEG),
               'OA2': ((TRAY_B_ROW_X[0], (TRAY_B_COL_Y[2] + TRAY_B_COL_Y[3]) / 2), L.TRAY_YAW_DEG)}


def tray_seats():
    """{seat label: dict(xy, class, printed, tray)} of both trays (the two-pin wells); label = the starting block."""
    out = {}
    for tray, rows, xs, ys in (('A', TRAY_A, TRAY_A_ROW_X, TRAY_A_COL_Y), ('B', TRAY_B, TRAY_B_ROW_X, TRAY_B_COL_Y)):
        for i, row in enumerate(rows):
            for j, block in enumerate(row):
                if block is None:
                    continue
                out[block] = {'xy': (xs[i], ys[j]), 'class': L.BLOCKS[block][2],
                              'printed': L.BLOCKS[block][1] or 'W', 'tray': tray}
    return out


def tray_extents():
    """{tray: ((x0, x1), (y0, y1))} around its wells (long op-amp wells in tray B)."""
    seats = tray_seats()
    out = {}
    mx, my = L.TRAY_MOUTH_HALF
    grow = opamp_well_grow()
    for tray in ('A', 'B'):
        boxes = [(s['xy'][0] - mx, s['xy'][0] + mx, s['xy'][1] - my, s['xy'][1] + my) for s in seats.values()
                 if s['tray'] == tray]
        if tray == 'B':
            boxes += [(xy[0] - mx, xy[0] + mx, xy[1] - my - grow, xy[1] + my + grow) for xy, _ in OPAMP_WELLS.values()]
        out[tray] = ((min(b[0] for b in boxes) - TRAY_MARGIN, max(b[1] for b in boxes) + TRAY_MARGIN + 0.004),
                     (min(b[2] for b in boxes) - TRAY_MARGIN, max(b[3] for b in boxes) + TRAY_MARGIN))
    return out


def opamp_well_grow():
    return L.BASE3[0] / 2 + (L.TRAY_WELL_HALF[1] - L.BASE[0] / 2) - L.TRAY_WELL_HALF[1]


def analyzer_key_centres():
    return {key: (ANALYZER_KEY_ROW_X[i], ANALYZER_KEY_COL_Y[j]) for i, row in enumerate(ANALYZER_KEYS)
            for j, key in enumerate(row)}


def analyzer_top_z():
    return L.TABLE_Z + ANALYZER_BOX[2]


def rail_edges():
    return [((c, GROUND_ROW), (c + 1, GROUND_ROW)) for c in range(1, COLS)]


def reference_geometry():
    """The bench's design geometry in world metres (internal: for reference controllers and probes; the task spec
    publishes only what each part is). Call with the profile applied."""
    def r(v):
        return [round(float(c), 4) for c in v]
    seats = {b: {'xy_m': r(s['xy']), 'takes': s['class']} for b, s in tray_seats().items()}
    seats.update({b: {'xy_m': r(xy), 'takes': 'opamp'} for b, (xy, _) in OPAMP_WELLS.items()})
    return {
        'board': {'columns': COLS, 'rows': ROWS, 'surface_z_m': round(L.BOARD_TOP_Z, 4),
                  'node_xy_m': {L.node_name(n): r(L.node_xy(n)) for n in L.all_nodes()},
                  'in_node': L.node_name(IN_NODE), 'ground_row': GROUND_ROW},
        'tray': {'seat_floor_z_m': round(L.TABLE_Z + L.TRAY_FLOOR, 4), 'seat_yaw_deg': L.TRAY_YAW_DEG, 'seats': seats},
        'blocks': {b: {'kind': v[0]} for b, v in {**BLOCKS, **THREE}.items()},
        'opamp': {'handle_x_m': OPAMP_HANDLE_X, 'pins_x_m': dict(L.CHANGEOVER_PINS)},
        'analyzer': {'key_xy_m': {k: r(v) for k, v in analyzer_key_centres().items()},
                     'key_top_z_m': round(analyzer_top_z() + L.CAP_LIFT + L.CAP_H, 4),
                     'screen_xy_m': [list(ANALYZER_SCREEN[0]), list(ANALYZER_SCREEN[1])]},
    }


@register
class FilterProfile(Profile):
    recipe = 'circuit_grid_filter'
    blocks = BLOCKS
    three = THREE
    tray_layout = tuple(tuple(r) for r in TRAY_A)          # (layout.tray_seats is replaced by tray_seats above)
    supply = False
    dut = 'filter'
    terminals = {}
    colours = {'resistor': R_COLOUR, 'capacitor': C_COLOUR, 'probe': PROBE_COLOUR}
    ground = GROUND_NODE
    in_node = IN_NODE
    ground_node = GROUND_NODE
    _extra_saved = None

    # --- bench geometry on the layout module ------------------------------------------------------------------
    def apply(self):
        from . import board
        super().apply()
        if self._extra_saved is None:
            self._extra_saved = {k: getattr(L, k) for k in ('COLS', 'ROWS', 'NODE_X0', 'NODE_Y0', 'tray_seats')}
        L.COLS, L.ROWS, L.NODE_X0, L.NODE_Y0 = COLS, ROWS, NODE_X0, NODE_Y0
        L.tray_seats = tray_seats
        board._SOCKETS = None
        return self

    def restore(self):
        from . import board
        super().restore()
        if self._extra_saved is not None:
            for k, v in self._extra_saved.items():
                setattr(L, k, v)
            self._extra_saved = None
        board._SOCKETS = None

    def rail_edges(self):
        return rail_edges()

    def three_seats(self):
        return dict(OPAMP_WELLS)

    # --- instance ---------------------------------------------------------------------------------------------
    def configuration(self, instance):
        from instance_episodes import appearance_seed
        return {'recipe': self.recipe, 'model_seed': int(instance['hidden']['model_seed']),
                'noise_seed': appearance_seed('grid-noise:' + instance['instance_id'])}

    def fixture(self, config):
        return []

    def device(self, config):
        from . import filter_bench as FB
        return {'model_version': M.MODEL_VERSION, 'source_ohm': M.R_SRC, 'source_emf_v_peak': 1.0,
                'probe_ohm': M.PROBE_R, 'probe_f': M.PROBE_C, 'point_s': FB.POINT_S, 'fp_hz': M.FP, 'fs_hz': M.FS,
                'f_lo_hz': M.F_LO, 'in_node': L.node_name(IN_NODE), 'ground_row': GROUND_ROW}

    def make_bench(self, episode_id, config, **kwargs):
        from .filter_bench import FilterBench
        return FilterBench(episode_id, self, config, physics_hz=kwargs.get('physics_hz', 240))

    def build(self, stage):
        from .filter_scene import build
        return build(stage, self)

    def bench_for(self, args, app, instance):
        """Backend and episode of an active_filter instance (circuit_grid.variants.grid_bench_for)."""
        backend = filter_backend(args, app, args.grid_configuration)
        episode = filter_episode(backend, instance, args.output, physics_hz=args.physics_hz,
                                 horizon_ticks=args.horizon_ticks)
        return backend, episode

    def sync_displays(self, backend):
        """The analyzer screen (before the episode's bench exists: the power-on state)."""
        from .filter_bench import Analyzer
        from .filter_scene import show_analyzer
        analyzer = backend.bench.analyzer if backend.bench is not None else Analyzer('', 240)
        return show_analyzer(backend, analyzer.display())

    def public_apparatus(self):
        looks = {'resistor': 'tan body with colour bands, printed with its value in ohm (k = kilo-ohm, 4k7 = 4.7 kilo-ohm)',
                 'capacitor': 'blue film capacitor printed with its nominal value (n = nanofarad, 2n2 = 2.2 nF)',
                 'wire': 'grey, no resistance',
                 'probe': 'the analyzer input: yellow handle printed PROBE, red band at the + end of the base'}
        blocks = {b: {'kind': k, 'printed': p or 'none', 'looks': looks[k], 'tray_seat': b,
                      'polarised': k == 'probe'} for b, (k, p, _, _) in BLOCKS.items()}
        for b, (k, p, _, _) in THREE.items():
            blocks[b] = {'kind': 'opamp', 'printed': p + '; its pins are marked IN+, IN- and OUT',
                         'looks': 'dark grey three-pin module with a handle, pins IN+, IN-, OUT in a line',
                         'tray_seat': b, 'polarised': True}
        seats = {b: {'printed': s['printed'], 'takes': s['class']} for b, s in tray_seats().items()}
        for b in THREE:
            seats[b] = {'printed': THREE[b][1], 'takes': 'opamp'}
        return {
            'board': {'what': 'dark green plate with brass node pads; column numbers and row numbers are printed on '
                              'its edges; the pads of one edge row are joined by a printed copper strip, the ground rail '
                              '(printed GND); one pad is printed IN',
                      'connection': 'everything with a pin or a terminal on one pad is connected there; pads are '
                                    'otherwise insulated from each other, except the ground rail\'s pads, which are all '
                                    'one node'},
            'fixed_terminals': {'analyzer_output': 'red strap from the analyzer\'s signal output to the IN pad',
                                'analyzer_ground': 'black strap from the analyzer\'s ground to the ground rail'},
            'analyzer': {'what': 'grey frequency response analyzer printed FREQUENCY RESPONSE ANALYZER with a screen '
                                 'and a keypad',
                         'keys': [ANALYZER_KEY_PRINT.get(k, k) for row in ANALYZER_KEYS for k in row],
                         'screen': ['gain trace (dB) against log frequency', 'START (Hz)', 'STOP (Hz)', 'PTS',
                                    'ENTRY', 'NO (the sweep shown)', 'RUN lamp', 'ERR lamp']},
            'blocks': blocks,
            'trays': {'what': 'two green-grey trays with one labelled well per block (long wells for the op-amp '
                              'modules)', 'seats': seats},
        }


# --- backend ------------------------------------------------------------------------------------------------------
def _backend_class():
    from .variants import _backend_class as variant_backend_class
    from .board import _rotation, classify, classify_three
    Base = variant_backend_class()

    class FilterBackend(Base):
        """The variants backend with the analyzer: no levers (op-amp modules are plain three-node blocks), either
        long well takes either module, the analyzer screen synced at the frame rate."""

        def _lever_throw(self, block):
            return None

        def _rocker(self):
            return None

        def rocker_angle_deg(self):
            return 0.0

        def placement(self):
            positions, quats = self.poses()
            busy = {self.held, self.releasing}
            out = {}
            for k, b in enumerate(self.block_ids):
                if b in THREE:
                    place = classify_three(positions[k], quats[k], held=b in busy)
                    if place[0] != 'board' and b not in busy:
                        tilt = math.degrees(math.acos(max(-1.0, min(1.0, _rotation(quats[k])[2, 2]))))
                        for well, (xy, _) in OPAMP_WELLS.items():
                            if (math.hypot(positions[k][0] - xy[0], positions[k][1] - xy[1]) <= L.SEAT_HOME_DXY_M
                                    and abs(positions[k][2] - (L.TABLE_Z + L.TRAY_FLOOR)) <= L.SEAT_HOME_DZ_M
                                    and tilt <= L.SEAT_HOME_TILT_DEG):
                                place = ('tray', well, True)
                    out[b] = place
                else:
                    out[b] = classify(b, positions[k], quats[k], held=b in busy)
            return out

        def sync_displays(self):
            from .backend import RESET_HOLD_RENDERS
            key = json.dumps(self.profile.sync_displays(self), sort_keys=True, default=str)
            if key != getattr(self, '_profile_display_key', None):
                self._profile_display_key = key
                self.reset_until = self.render_index + RESET_HOLD_RENDERS
                self.reset_edge = True

    return FilterBackend


def filter_backend(args, app, configuration):
    return _backend_class()(args, app, configuration)


# --- episode ------------------------------------------------------------------------------------------------------
TOOLS = ('get_task_spec', 'get_observation', 'get_robot_state', 'execute_joint_trajectory', 'set_gripper', 'wait',
         'export_sweeps', 'submit_result')


def _text(p):
    if p[0] == 'board':
        return ('board',) + tuple(L.node_name(n) for n in p[1:])
    if p[0] == 'tray':
        return ('tray', p[1], bool(p[2]))
    return ('loose',)


def filter_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin
    from .filter_bench import canonical_sha256

    config = backend.configuration
    profile = backend.profile

    class FilterEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({'apparatus': profile.public_apparatus()})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': backend.held is not None}

        def observation(self):
            value = super().observation()
            bench = backend.bench
            n = len(self.captures) + 1
            eid = f'{self.episode_id}:obs:{n}'
            views = backend.view_pngs
            images = {camera: hashlib.sha256(png).hexdigest() for camera, png in views.items()}
            public = {'evidence_id': eid, 'physics_tick': self.tick, 'sim_time_s': self.tick / physics_hz,
                      'images_sha256': images}
            directory = backend.private / 'evidence'
            directory.mkdir(exist_ok=True)
            for camera, png in views.items():
                (directory / f'obs_{n}_{camera}.png').write_bytes(png)
            self.captures[eid] = {
                'kind': 'screen_capture', 'public': public, 'sha256': canonical_sha256(public),
                'state': bench.observation_state(self.tick),
                'images': {camera: {'file': f'evidence/obs_{n}_{camera}.png', 'sha256': digest}
                           for camera, digest in images.items()}}
            value['evidence_id'] = eid
            return value

        def terminal_state(self):
            bench = backend.bench
            placement = backend.placement()
            seats = {}
            for block, place in placement.items():
                if place[0] == 'tray' and place[2]:
                    seats.setdefault(place[1], []).append(block)
            return {'held': backend.held or backend.releasing,
                    'sweep_running': bench.analyzer.active is not None,
                    'placement': {b: _text(p) for b, p in placement.items()}, 'tray_seats': seats,
                    'board': {b: list(n) for b, n in bench.board_state(placement).items()}}

        def seal_payload(self, submission=None):
            bench = backend.bench
            an = bench.analyzer
            evidence = {}
            for public in an.sweeps:
                if public is None:
                    continue
                eid = public['evidence_id']
                evidence[eid] = {'kind': 'filter_sweep', 'public': deepcopy(public), 'sha256': canonical_sha256(public),
                                 **deepcopy(an.private[eid])}
            evidence.update(deepcopy(self.captures))
            cited = set()
            if isinstance(submission, dict) and isinstance(submission.get('evidence_ids'), list):
                cited = {e for e in submission['evidence_ids'] if isinstance(e, str)}
            directory = backend.private / 'evidence'
            files = [(image['file'], (directory / image['file'].rsplit('/', 1)[-1]).read_bytes())
                     for eid, record in self.captures.items() if eid in cited for image in record['images'].values()]
            placement = backend.placement()
            board = bench.board_state(placement)
            els, probe = bench.netlist(board)
            device = dict(profile.device(config), physics_hz=physics_hz, recipe=profile.recipe,
                          board=deepcopy({b: list(n) for b, n in board.items()}),
                          probe=None if probe is None else {'plus': probe[0], 'minus': probe[1]},
                          signature=canonical_sha256(sorted([b, list(n)] for b, n in board.items())),
                          transitions=deepcopy(bench.transitions))
            return self.terminal_state(), device, evidence, files

        def process_extra(self):
            bench = backend.bench
            return {'display_log': list(bench.display_log), 'analyzer_presses': list(bench.analyzer.presses),
                    'transitions': list(bench.transitions)}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name not in TOOLS or not isinstance(args, dict):
                raise ValueError('unknown circuit board tool')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_sweeps':
                if args:
                    raise ValueError('export_sweeps takes no arguments')
                return backend.bench.analyzer.export()
            if name == 'submit_result':
                terminal, device, evidence, files = self.seal_payload(args)
                return self.freeze_and_seal(args, terminal, device, evidence, files)
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open the gripper and let the blocks settle in their tray seats.
    backend.set_gripper(.08)
    for _ in range(40 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest():
        raise RuntimeError('circuit board blocks failed to settle')
    placement = backend.placement()
    away = [b for b, p in placement.items() if not (p[0] == 'tray' and p[1] == b)]
    if away:
        raise RuntimeError(f'blocks not in their tray seats after set-up: {away}')
    backend.tick_index = 0
    backend.history.clear()
    episode = FilterEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    episode.captures = {}
    backend.horizon_ticks = horizon_ticks
    bench = profile.make_bench(episode.episode_id, config, physics_hz=physics_hz)
    backend.bench = bench
    bench.update(0, placement, None, reason='start')
    return episode
