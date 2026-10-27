"""Hand-operated controls of the coupled bench (USD authoring and in-scene displays; no dynamics, no policy).

World frame: the robot stands at world (0.95, 0.35) on the +Y side facing -Y; world +X is the robot's left. The fixed
cameras look toward -Y from above, so world +X is image left in them.

- Catch carriages (one per slider, between the sliders): a carriage rides on the far rail (the rail away from the
  robot, world Y -0.16), offset outward so that it stands beside the slider's path, never in it: a free slider
  passes it without touching. Its X offset from its zero mark is read on a millimetre scale strip on the inner side
  of the rail (world Y -0.114) by a red index pointer; zero is where the carriage stands when its slider is at
  encoder zero and hooked. A blue CLAMP push button on the carriage top toggles the clamp (amber lamp lit =
  clamped). A clamped carriage is fixed on its rail; an unclamped one slides with a friction of CATCH_FRICTION_N.
- Spring hook: a steel pin at the slider's inner end (in the gap between the slider and its carriage) meets a pawl
  in the carriage's slider-side corner (HOOK_* in controls.py). The hook is a D6 joint between slider and carriage
  (no collision between them) whose X drive is on (stiff) while hooked and off (no force at all) otherwise.
- Recorder keypad (robot side, right of the robot): 12 keys, 1-9, CLR, 0, REC, on a console.
- Recorder display (far side, raised, between the CH1 and CH2 position displays): state word and SET / T / NO fields.
- The damping lever is benchmark/core/damper_lever.py, placed by DAMPER_LEVER below.
"""
from __future__ import annotations

import math

SLIDER_CENTERS = (0.43, 0.97)
SLIDER_HALF_X = 0.12
SLIDER_Z = 0.99
RAIL_FAR_Y = -0.16
RAIL_TOP_Z = 0.8875

# --- catch carriages ---------------------------------------------------------------------------------------------
SLIDER_HALF_Y = 0.14
GRIP_POST_DY = -0.19                        # grip post centre from the slider centre along Y (asset.py Handle)
GRIP_POST_TOP_Z = 1.19                      # grip post top (slider body z 0.99 + 0.13 + 0.07)
RELEASE_POST_XY = (0.70, -0.23)             # release post at rest (asset.py ReleaseBar)
CATCH_SIZE = (0.070, 0.050, 0.060)          # collider block above the far rail, X x Y x Z
CATCH_Y = -0.175                            # its +Y face 10 mm outside the slider housing's -Y face
CATCH_GAP = 0.004                           # carriage end face to slider end face (along X) at zero
CATCH_Z = RAIL_TOP_Z + 0.002 + CATCH_SIZE[2] / 2
CATCH_TOP_Z = CATCH_Z + CATCH_SIZE[2] / 2
CATCH_MASS_KG = 0.15
CATCH_TRAVEL_M = 0.062                      # rail stops, either side of the zero mark
CATCH_FRICTION_N = 0.6                      # sliding friction of an unclamped carriage (drive force limit)
CATCH_FRICTION_DAMPING = 1.0e6              # N s/m below the force limit: it stays where it is left
# Catch zero centres: CH1 to +X of slider 1's inner (right) face, CH2 to -X of slider 2's inner (left) face.
CATCH_ZERO_X = (SLIDER_CENTERS[0] + SLIDER_HALF_X + CATCH_GAP + CATCH_SIZE[0] / 2,
                SLIDER_CENTERS[1] - SLIDER_HALF_X - CATCH_GAP - CATCH_SIZE[0] / 2)
CATCH_SIDE = (+1, -1)                       # direction from the slider to its carriage along X
BUTTON_DX = 0.015                           # clamp button centre from the carriage centre, away from the slider
LAMP_DX = -0.022                            # lamp centre from the carriage centre, toward the slider
HOOK_K = 20000.0                            # hook drive while hooked (N/m, N s/m, N)
HOOK_C = 120.0
HOOK_MAX_N = 200.0
# Scale strip on the base beside the far rail, read by the index pointer at the carriage centre.
SCALE_Y = -0.114
SCALE_WIDTH = 0.022
SCALE_TOP_Z = 0.894
SCALE_RANGE_MM = 60

# --- push buttons (keypad keys, clamp buttons): the chemistry key mechanics ---------------------------------------
CAP_LIFT = 0.010
CAP_H = 0.010
KEY_SIZE = 0.022
CLAMP_CAP_R = 0.010
TRAVEL_M = 0.009
SWITCH_DOWN_M = 0.004
SWITCH_UP_M = 0.002
CAP_MASS_KG = 0.010
SPRING_N_PER_M = 400.0
SPRING_PRELOAD_M = 0.001
SPRING_DAMPING = 4.0

# --- recorder keypad console --------------------------------------------------------------------------------------
# Recorder panel (keypad console + display), placeable by any bench through build_recorder_panel. Panel-local frame
# of the keypad: x = world X and y = world Y at yaw 0, origin at the console's centre; first column (1, 4, 7, CLR) at
# local +x, first row (1 2 3) at local -y.
KEY_DX = (0.048, 0.0, -0.048)
KEY_DY = (-0.056, -0.022, 0.012, 0.046)
KEY_ROWS = (('1', '2', '3'), ('4', '5', '6'), ('7', '8', '9'), ('clear', '0', 'rec'))
KEY_PRINT = {'clear': 'CLR', 'rec': 'REC'}
CONSOLE_HALF = (0.080, 0.080)
TABLE_TOP_Z = 0.76
# Coupled bench placement of the panel.
CONSOLE_TOP_Z = 0.90
KEYPAD_CENTRE_XY = (0.628, 0.268)
KEY_COL_X = tuple(round(KEYPAD_CENTRE_XY[0] + d, 4) for d in KEY_DX)     # (0.676, 0.628, 0.580)
KEY_ROW_Y = tuple(round(KEYPAD_CENTRE_XY[1] + d, 4) for d in KEY_DY)     # (0.212, 0.246, 0.280, 0.314)
CONSOLE_BOX = ((0.548, 0.708), (0.188, 0.348))

# --- recorder display ---------------------------------------------------------------------------------------------
DISPLAY_CENTRE = (0.70, -0.37, 1.30)        # coupled bench placement (yaw 0: the face looks along +Y)
DISPLAY_HOUSING = (0.25, 0.025, 0.155)
STATE_WORDS = ('IDLE', 'READY', 'REC', 'DONE', 'ERR')
STATE_HEIGHT = 0.040
FIELD_HEIGHT = 0.038
FIELDS = {'set': (0.075, 3), 't': (-0.005, 3), 'no': (-0.085, 2)}    # display-local x of each field, digit count

# --- damping lever (benchmark/core/damper_lever.py) ---------------------------------------------------------------
# Light detent (1 N mid-way): this bench's fingers press with at most 3 N each, so a firm parallel grip on the
# post must carry the carriage across by friction or by a finger's push.
DAMPER_LEVER = dict(name='Damper', detent0=(1.15, -0.30), direction=(0.0, 1.0), spacing=0.04, top_z=0.90,
                    stiffness=50.0, damping=5.0,
                    labels=('OFF', 'ON'))
DAMPING_ON_PER_S = 8.0                     # extra viscous deceleration rate per unit speed while ON (as the old brake)


def recorder_key_centres(centre_xy, yaw_deg=0.0):
    """{key: (x, y)} world key centres of a recorder keypad placed at `centre_xy` with yaw `yaw_deg` (about +Z)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    out = {}
    for r, row in enumerate(KEY_ROWS):
        for k, key in enumerate(row):
            dx, dy = KEY_DX[k], KEY_DY[r]
            out[key] = (centre_xy[0] + c * dx - s * dy, centre_xy[1] + s * dx + c * dy)
    return out


def recorder_key_top_z(top_z):
    """World Z of a key cap's top at rest, for a console whose top is at `top_z`."""
    return top_z + CAP_LIFT + CAP_H


def key_centres():
    """{key: (x, y)} keypad key centres of the coupled bench in the world."""
    return {key: (KEY_COL_X[c], KEY_ROW_Y[r]) for r, row in enumerate(KEY_ROWS) for c, key in enumerate(row)}


def catch_button_xy(index, offset_m=0.0):
    return CATCH_ZERO_X[index] + offset_m + CATCH_SIDE[index] * BUTTON_DX, CATCH_Y


def key_rest_top_z():
    return recorder_key_top_z(CONSOLE_TOP_Z)


def clamp_rest_top_z():
    return CATCH_TOP_Z + CAP_LIFT + CAP_H


def _ensure_glyphs():
    from pendulum_asset import _FONT
    for char, bits in (('F', (31, 16, 16, 30, 16, 16, 16)), ('H', (17, 17, 17, 31, 17, 17, 17)),
                       ('Z', (31, 1, 2, 4, 8, 16, 31))):
        _FONT.setdefault(char, bits)


def _physics_helpers(stage, contact_material):
    """(rigid, collider, spring_cap) authoring helpers bound to `stage` and the bench's contact material."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    def physical(prim):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(contact_material, UsdShade.Tokens.weakerThanDescendants, 'physics')

    def rigid(prim, mass, iterations=(16, 8)):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateSolverPositionIterationCountAttr().Set(iterations[0])
        api.CreateSolverVelocityIterationCountAttr().Set(iterations[1])
        api.CreateSleepThresholdAttr().Set(0.0)
        api.CreateLinearDampingAttr().Set(0.0)
        api.CreateAngularDampingAttr().Set(0.0)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass)

    def collider(geom):
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        api = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        api.CreateContactOffsetAttr().Set(.0005)
        api.CreateRestOffsetAttr().Set(0.0)
        physical(geom.GetPrim())
        return geom

    def spring_cap(cap_path, body0_path, local0, world_centre, make_shape, mass=CAP_MASS_KG):
        """A cap body on a spring-returned vertical prismatic joint (travel TRAVEL_M) to body0 (None: a kinematic
        anchor at the cap's rest pose)."""
        cap = UsdGeom.Xform.Define(stage, cap_path)
        cap.AddTranslateOp().Set(Gf.Vec3d(*world_centre))
        rigid(cap.GetPrim(), mass)
        make_shape(cap_path)
        if body0_path is None:
            body0_path = cap_path + 'Anchor'
            anchor = UsdGeom.Xform.Define(stage, body0_path)
            anchor.AddTranslateOp().Set(Gf.Vec3d(*world_centre))
            UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr().Set(True)
            m = UsdPhysics.MassAPI.Apply(anchor.GetPrim())
            m.CreateMassAttr().Set(1.0)
            m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0))
            local0 = (0, 0, 0)
        joint = UsdPhysics.PrismaticJoint.Define(stage, cap_path + 'Joint')
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(cap_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateAxisAttr().Set('Z')
        joint.CreateLowerLimitAttr().Set(-TRAVEL_M)
        joint.CreateUpperLimitAttr().Set(0.0)
        joint.CreateCollisionEnabledAttr().Set(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr().Set('force')
        drive.CreateStiffnessAttr().Set(SPRING_N_PER_M)
        drive.CreateDampingAttr().Set(SPRING_DAMPING)
        drive.CreateTargetPositionAttr().Set(SPRING_PRELOAD_M)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(100.0)
        return cap_path

    return rigid, collider, spring_cap


def _placed_xform(stage, path, translate, yaw_deg):
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if yaw_deg:
        xform.AddRotateZOp().Set(float(yaw_deg))
    return path


def build_recorder_panel(stage, path, *, centre_xy, top_z, contact_material, yaw_deg=0.0, base_z=TABLE_TOP_Z,
                         display_centre, display_yaw_deg=0.0, spec_text='40-180 S'):
    """Author a recorder front panel under `path`: the keypad console (`path`/Keypad, keys `path`/Keypad/Key_<k> for
    k in 0-9, clear, rec; push caps on spring prismatic Z joints, switch at SWITCH_DOWN_M / SWITCH_UP_M) centred at
    `centre_xy` with its top at `top_z` and its base at `base_z`, turned by `yaw_deg` about +Z; and the display
    (`path`/RecorderDisplay with State, Set, T, No) centred at `display_centre`, its face looking along +Y turned by
    `display_yaw_deg`. Materials go under `path`/Materials. Nothing else (no catches, hooks or lever).
    Key positions: recorder_key_centres(centre_xy, yaw_deg), recorder_key_top_z(top_z).
    Runtime: update_recorder_display(stage, path, RecorderPanel.display)."""
    from pendulum_asset import _box, _digits, _flat_text, _material, _text
    _ensure_glyphs()
    rigid, collider, spring_cap = _physics_helpers(stage, contact_material)
    mats = {
        'ink': _material(stage, path + '/Materials/ScaleInk', (.006, .012, .02)),
        'white': _material(stage, path + '/Materials/ControlsPrint', (.93, .95, .97)),
        'console': _material(stage, path + '/Materials/Console', (.12, .16, .22), .3),
        'key': _material(stage, path + '/Materials/Key', (.80, .81, .83)),
        'rec_key': _material(stage, path + '/Materials/RecKey', (.80, .10, .08)),
        'led_off': _material(stage, path + '/Materials/RecorderLedOff', (.002, .004, .003)),
        'housing': _material(stage, path + '/Materials/RecorderHousing', (.006, .012, .02)),
    }
    out = {'keys': {}}

    # --- keypad console (keypad-local coordinates, z in the world) -------------------------------------------------
    keypad = _placed_xform(stage, path + '/Keypad', (centre_xy[0], centre_xy[1], 0.0), yaw_deg)
    hx, hy = CONSOLE_HALF
    collider(_box(stage, keypad + '/Console', (0.0, 0.0, (base_z + top_z) / 2), (2 * hx, 2 * hy, top_z - base_z),
                  mats['console']))
    _flat_text(stage, keypad + '/Title', 'RECORDER', (0.0, hy - .0085, top_z + .0004), .008, mats['white'])
    for r, row in enumerate(KEY_ROWS):
        for k, key in enumerate(row):
            x, y = KEY_DX[k], KEY_DY[r]
            centre = (x, y, top_z + CAP_LIFT + CAP_H / 2)
            material = mats['rec_key'] if key == 'rec' else mats['key']
            ink = mats['white'] if key == 'rec' else mats['ink']

            def key_shape(cap_path, material=material):
                collider(_box(stage, cap_path + '/Cap', (0, 0, 0), (KEY_SIZE, KEY_SIZE, CAP_H), material))

            cap = spring_cap(f'{keypad}/Key_{key}', None, None, centre, key_shape)
            text = KEY_PRINT.get(key, key)
            _flat_text(stage, cap + '/Print', text, (0, 0, CAP_H / 2 + .0004), .010 if len(text) == 1 else .0055, ink)
            _box(stage, f'{keypad}/Key_{key}Collar', (x, y, top_z + (CAP_LIFT - .001) / 2),
                 (KEY_SIZE + .004, KEY_SIZE + .004, CAP_LIFT - .001), mats['ink'])
            out['keys'][key] = cap

    # --- display (display-local coordinates) -----------------------------------------------------------------------
    display = _placed_xform(stage, path + '/RecorderDisplay', display_centre, display_yaw_deg)
    _box(stage, display + '/Housing', (0.0, 0.0, 0.0), DISPLAY_HOUSING, mats['housing'])
    face_y = DISPLAY_HOUSING[1] / 2 + .001
    _text(stage, display + '/Title', 'RECORDER', (0.0, face_y + .001, .066), .010, mats['white'])
    out['state_matrix'] = _matrix(stage, display + '/State', (0.0, face_y + .001, .030), 5, STATE_HEIGHT, path)
    for name, (fx, count) in FIELDS.items():
        label = {'set': 'SET S', 't': 'T S', 'no': 'NO'}[name]
        _text(stage, display + f'/{name.title()}Label', label, (fx, face_y + .001, -.004), .008, mats['white'])
        out[f'field_{name}'] = _digits(stage, display + f'/{name.title()}', (fx, face_y + .001, -.033), count,
                                       FIELD_HEIGHT, mats['led_off'])
    if spec_text:
        _text(stage, display + '/Spec', spec_text, (0.0, face_y + .001, -.066), .0065, mats['white'])
    out['display'] = display
    return out


# ------------------------------------------------------------------------------------------------------------------
def build_controls(stage, root, contact_material):
    """Author the coupled bench's catches, hooks, clamp buttons, scales, recorder panel (build_recorder_panel at the
    coupled placement) and damper lever under `root`."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade
    from pendulum_asset import _FONT, _box, _cylinder, _digits, _flat_text, _material, _text
    from damper_lever import build_lever
    _ensure_glyphs()

    mats = {
        'catch': _material(stage, root + '/Materials/Catch', (.20, .22, .25), .5, .35),
        'cap': _material(stage, root + '/Materials/ClampCap', (.10, .25, .72)),
        'pointer': _material(stage, root + '/Materials/CatchIndex', (.85, .06, .05), emission=.2),
        'paper': _material(stage, root + '/Materials/ScalePaper', (.95, .95, .89)),
        'ink': _material(stage, root + '/Materials/ScaleInk', (.006, .012, .02)),
        'white': _material(stage, root + '/Materials/ControlsPrint', (.93, .95, .97)),
        'steel': _material(stage, root + '/Materials/HookSteel', (.55, .58, .62), .7, .25),
    }

    rigid, collider, spring_cap = _physics_helpers(stage, contact_material)

    out = {'catches': [], 'clamp_caps': [], 'lamps': [], 'keys': {}, 'hooks': [], 'catch_joints': []}

    # --- catches, hooks, clamp buttons, scales ----------------------------------------------------------------------
    for i in range(2):
        ch = f'CH{i + 1}'
        x0, side = CATCH_ZERO_X[i], CATCH_SIDE[i]
        path = f'{root}/Catch{i + 1}'
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(x0, CATCH_Y, CATCH_Z))
        rigid(body.GetPrim(), CATCH_MASS_KG, (32, 4))
        sx, sy, sz = CATCH_SIZE
        collider(_box(stage, path + '/Block', (0, 0, 0), CATCH_SIZE, mats['catch']))
        # Outer skirt down beside the rail (render only); pawl notch in the slider-side corner of the +Y face.
        _box(stage, path + '/Skirt', (0, -sy / 2 + .004, -sz / 2 - .022), (sx, .008, .044), mats['catch'])
        _box(stage, path + '/Pawl', (-side * (sx / 2 - .006), sy / 2 + .0004, .0), (.012, .0008, .020), mats['ink'])
        _text(stage, path + '/Label', ch, (0, sy / 2 + .0008, -.012), .016, mats['white'])
        _flat_text(stage, path + '/TopLabel', ch, (-side * .004, sy / 2 - .0065, sz / 2 + .0004), .008, mats['white'])
        _flat_text(stage, path + '/ClampLabel', 'CLAMP', (side * BUTTON_DX, -sy / 2 + .0055, sz / 2 + .0004), .006,
                   mats['white'])
        # Index pointer: an arm from the +Y face over the scale strip, its red tip at the carriage centre.
        arm_y0, arm_y1 = sy / 2, SCALE_Y - SCALE_WIDTH / 2 + .007 - CATCH_Y
        _box(stage, path + '/IndexArm', (0, (arm_y0 + arm_y1) / 2, SCALE_TOP_Z + .0035 - CATCH_Z),
             (.006, arm_y1 - arm_y0, .003), mats['catch'])
        _box(stage, path + '/IndexTip', (0, SCALE_Y - SCALE_WIDTH / 2 + .0055 - CATCH_Y, SCALE_TOP_Z + .0007 - CATCH_Z),
             (.0012, .011, .0006), mats['pointer'])
        # Lamp: own material so its colour can change.
        lamp = path + '/Lamp'
        _cylinder(stage, lamp, (LAMP_DX * side, 0, sz / 2 + .002), .006, .004,
                  _material(stage, root + f'/Materials/ClampLamp{i + 1}', (1.0, .55, .05), emission=.6))
        out['lamps'].append(lamp)
        # Clamp button on top, on a spring joint to the carriage.
        local0 = (BUTTON_DX * side, 0, sz / 2 + CAP_LIFT + CAP_H / 2)
        centre = (x0 + local0[0], CATCH_Y, CATCH_Z + local0[2])

        def clamp_shape(cap_path):
            collider(_cylinder(stage, cap_path + '/Cap', (0, 0, 0), CLAMP_CAP_R, CAP_H, mats['cap']))

        cap = spring_cap(f'{root}/Clamp{i + 1}Cap', path, local0, centre, clamp_shape)
        _cylinder(stage, path + '/ClampCollar', (local0[0], 0, sz / 2 + (CAP_LIFT - .001) / 2), CLAMP_CAP_R + .003,
                  CAP_LIFT - .001, mats['ink'])
        out['clamp_caps'].append(cap)
        # Rail joint (world -> carriage), X; the drive is the carriage friction; clamping pins the limits.
        joint = UsdPhysics.PrismaticJoint.Define(stage, f'{root}/CatchJoint{i + 1}')
        joint.CreateBody0Rel().SetTargets([])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(x0, CATCH_Y, CATCH_Z))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateAxisAttr().Set('X')
        joint.CreateLowerLimitAttr().Set(0.0)          # clamped at zero at the start
        joint.CreateUpperLimitAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr().Set('force')
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(CATCH_FRICTION_DAMPING)
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(CATCH_FRICTION_N)
        out['catches'].append(path)
        out['catch_joints'].append(str(joint.GetPath()))
        # Hook: D6 joint slider -> carriage, only its X drive acts (hooked at the start).
        slider = f'{root}/Slider{i + 1}'
        sx0 = SLIDER_CENTERS[i]
        hook = UsdPhysics.Joint.Define(stage, f'{root}/HookJoint{i + 1}')
        hook.CreateBody0Rel().SetTargets([Sdf.Path(slider)])
        hook.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        hook.CreateLocalPos0Attr().Set(Gf.Vec3f(x0 - sx0, CATCH_Y, CATCH_Z - SLIDER_Z))
        hook.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        hook.CreateCollisionEnabledAttr().Set(False)
        hd = UsdPhysics.DriveAPI.Apply(hook.GetPrim(), 'transX')
        hd.CreateTypeAttr().Set('force')
        hd.CreateStiffnessAttr().Set(HOOK_K)
        hd.CreateDampingAttr().Set(HOOK_C)
        hd.CreateTargetPositionAttr().Set(0.0)
        hd.CreateTargetVelocityAttr().Set(0.0)
        hd.CreateMaxForceAttr().Set(HOOK_MAX_N)
        out['hooks'].append(str(hook.GetPath()))
        # Hook pin at the slider's inner end, in the gap beside the carriage's pawl (render only).
        gap_y = (-SLIDER_HALF_Y + CATCH_Y + CATCH_SIZE[1] / 2) / 2
        _box(stage, slider + '/HookPin', (side * (SLIDER_HALF_X + .002), gap_y, CATCH_Z - SLIDER_Z),
             (.008, .007, .014), mats['steel'])
        # Millimetre scale strip: ticks from its far edge, numbers every 20 mm on its near side.
        _box(stage, f'{root}/Scale{i + 1}/Strip', (x0, SCALE_Y, SCALE_TOP_Z - .003),
             (2 * SCALE_RANGE_MM / 1000 + .012, SCALE_WIDTH, .006), mats['paper'])
        _box(stage, f'{root}/Scale{i + 1}/Bracket', (x0, SCALE_Y, (0.76 + SCALE_TOP_Z - .006) / 2),
             (2 * SCALE_RANGE_MM / 1000 + .012, .004, SCALE_TOP_Z - .006 - 0.76), mats['catch'])
        _tick_mesh(stage, f'{root}/Scale{i + 1}/Ticks', x0, mats['ink'])
        for mm in range(-SCALE_RANGE_MM, SCALE_RANGE_MM + 1, 20):
            _flat_text(stage, f'{root}/Scale{i + 1}/N{mm + 100}', str(mm), (x0 + mm / 1000, SCALE_Y + .0065, SCALE_TOP_Z + .0004),
                       .008, mats['ink'])
        _flat_text(stage, f'{root}/Scale{i + 1}/Unit', ch + ' MM', (x0 + side * (SCALE_RANGE_MM / 1000 + .028), SCALE_Y, SCALE_TOP_Z + .0004),
                   .007, mats['ink'])
        _box(stage, f'{root}/Scale{i + 1}/Unitboard', (x0 + side * (SCALE_RANGE_MM / 1000 + .028), SCALE_Y, SCALE_TOP_Z - .003),
             (.042, SCALE_WIDTH, .006), mats['paper'])

    # --- recorder panel (keypad console + display) ------------------------------------------------------------------
    panel = build_recorder_panel(stage, root, centre_xy=KEYPAD_CENTRE_XY, top_z=CONSOLE_TOP_Z,
                                 contact_material=contact_material, display_centre=DISPLAY_CENTRE,
                                 spec_text='120 HZ  2 CH  40-180 S')
    out['keys'] = panel['keys']
    out['state_matrix'], out['display'] = panel['state_matrix'], panel['display']
    out.update({k: v for k, v in panel.items() if k.startswith('field_')})

    # --- damping lever ---------------------------------------------------------------------------------------------
    out['lever'] = build_lever(stage, root + '/DamperLever', contact_material=contact_material, **DAMPER_LEVER)
    return out


def _tick_mesh(stage, path, x0, material):
    """1 mm ticks from the strip's far edge: 4 mm long, 7 mm at 5 mm, 11 mm at 10 mm; one mesh."""
    from pxr import UsdGeom, UsdShade
    y_far = SCALE_Y - SCALE_WIDTH / 2
    z = SCALE_TOP_Z + .0003
    points, counts, indices = [], [], []
    for mm in range(-SCALE_RANGE_MM, SCALE_RANGE_MM + 1):
        length = .011 if mm % 10 == 0 else (.007 if mm % 5 == 0 else .004)
        w = .00045 if mm % 5 else .0007
        x = x0 + mm / 1000
        k = len(points)
        points += [(x - w / 2, y_far, z), (x + w / 2, y_far, z), (x + w / 2, y_far + length, z), (x - w / 2, y_far + length, z)]
        counts.append(4)
        indices += [k, k + 1, k + 2, k + 3]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _matrix(stage, path, centre, count, height, root):
    """A row of `count` 5x7 dot-matrix cells facing +Y; first cell at the largest X (image left). Each dot is a box
    with its own LED material (colour carries the state; geometry never changes)."""
    from pxr import UsdGeom
    from pendulum_asset import _box, _material
    UsdGeom.Xform.Define(stage, path)
    pitch = height / 7.0
    width = (count * 6 - 1) * pitch
    for c in range(count):
        for row in range(7):
            for col in range(5):
                x = centre[0] + width / 2 - (c * 6 + col) * pitch - pitch / 2
                z = centre[2] + height / 2 - row * pitch - pitch / 2
                _box(stage, f'{path}/C{c}_{row}_{col}', (x, centre[1], z), (pitch * .92, .0005, pitch * .92),
                     _material(stage, f'{path}/C{c}_{row}_{col}/LED', (.002, .004, .003)))
    return path


# --- runtime visuals (colour changes only) ------------------------------------------------------------------------
_SHADERS = {}
_LED = {}
LED_ON = (.22, 1.0, .58)
LED_OFF = (.002, .004, .003)
LAMP_ON = (1.0, .55, .05)
LAMP_OFF = (.07, .05, .04)


def _shader(stage, material_path):
    from pxr import Gf, Sdf, UsdShade
    key = (str(stage.GetRootLayer().identifier), material_path)
    if key not in _SHADERS:
        shader = UsdShade.Shader(stage.GetPrimAtPath(material_path + '/Shader'))
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        _SHADERS[key] = shader
    return key, _SHADERS[key]


def _set_colour(stage, material_path, colour, emission):
    from pxr import Gf
    key, shader = _shader(stage, material_path)
    value = (tuple(colour), emission)
    if _LED.get(key) == value:
        return
    _LED[key] = value
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(*colour))
    shader.GetInput('emissiveColor').Set(Gf.Vec3f(*(emission * c for c in colour)))


def set_lamp(stage, root, index, on):
    _set_colour(stage, root + f'/Materials/ClampLamp{index + 1}', LAMP_ON if on else LAMP_OFF, .6 if on else 0.0)


def set_state_word(stage, root, word):
    from pendulum_asset import _FONT
    text = (word or '').upper().ljust(5)[:5]
    path = root + '/RecorderDisplay/State'
    for c, char in enumerate(text):
        bits = _FONT.get(char, _FONT[' '])
        for row in range(7):
            for col in range(5):
                on = bool(bits[row] & (1 << (4 - col)))
                _set_colour(stage, f'{path}/C{c}_{row}_{col}/LED', LED_ON if on else LED_OFF, 1.0 if on else 0.0)


def set_field(stage, root, name, value):
    """A numeric field: value None shows blank; integers right-aligned."""
    from pendulum_asset import _SEGMENTS
    count = FIELDS[name][1]
    text = ' ' * count if value is None else str(int(value))[-count:].rjust(count)
    path = root + f'/RecorderDisplay/{name.title()}'
    for index, char in enumerate(text):
        for segment in 'abcdefg':
            prim = f'{path}/D{index}_{segment}'
            _seg_colour(stage, prim, segment in _SEGMENTS[char])


def _seg_colour(stage, prim_path, on):
    """Seven-segment cell colours (each segment gets its own LED material on first use, as instrument_visuals)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    from pendulum_asset import _material
    key = (str(stage.GetRootLayer().identifier), prim_path)
    if key not in _SHADERS:
        prim = stage.GetPrimAtPath(prim_path)
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set('inherited')
        material = _material(stage, prim_path + '/LEDMaterial', LED_OFF)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        shader = UsdShade.Shader(stage.GetPrimAtPath(prim_path + '/LEDMaterial/Shader'))
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        _SHADERS[key] = shader
    if _LED.get(key) == on:
        return
    _LED[key] = on
    shader = _SHADERS[key]
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(*(LED_ON if on else LED_OFF)))
    shader.GetInput('emissiveColor').Set(Gf.Vec3f(*((1.0 * c for c in LED_ON) if on else (0, 0, 0))))


def update_recorder_display(stage, root, shown):
    set_state_word(stage, root, shown['state'])
    set_field(stage, root, 'set', shown['set_s'])
    set_field(stage, root, 't', shown['t_s'])
    set_field(stage, root, 'no', shown['number'])
