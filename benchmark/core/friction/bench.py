"""Independent, robot-accessible incline bench; development asset, not accepted.

Only the powered ramp hinge and retracting gate have actuator drives. The two test
sliders (A and B) are free rigid bodies, physically recovered by the robot after each run.
"""
import math

ROOT = '/World/InclineLab'
# The ramp axis runs 0.37 m in front of the robot base (y = 0.35). At 0.31 m (first layout, y = .04) the gate
# housing of the raised ramp met the robot's shoulder (links 1 and 2).
# The hinge stands 2 cm right of the robot base (world x 0.97); 15 cm further right (first layouts) the hinge end
# covered the only table area that the public left camera sees beside the robot, where the console now stands.
ORIGIN = (.97, -.02, .98)
# 3 cm further from the sensor bridge than the first layout (-.46): a set-down error of a few centimetres along the
# ramp is harmless on both sides.
LOAD_S = -.43
RAMP_LENGTH = .68
DECK_WIDTH = .32  # Real support width; allow free lateral drift before guards.
BLOCK_DIMS = (.16, .12, .01)
BLOCK_MASS = .25
# The plate is diagonal to travel; the square hollow grip stays aligned with
# the ramp and Panda jaws. This orientation is visible and public CAD geometry.
BLOCK_BODY_YAW_DEG = 45.
BODY_HALF_SPAN = (.08 + .06) / math.sqrt(2.)
GATE_S = LOAD_S + BODY_HALF_SPAN + .010
RECOVERY_S = .195 - .018 / 2 - BODY_HALF_SPAN
LOADING_PAD_DIMS = (.23, .23, .016)
# Both upstream fixtures must clear the full diagonal loading footprint.
# Mount the sensor on the back face of its crossbar, away from descending plate.
SENSOR_BRIDGE_CENTER = (-.587, 0., .115)
SENSOR_BRIDGE_DIMS = (.024, DECK_WIDTH+.05, .025)
POSITION_SENSOR_CENTER = (-.610, 0., .105)
POSITION_SENSOR_DIMS = (.028, .052, .042)
# Desired midpoint of finger link origins, not the visible handle center.
# Panda fingertips extend 54mm below this midpoint. Keep the palm above the
# 70mm handle top while letting the fingertips overlap its upper 24mm.
# 18 mm above the handle top (z = .070): where the public top-down grasp skill brings the finger midpoint,
# as on the other benches. (First value .100; the skill then stopped 12 mm short of the capture radius.)
GRIP_LOCAL = (0., 0., .088)
GRIP_DIMS = (.032, .032, .065)
# Beside the upper end on the robot's side (world (0.50, 0.44), 0.46 m from the robot base, seen by the public left
# camera); used only while the ramp is level. The first position, beyond the upper end (0.70 m away), put the arm
# across the sensor bridge.
PARK_LOCAL = (-.47, .46, -.105)
# Console right of the hinge end, which never rises: world x 1.29 and 1.37, y -0.06 and -0.14, 0.5 to 0.6 m from the
# robot base, in the public left camera's view (the arm blocks the right camera's view of it). Earlier places:
# beside the upper end, where the forearm met the raised ramp while pressing (seed 0, ramp at 30 deg), and behind
# the robot base line, which no public camera sees.
BUTTONS = {'raise': (.32, -.04, -.055), 'hold': (.40, -.04, -.055),
           'lower': (.32, -.12, -.055), 'release': (.40, -.12, -.055)}
CONSOLE_LOCAL = (.36, -.08)
GATE_SPEED_M_S = .8
# Recorder front panel: the shared panel (coupled_oscillator.controls_asset.build_recorder_panel), in world
# coordinates. The keypad console stands on the table in front of the push-button console, on the robot's side
# (world X 1.26 to 1.42, Y 0.01 to 0.17, 0.5 m from the robot base): the arm reaches it from the robot's side and
# never crosses the ramp. The display stands on a post behind the push-button console, facing the robot; the public
# left camera sees it beside the ramp's lower end.
RECORDER_PATH = '/World/InclineRecorder'
TABLE_TOP_Z = .82
KEYPAD_CENTRE_XY = (1.34, .09)
KEYPAD_TOP_Z = .90
DISPLAY_CENTRE = (1.32, -.32, 1.10)
DISPLAY_HOUSING = (.25, .025, .155)      # controls_asset.DISPLAY_HOUSING
RECORD_S_RANGE = (5, 120)                # record lengths REC accepts, whole seconds
# Two sliders: A and B, the same plate and handle, printed "A" and "B" on the plate top; their
# undersides differ (each slider has its own contact material). Slider A starts on the first loading pad (PARK_LOCAL),
# B on the second one, 2 cm beyond it on the far side from the robot, beside the ramp's upper end (world (0.25, 0.37),
# 0.70 m from the robot base, inside the top-down reach and at the right edge of the public left camera; y 0.37 keeps
# the pad clear of the base plate, which ends at y 0.24).
SLIDERS = ('A', 'B')
PARK_LOCALS = {'A': PARK_LOCAL, 'B': (-.72, .39, -.105)}
PAD_CENTERS_XY = {k: (ORIGIN[0]+p[0], ORIGIN[1]+p[1]) for k, p in PARK_LOCALS.items()}
PAD_CENTER_XY = PAD_CENTERS_XY['A']
PAD_TOP_Z = ORIGIN[2]-.119+LOADING_PAD_DIMS[2]/2
# The deck's own material takes no part in the slider contact: with the 'min' combine mode each slider-deck pair
# has the slider's coefficients (both below 1).
DECK_MU = 1.0
LETTER_HEIGHT_M = .034         # printed letter on the plate top, at both ends of the plate's long axis
LETTER_OFFSET_M = .052         # along the plate's long axis from its centre (clear of the 32 mm handle)


def slider_path(name):
    return ROOT+'/Slider'+name


def pad_path(name):
    return ROOT+'/LoadingPad'+name


def block_projection_half_extent(direction):
    """Project actual plate and grip geometry onto a direction in body frame."""
    x, y, z = direction
    angle = math.radians(BLOCK_BODY_YAW_DEG)
    c, s = math.cos(angle), math.sin(angle)
    plate = sum(abs(a)*d/2 for a,d in zip((c*x+s*y,-s*x+c*y,z), BLOCK_DIMS))
    grip = abs(.0375*z) + sum(abs(a)*d/2 for a,d in zip(direction,GRIP_DIMS))
    return max(plate,grip)


def block_mass_properties():
    """COM and principal inertia of the real rotated plate + axis-aligned grip."""
    mb,mh=.248,.002
    zc=mh*.0375/BLOCK_MASS
    dx,dy,dz=BLOCK_DIMS; hx,hy,hz=GRIP_DIMS
    # The square handle is isotropic in XY, so the combined principal axes
    # rotate with the plate. Both parallel-axis shifts are only along Z.
    inertia=(mb*(dy*dy+dz*dz)/12+mb*zc*zc+mh*(hy*hy+hz*hz)/12+mh*(.0375-zc)**2,
             mb*(dx*dx+dz*dz)/12+mb*zc*zc+mh*(hx*hx+hz*hz)/12+mh*(.0375-zc)**2,
             mb*(dx*dx+dy*dy)/12+mh*(hx*hx+hy*hy)/12)
    half=math.radians(BLOCK_BODY_YAW_DEG)/2
    return (0.,0.,zc), inertia, (math.cos(half),0.,0.,math.sin(half))


def build_bench(stage, sliders):
    """sliders: {'A': (mu_s, mu_k), 'B': (mu_s, mu_k)}, each slider's underside against the deck."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    if set(sliders) != set(SLIDERS):
        raise ValueError('friction coefficients for sliders A and B required')

    def material(path, ms, mk):
        mat = UsdShade.Material.Define(stage, path)
        props = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        props.CreateStaticFrictionAttr(float(ms)); props.CreateDynamicFrictionAttr(float(mk))
        props.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('min')
        return mat

    UsdGeom.Xform.Define(stage, ROOT).AddTranslateOp().Set(Gf.Vec3d(*ORIGIN))
    contact = material(ROOT+'/SurfaceContact', DECK_MU, DECK_MU)
    undersides = {k: material(ROOT+'/SliderContact'+k, *sliders[k]) for k in SLIDERS}
    structure = material(ROOT+'/StructuralContact', .8, .6)
    dark, metal, white = (.07,.10,.13), (.40,.45,.50), (.83,.85,.80)

    def box(path, pos, dims, color, mat=structure, collision=True, yaw_deg=0.):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if yaw_deg:
            half=math.radians(yaw_deg)/2
            cube.AddOrientOp().Set(Gf.Quatf(math.cos(half),Gf.Vec3f(0,0,math.sin(half))))
        cube.AddScaleOp().Set(Gf.Vec3f(*dims))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            props = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            props.CreateContactOffsetAttr(.001); props.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, materialPurpose='physics')
        return cube

    def rigid(path, pos, mass):
        prim = UsdGeom.Xform.Define(stage, path)
        prim.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(mass)
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim.GetPrim())
        px.CreateSleepThresholdAttr(0.); px.CreateStabilizationThresholdAttr(0.)
        px.CreateSolverPositionIterationCountAttr(32); px.CreateSolverVelocityIterationCountAttr(16)
        px.CreateEnableCCDAttr(True)
        return prim.GetPrim()

    def world(local):
        return Gf.Vec3f(*(a+b for a,b in zip(ORIGIN, local)))

    # Tabletop is Z=.82. Plate bottom rests on feet whose bottoms touch it.
    box(ROOT+'/BasePlate', (-.22,-.06,-.144), (.91,.64,.016), dark)
    for i,x in enumerate((-.61,.16)):
        for j,y in enumerate((-.23,.16)):
            box(ROOT+f'/Foot{i}{j}', (x,y,-.156), (.055,.05,.008), dark)
    for side in (-1,1):
        box(ROOT+f'/HingePillar{side+1}', (0,side*(DECK_WIDTH/2+.04),-.061), (.07,.045,.150), metal)
        cap = UsdGeom.Cylinder.Define(stage, ROOT+f'/HingeBearing{side+1}')
        cap.CreateAxisAttr('Y'); cap.CreateRadiusAttr(.036); cap.CreateHeightAttr(.05)
        cap.AddTranslateOp().Set(Gf.Vec3d(0,side*(DECK_WIDTH/2+.04),0))
        cap.CreateDisplayColorAttr([Gf.Vec3f(*dark)])

    ramp = ROOT+'/Ramp'; rigid(ramp, (0,0,0), 2.)
    # Contact deck, underside beams and guards rotate together around the axle.
    # One continuous collider through the receiving pocket avoids internal
    # coplanar seams striking the leading edge of the free block.
    box(ramp+'/Deck', (-.20,0,-.009), (.80,DECK_WIDTH,.018), (.28,.40,.49), contact)
    # Hinge constraint represents the shaft/bearing contact; a second solid
    # shaft collider would interpenetrate the fixed bearing supports.
    box(ramp+'/Axle', (0,0,-.023), (.035,DECK_WIDTH+.11,.035), metal, collision=False)
    for side in (-1,1):
        box(ramp+f'/Beam{side+1}', (-.26,side*.085,-.032), (.67,.025,.028), dark)
        box(ramp+f'/Guard{side+1}', (-.20,side*(DECK_WIDTH/2+.008),.022), (.80,.016,.080), metal)
    # End pocket stays on the ramp; lower the apparatus before recovering block.
    # The white pocket marking is visual only; Deck supplies its support.
    box(ramp+'/CatchDeck', (.105,0,.00015), (.19,DECK_WIDTH,.0002), white, collision=False)
    box(ramp+'/CatchStop', (.195,0,.032), (.018,DECK_WIDTH,.082), (.18,.24,.21))
    for i in range(31):
        s = -.55+i*.02
        box(ramp+f'/Ruler{i}', (s,-DECK_WIDTH/2+.014,.0008), (.001,.014 if i%5 else .025,.0008), white, collision=False)
    # Optical heads and their mast are attached to the moving reference frame.
    box(ramp+'/SensorBridge', SENSOR_BRIDGE_CENTER, SENSOR_BRIDGE_DIMS, dark)
    for side in (-1,1):
        box(ramp+f'/SensorMast{side+1}', (-.587,side*(DECK_WIDTH/2+.014),.05), (.025,.02,.13), dark)
    box(ramp+'/PositionSensor', POSITION_SENSOR_CENTER, POSITION_SENSOR_DIMS, (.8,.60,.08))

    hinge = UsdPhysics.RevoluteJoint.Define(stage, ROOT+'/Hinge')
    hinge.CreateBody1Rel().SetTargets([Sdf.Path(ramp)])
    hinge.CreateAxisAttr('Y'); hinge.CreateLocalPos0Attr(world((0,0,0)))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1)); hinge.CreateLocalRot1Attr(Gf.Quatf(1))
    hinge.CreateLowerLimitAttr(0.); hinge.CreateUpperLimitAttr(45.)
    motor = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular')
    motor.CreateStiffnessAttr(100000.); motor.CreateDampingAttr(3000.)
    motor.CreateMaxForceAttr(10000.); motor.CreateTargetPositionAttr(0.)

    # Vertical gate: it withdraws without pushing the block downhill.
    gate = ROOT+'/Gate'; rigid(gate, (GATE_S,0,.025), .04)
    box(gate+'/Bar', (0,0,0), (.008,DECK_WIDTH-.02,.045), white)
    guide = UsdPhysics.PrismaticJoint.Define(stage, ROOT+'/GateGuide')
    guide.CreateBody0Rel().SetTargets([Sdf.Path(ramp)])
    guide.CreateBody1Rel().SetTargets([Sdf.Path(gate)])
    guide.CreateAxisAttr('Z'); guide.CreateLocalPos0Attr(Gf.Vec3f(GATE_S,0,.025))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0))
    guide.CreateLocalRot0Attr(Gf.Quatf(1)); guide.CreateLocalRot1Attr(Gf.Quatf(1))
    guide.CreateLowerLimitAttr(0.); guide.CreateUpperLimitAttr(.12)
    gate_motor = UsdPhysics.DriveAPI.Apply(guide.GetPrim(),'linear')
    gate_motor.CreateStiffnessAttr(3000.); gate_motor.CreateDampingAttr(25.)
    gate_motor.CreateMaxForceAttr(40.); gate_motor.CreateTargetPositionAttr(0.)
    # Bridge supports the vertical gate above the sliding and loading region.
    # Keep the cross-beam beyond the payload's vertical loading footprint.
    # The cross-beam stands 7 cm downstream of the block (first layout: 2 cm), clear of a block lowered from above.
    box(ramp+'/GateHousing', (LOAD_S+.18,0,.18), (.025,DECK_WIDTH+.07,.025), dark)
    for side in (-1,1):
        box(ramp+f'/GatePillar{side+1}', (LOAD_S+.18,side*(DECK_WIDTH/2+.026),.075), (.025,.018,.20), dark)
        box(ramp+f'/GateGuideArm{side+1}', (LOAD_S+.1375,side*(DECK_WIDTH/2+.026),.18), (.085,.018,.025), dark)

    from circuit_apparatus import label
    half = math.radians(BLOCK_BODY_YAW_DEG)
    for name in SLIDERS:
        block = slider_path(name)
        # Beside the ramp on the robot's side, reachable from above; each slider on its own pad.
        park = PARK_LOCALS[name]
        rigid(block, park, BLOCK_MASS)
        box(block+'/Body', (0,0,0), BLOCK_DIMS, (.16,.49,.31), undersides[name], yaw_deg=BLOCK_BODY_YAW_DEG)
        box(block+'/Grip', (0,0,.0375), GRIP_DIMS, (.72,.025,.38))
        # Printed letter (render only) on the plate top at both ends of its long axis, reading with its top away from
        # the robot while the slider stands as it starts.
        for end, sign in (('0', 1.), ('1', -1.)):
            label(stage, block+'/Letter'+end, name, (sign*LETTER_OFFSET_M*math.cos(half), sign*LETTER_OFFSET_M*math.sin(half)),
                  BLOCK_DIMS[2]/2+.0004, height_m=LETTER_HEIGHT_M, align='centre', colour=(.96,.96,.93), up=(0.,-1.))
        # Public construction: .248kg body + .002kg hollow handle. Explicit inertia
        # prevents the lightweight handle from being treated as solid body material.
        com, inertia, axes = block_mass_properties()
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(block))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(axes[0],Gf.Vec3f(*axes[1:])))
        box(pad_path(name), (park[0],park[1],-.119), LOADING_PAD_DIMS, white)
        box(ROOT+'/LoadingStand'+name, (park[0],park[1],-.1435), (.13,.10,.033), dark)
    # Console bottom rests on the same Z=.82 tabletop; button bases meet its top.
    box(ROOT+'/ControlConsole',(*CONSOLE_LOCAL,-.148),(.17,.17,.024),dark)
    colors={'raise':(.14,.62,.26),'hold':(.82,.30,.12),'lower':(.12,.35,.70),'release':white}
    for name,pos in BUTTONS.items():
        path=ROOT+'/Button_'+name; rigid(path,pos,.03)
        box(path+'/Cap',(0,0,0),(.055,.055,.008),colors[name])
        box(path+'/Stem',(0,0,-.014),(.012,.012,.028),metal,collision=False)
        box(ROOT+'/ButtonBase_'+name,(pos[0],pos[1],-.106),(.08,.08,.06),dark)
        joint=UsdPhysics.PrismaticJoint.Define(stage,ROOT+'/ButtonGuide_'+name)
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateAxisAttr('Z');joint.CreateLocalPos0Attr(world(pos))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1));joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateLowerLimitAttr(-.012);joint.CreateUpperLimitAttr(0.)
        spring=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
        spring.CreateStiffnessAttr(800.);spring.CreateDampingAttr(4.)
        spring.CreateMaxForceAttr(15.);spring.CreateTargetPositionAttr(0.)
    # Recorder keypad and display (shared panel); the display's housing hangs on a post with a foot.
    from coupled_oscillator.controls_asset import build_recorder_panel
    panel=build_recorder_panel(stage,RECORDER_PATH,centre_xy=KEYPAD_CENTRE_XY,top_z=KEYPAD_TOP_Z,contact_material=structure,
        base_z=TABLE_TOP_Z,display_centre=DISPLAY_CENTRE,spec_text=f'240 HZ  {RECORD_S_RANGE[0]}-{RECORD_S_RANGE[1]} S')
    x,y,z=DISPLAY_CENTRE;post_y=y-DISPLAY_HOUSING[1]/2-.011;bottom=z-DISPLAY_HOUSING[2]/2
    box(RECORDER_PATH+'/DisplayStand/Post',(x,post_y,(TABLE_TOP_Z+bottom+.04)/2),(.03,.02,bottom+.04-TABLE_TOP_Z),dark)
    box(RECORDER_PATH+'/DisplayStand/Foot',(x,post_y,TABLE_TOP_Z+.006),(.14,.09,.012),dark)
    return dict(sliders={k: slider_path(k) for k in SLIDERS},ramp=ramp,gate=gate,motor=motor,gate_motor=gate_motor,panel=panel)
