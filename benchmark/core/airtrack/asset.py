"""Author the airtrack_collision air track; only the gliders' mass and restitution use private values.

Everything is placed in track coordinates under ROOT, which stands at world TRACK_ORIGIN_WORLD (world X = track X +
0.45). The X guides support gravity without axial drag. Hand-operated parts (mechanism.py): two start catches, the
spring launcher behind A's mark (its plunger face is the absorbing left-end bumper), the release lever, the recorder
keypad and display (the shared recorder panel), and the two gate screens (screens.py). The right end has a dead
(absorbing) bumper. No glider drive, position reset or scripted impulse exists in this asset.
"""
from . import mechanism as M

ROOT = "/World/AirtrackLab"
TRACK_ORIGIN_WORLD = (0.45, 0.0, 0.0)
CENTERS = M.MARKS_M
TRACK_LIMITS = (0.045, 0.93)            # backup rail stops (glider centre); the bumpers act first
HEIGHT = 0.99                           # glider centre height (world Z)
DECK_TOP = 0.880
PLUNGER_REST_X = M.FACE_FREE_X_M - 0.010    # plunger body origin (pad centre) at free length
KNOB_Y = 0.085
CATCH_PIN_DX = 0.035                    # catch pin on the glider's +Y face, this far towards +X from its centre
CATCH_Y = 0.059
CATCH_DOWN_Z = 0.9985
CATCH_LIFT_M = 0.008
# Hand controls the robot operates stand on the table on its side of the bench (left of the robot base: keypad;
# right of it: release lever), so that operating them never takes the arm across the track and the glider handles.
KEYPAD_CENTRE = (0.17, 0.27)            # track coordinates (world 0.62, 0.27)
KEYPAD_TOP_Z = 0.90
TABLE_TOP_Z = 0.77
DISPLAY_Y = -0.40
DISPLAY_Z = 1.48
SCREEN_X = {1: 0.15, 2: 0.75}
RECORDER_DISPLAY_X = 0.45
LEVER_TOP_Z = 0.98                      # the lever pedestal stands on the table (TABLE_TOP_Z)
RECORD_SPEC = '5-60 S'


def world(x, y=0.0, z=0.0):
    return (x + TRACK_ORIGIN_WORLD[0], y + TRACK_ORIGIN_WORLD[1], z + TRACK_ORIGIN_WORLD[2])


def lever_info():
    from damper_lever import lever_info as info
    return info(ROOT + "/ReleaseLever", name="Release", detent0=world(*M.LEVER_DETENT0)[:2], direction=(0.0, 1.0),
                spacing=M.LEVER_SPACING_M, labels=M.LEVER_LABELS)


def build_apparatus(stage, params, seed=0):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from .measurement import GATES, FLAG_LENGTH_M
    from .screens import build_screen
    from coupled_oscillator.controls_asset import build_recorder_panel
    from pendulum_asset import _FONT, _flat_text, _text

    for char, bits in (('F', (31, 16, 16, 30, 16, 16, 16)), ('H', (17, 17, 17, 31, 17, 17, 17))):
        _FONT.setdefault(char, bits)
    UsdGeom.Xform.Define(stage, ROOT).AddTranslateOp().Set(Gf.Vec3d(*TRACK_ORIGIN_WORLD))

    def visual(name, color):
        path = ROOT + "/Materials/" + name
        m = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.4)
        m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return m

    def contact(name, friction, restitution):
        m = UsdShade.Material.Define(stage, ROOT + "/Materials/" + name)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(friction)
        api.CreateDynamicFrictionAttr(friction)
        api.CreateRestitutionAttr(restitution)
        # Glider/glider restitution survives; zero-restitution bumpers and the plunger face are genuinely inelastic.
        PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim()).CreateRestitutionCombineModeAttr("min")
        return m

    mats = {n: visual(n, color) for n, color in {"Steel": (.42, .48, .55), "Dark": (.035, .05, .07),
            "A": (.05, .40, .65), "B": (.70, .17, .12), "Grip": (.95, .70, .08), "White": (.9, .92, .94),
            "Rubber": (.02, .02, .02), "Brass": (.72, .55, .22)}.items()}
    collision = contact("CollisionMaterial", 0, float(params["restitution"]))
    grip = contact("ContactMaterial", .8, 0)
    dead = contact("StopMaterial", 0, 0)

    def collide(prim, material):
        UsdPhysics.CollisionAPI.Apply(prim)
        c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        c.CreateContactOffsetAttr(.0005)
        c.CreateRestOffsetAttr(0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    def box(path, pos, dims, appearance, material=None):
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.)
        g.AddTranslateOp().Set(Gf.Vec3d(*pos))
        g.AddScaleOp().Set(Gf.Vec3f(*dims))
        g.CreateExtentAttr([Gf.Vec3f(-.5), Gf.Vec3f(.5)])
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[appearance])
        if material:
            collide(g.GetPrim(), material)
        return g

    def cylinder(path, pos, radius, height, appearance, axis="Z", material=None):
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(radius)
        g.CreateHeightAttr(height)
        g.CreateAxisAttr(axis)
        g.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[appearance])
        if material:
            collide(g.GetPrim(), material)
        return g

    def body(path, pos, mass):
        b = UsdGeom.Xform.Define(stage, path)
        b.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(b.GetPrim()).CreateKinematicEnabledAttr(False)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(b.GetPrim())
        rb.CreateSleepThresholdAttr(0)
        rb.CreateLinearDampingAttr(0)
        rb.CreateAngularDampingAttr(0)
        rb.CreateSolverPositionIterationCountAttr(16)
        rb.CreateSolverVelocityIterationCountAttr(8)
        massapi = UsdPhysics.MassAPI.Apply(b.GetPrim())
        massapi.CreateMassAttr(mass)
        massapi.CreateCenterOfMassAttr(Gf.Vec3f(0))
        massapi.CreatePrincipalAxesAttr(Gf.Quatf(1))
        massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(mass * .0015, mass * .0015, mass * .0015))
        return b

    def rail(path, body_path, pos, axis, limits):
        j = UsdPhysics.PrismaticJoint.Define(stage, path)
        j.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        # A world-anchored joint's localPos0 is in world coordinates, even under the translated apparatus root.
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*world(*pos)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0))
        j.CreateLocalRot0Attr(Gf.Quatf(1))
        j.CreateLocalRot1Attr(Gf.Quatf(1))
        j.CreateLowerLimitAttr(limits[0])
        j.CreateUpperLimitAttr(limits[1])
        return j

    def drive(joint, stiffness, damping, max_force):
        d = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(stiffness)
        d.CreateDampingAttr(damping)
        d.CreateTargetPositionAttr(0)
        d.CreateTargetVelocityAttr(0)
        d.CreateMaxForceAttr(max_force)
        return d

    # --- deck, rail, right bumper ----------------------------------------------------------------------------------
    # Rear edge y .185 clears the robot pedestal (from .190); the deck reaches from behind the launcher (-0.16) to
    # beyond the right bumper and forward to the display frame (-0.42).
    box(ROOT + "/MountingDeck", (.455, -.1175, .865), (1.23, .605, .030), "Dark", dead)
    for i, x in enumerate((-.13, 1.005)):
        for j, y in enumerate((-.35, .14)):
            box(ROOT + f"/RubberFoot{i}{j}", (x, y, .835), (.06, .06, .030), "Dark", dead)
            cylinder(ROOT + f"/DeckBolt{i}{j}", (x, y, .882), .007, .004, "Steel")
    rail_top = HEIGHT - .030 - .0015
    rail_bottom = rail_top - .050
    for i, x in enumerate((.06, .49, .91)):
        h = rail_bottom - DECK_TOP
        box(ROOT + f"/RailPedestal{i}", (x, 0, DECK_TOP + h / 2), (.09, .12, h), "Steel", dead)
        for j, y in enumerate((-.045, .045)):
            cylinder(ROOT + f"/LevelingScrew{i}{j}", (x, y, rail_bottom + .004), .006, .008, "Dark")
    box(ROOT + "/AirPlenumRail", (.49, 0, (rail_top + rail_bottom) / 2), (1.05, .078, .050), "Steel", dead)
    for name, y in (("Near", -.052), ("Far", .052)):
        box(ROOT + "/RailFlange" + name, (.49, y, rail_bottom + .009), (1.05, .026, .018), "Steel", dead)
    for index in range(36):
        cylinder(ROOT + f"/AirOutlet{index}", (-.015 + index * .0285, 0, rail_top + .0002), .0018, .0004, "Dark")
    box(ROOT + "/StopPedestalRight", (.99, 0, .919), (.05, .15, .078), "Steel", dead)
    box(ROOT + "/EndStopRight", (.99, 0, HEIGHT), (.025, .13, .09), "Dark", dead)
    box(ROOT + "/BumperPadRight", (.9765, 0, HEIGHT), (.002, .09, .05), "Rubber")

    # --- gliders ---------------------------------------------------------------------------------------------------
    for i, (x, mass, name) in enumerate(zip(CENTERS, (.2, float(params["mass_B_kg"])), ("A", "B")), 1):
        path = ROOT + f"/Slider{i}"
        body(path, (x, 0, HEIGHT), mass)
        box(path + "/Housing", (0, 0, 0), (.10, .075, .06), name, collision)
        box(path + "/HandleMount", (0, -.08, .015), (.035, .14, .020), "Steel", grip)
        box(path + "/HandleSeat", (0, -.14, .040), (.034, .034, .040), "Steel", grip)
        cylinder(path + "/Handle", (0, -.14, .13), .012, .14, "Grip", material=grip)
        box(path + "/FlagMount", (0, .040, .0325), (.020, .024, .025), "Steel")
        box(path + "/OpticalFlag", (0, .045, .065), (FLAG_LENGTH_M, .008, .045), name)
        cylinder(path + "/CatchPin", (CATCH_PIN_DX, .043, 0), .004, .011, "Steel", axis="Y")
        joint = rail(ROOT + f"/ReturnJoint{i}", path, (x, 0, HEIGHT), "X",
                     (TRACK_LIMITS[0] - x, TRACK_LIMITS[1] - x))
        drive(joint, 0., 0., M.CATCH_MAX_N)          # the catch spring: on only while the glider is caught
        box(ROOT + f"/ResetMark{name}", (x, -.070, rail_bottom + .019), (.004, .028, .002), name)
        # Start catch: a post beside the rail and a spring tongue that drops over the glider's catch pin.
        cx = x + CATCH_PIN_DX
        box(ROOT + f"/CatchPost{name}", (cx, .076, (DECK_TOP + 1.012) / 2), (.012, .012, 1.012 - DECK_TOP), "Dark", dead)
        tongue = UsdGeom.Xform.Define(stage, ROOT + f"/CatchTongue{name}")
        tongue.AddTranslateOp().Set(Gf.Vec3d(cx, CATCH_Y, CATCH_DOWN_Z))
        box(ROOT + f"/CatchTongue{name}/Blade", (0, 0, 0), (.010, .034, .008), "Brass")
        _flat_text(stage, ROOT + f"/CatchLabel{name}", name, (cx, .095, DECK_TOP + .0004), .012, mats["White"])

    # --- photogates (fixed positions) ------------------------------------------------------------------------------
    for gate, planes in GATES.items():
        name = {1: "incoming", 2: "outgoing"}[gate]
        middle = sum(planes) / 2
        box(ROOT + f"/GateFoot_{name}", (middle, .135, .886), (.07, .075, .012), "Dark", dead)
        box(ROOT + f"/GateColumn_{name}", (middle, .135, .982), (.025, .025, .180), "Steel", dead)
        box(ROOT + f"/GateArm_{name}", (middle, .095, 1.100), (.06, .105, .014), "Dark", dead)
        _text(stage, ROOT + f"/GateLabel_{name}", str(gate), (middle, .1485, 1.04), .014, mats["White"])
        for i, x in enumerate(planes):
            box(ROOT + f"/GateBridge_{name}_{i}", (x, .0475, 1.095), (.012, .052, .012), "Dark", dead)
            for side, y in (("Near", .027), ("Far", .068)):
                box(ROOT + f"/Gate_{name}_{i}_{side}", (x, y, 1.067), (.012, .009, .054), "Dark", dead)
                box(ROOT + f"/GateLED_{name}_{i}_{side}", (x - .0062, y, 1.068), (.0004, .004, .006), "White")
    for index in range(11):
        box(ROOT + f"/ScaleTick{index}", (index * .1, -.061, rail_bottom + .019), (.002, .009, .001), "White")

    # --- spring launcher (left end) --------------------------------------------------------------------------------
    box(ROOT + "/LauncherPedestal", (-.093, 0, .92), (.114, .11, .08), "Steel", dead)
    box(ROOT + "/LauncherHousing", (-.09, 0, HEIGHT), (.12, .09, .06), "Dark", dead)
    _flat_text(stage, ROOT + "/LauncherTitle", "LAUNCHER", (-.09, 0, HEIGHT + .0305), .011, mats["White"])
    body(ROOT + "/Plunger", (PLUNGER_REST_X, 0, HEIGHT), M.PLUNGER_MASS_KG)
    box(ROOT + "/Plunger/Pad", (0, 0, 0), (.02, .075, .06), "Dark", dead)
    cylinder(ROOT + "/Plunger/Rod", (-.0675, 0, 0), .006, .115, "Steel", axis="X")
    box(ROOT + "/Plunger/KnobArm", (0, .0625, 0), (.012, .045, .012), "Steel", grip)
    cylinder(ROOT + "/Plunger/Knob", (0, KNOB_Y, .045), .010, .09, "Grip", material=grip)
    j = rail(ROOT + "/LauncherJoint", ROOT + "/Plunger", (PLUNGER_REST_X, 0, HEIGHT), "X",
             (-M.REAR_LIMIT_M, M.FORWARD_LIMIT_M))
    drive(j, M.SPRING_N_PER_M, M.DASHPOT_NS_PER_M, 100.)
    # Notch scale beside the knob path: the knob stands over 0 uncocked and over 1, 2, 3 when the pawl holds a notch.
    box(ROOT + "/LauncherScaleBracket", (.02, .110, (DECK_TOP + .98) / 2), (.07, .012, .98 - DECK_TOP), "Dark", dead)
    box(ROOT + "/LauncherScale", (.02, .110, .985), (.075, .014, .010), "White")
    for k, d in enumerate((0.0,) + M.NOTCHES_M):
        x = PLUNGER_REST_X - d
        box(ROOT + f"/LauncherScaleTick{k}", (x, .1055, .9902), (.0012, .005, .0004), "Dark")
        _flat_text(stage, ROOT + f"/LauncherScaleLabel{k}", str(k), (x, .1135, .9902), .0055, mats["Dark"])

    # --- release lever (far side): HOLD / RELEASE detents along +Y -------------------------------------------------
    lx, ly0 = M.LEVER_DETENT0
    spacing = M.LEVER_SPACING_M
    lever = ROOT + "/ReleaseLever"
    UsdGeom.Xform.Define(stage, lever)
    ymid = ly0 + spacing / 2
    box(lever + "/Pedestal", (lx, ymid, (TABLE_TOP_Z + LEVER_TOP_Z) / 2), (.08, spacing + .07, LEVER_TOP_Z - TABLE_TOP_Z),
        "Dark", dead)
    box(lever + "/Slot", (lx, ymid, LEVER_TOP_Z + .0003), (.010, spacing + .030, .0006), "Rubber")
    for k, (text, y) in enumerate(zip(M.LEVER_LABELS, (ly0, ly0 + spacing))):
        box(lever + f"/Tick{k}", (lx - .016, y, LEVER_TOP_Z + .0004), (.012, .0015, .0006), "White")
        _flat_text(stage, lever + f"/Label{k}", text, (lx - .033 - .0035 * len(text), y, LEVER_TOP_Z + .0004), .008,
                   mats["White"])
    _text(stage, lever + "/Title", "RELEASE", (lx, ly0 + spacing + .0355, LEVER_TOP_Z - .022), .009, mats["White"])
    cz = LEVER_TOP_Z + .001 + .006
    body(lever + "/Carriage", (lx, ly0, cz), .10)
    box(lever + "/Carriage/Block", (0, 0, 0), (.034, .024, .012), "Steel", grip)
    cylinder(lever + "/Carriage/Post", (0, 0, .006 + .05), .012, .10, "Grip", material=grip)
    j = rail(lever + "/RailJoint", lever + "/Carriage", (lx, ly0, cz), "Y", (-.003, spacing + .003))
    drive(j, 50., 5., 50.)      # light detent (1 N mid-way), as on coupled_modes: a 3 N-per-finger grip carries the carriage

    # --- recorder panel (shared component), gate screens, display frame -----------------------------------------
    build_recorder_panel(stage, ROOT + "/Recorder", centre_xy=KEYPAD_CENTRE, top_z=KEYPAD_TOP_Z,
                         contact_material=grip, base_z=TABLE_TOP_Z,
                         display_centre=(RECORDER_DISPLAY_X, DISPLAY_Y, DISPLAY_Z), spec_text=RECORD_SPEC)
    for gate, x in SCREEN_X.items():
        build_screen(stage, ROOT + f"/Gate{gate}Screen", (x, DISPLAY_Y, DISPLAY_Z), f"GATE {gate}")
    box(ROOT + "/DisplayBoard", (.45, -.4175, DISPLAY_Z), (.92, .010, .24), "Dark")
    for side, x in (("Left", .01), ("Right", .89)):
        box(ROOT + "/DisplayLeg" + side, (x, -.41, (DECK_TOP + DISPLAY_Z - .12) / 2),
            (.025, .02, DISPLAY_Z - .12 - DECK_TOP), "Steel", dead)

    # --- air supply (schematic, non-interactive) -------------------------------------------------------------------
    import math
    box(ROOT + "/AirSupply", (.89, -.25, .930), (.17, .16, .10), "Dark", dead)
    cylinder(ROOT + "/AirSupplyGaugeBezel", (.89, -.25, .982), .030, .012, "Steel")
    cylinder(ROOT + "/AirSupplyGaugeFace", (.89, -.25, .989), .026, .002, "White")
    for i in range(9):
        angle = math.radians(-135 + i * 270 / 8)
        mark = box(ROOT + f"/AirSupplyDialTick{i}", (.89 + .021 * math.cos(angle), -.25 + .021 * math.sin(angle), .9905),
                   (.004, .0015, .0006), "Dark")
        mark.AddRotateZOp().Set(math.degrees(angle))
    needle = box(ROOT + "/AirSupplyDialNeedle", (.897, -.243, .991), (.020, .0015, .0008), "B")
    needle.AddRotateZOp().Set(45.)
    cylinder(ROOT + "/AirSupplyDialHub", (.89, -.25, .9915), .0025, .001, "Dark")
    cylinder(ROOT + "/SupplyHose", (.91, -.143, .888), .009, .17, "Dark", axis="Y")

    # 5 mm/s bounce threshold preserves low-speed restitution in the glider collision.
    scene = UsdPhysics.Scene.Define(stage, ROOT + "/Physics")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateBounceThresholdAttr(.005)
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT))
    stage.GetRootLayer().Save()
