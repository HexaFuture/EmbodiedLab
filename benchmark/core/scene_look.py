"""Laboratory look (visual only): room light, ceiling panels, a window sun, textured wall, floor and bench top, and
the dressing of the bench wall (wainscot or tiles, socket trunking, notice board, safety signs, a reagent shelf).

Only render-side prims change here: light intensities and new lights, surfaces added to the existing visual
materials (their bindings, the physics materials and every collider stay as authored), and new visual-only prims
with no collision API, all on the wall beyond the arm's reach. Cameras untouched.

Per instance: the bench top (4 kinds) and the wall scheme (3 kinds) rotate with the instance number, so any 12
consecutive instances of an experiment's split show all 12 combinations once and neighbouring instances differ in
both; the sun direction and strength and the shelf contents follow the instance's appearance seed. The server calls
configure() with the instance before the scene is built, and every scene backend calls apply() once on its stage, so
the look holds for the whole trial and the same instance always looks the same. Colours the task cards name,
displays and chemical colours are never varied.
"""
import hashlib
import random

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

LOOK_DIR = "/isaac-sim/documents/look"
F, B, F2, C3, A, T = (Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Bool, Sdf.ValueTypeNames.Float2,
                      Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.Token)
NEUTRAL = Gf.Vec3f(1.0, 0.97, 0.92)       # about 4500 K
WALL_Y = -0.755                           # the bench wall: front face of the backdrop (read at run time)
BENCH_Z = 0.80

BENCHES = {   # bench top: texture, tint, roughness
    "black_epoxy": dict(tex="bench_top.png", color=(0.15, 0.15, 0.16), rough=0.28),
    "green_phenolic": dict(tex="bench_top.png", color=(0.10, 0.24, 0.18), rough=0.32),
    "beech": dict(tex="bench_beech.png", color=(0.86, 0.74, 0.60), rough=0.45),
    "walnut": dict(tex="bench_beech.png", color=(0.52, 0.36, 0.24), rough=0.4),
}
WALLS = {     # wall paint tint, and the band above the bench: painted wainscot or white tiles
    "cream_green": dict(wall=(0.64, 0.62, 0.55), band=(0.34, 0.52, 0.40)),
    "white_blue": dict(wall=(0.63, 0.63, 0.62), band=(0.32, 0.45, 0.62)),
    "warm_tile": dict(wall=(0.64, 0.61, 0.56), band=None),
}


BENCH_ORDER = ("black_epoxy", "beech", "green_phenolic", "walnut")     # dark and light alternate
WALL_ORDER = ("cream_green", "warm_tile", "white_blue")
N_LOOKS = len(BENCH_ORDER) * len(WALL_ORDER)                           # 12; 4 and 3 coprime


def combo(k):
    """Look number k (0..11) -> (bench, wall); k mod 4 and k mod 3 together pick each pair exactly once."""
    return BENCH_ORDER[k % len(BENCH_ORDER)], WALL_ORDER[k % len(WALL_ORDER)]


def look_number(exp_id, split, index):
    """The instance's look number: its index in the split plus a fixed per-experiment offset, modulo 12."""
    offset = int(hashlib.sha256(f"look:{exp_id}:{split}".encode()).hexdigest()[:8], 16) % N_LOOKS
    return (int(index) + offset) % N_LOOKS


def choose(exp_id="", split="", index=0, seed=0, k=None):
    k = look_number(exp_id, split, index) if k is None else k
    bench, wall = combo(k)
    rng = random.Random(seed ^ 0x100C)
    return {"k": k, "bench": bench, "wall": wall,
            "sun_yaw": 25.0 + rng.uniform(-15.0, 15.0), "sun_gain": rng.uniform(0.85, 1.15),
            "shelf": rng.randrange(1 << 30)}


_LOOK = choose()


def configure(exp_id, split, index, seed):
    """Set the look of the instance this process runs; apply() uses it."""
    global _LOOK
    _LOOK = choose(exp_id, split or "", int(index or 0), int(seed or 0))


def _omnipbr(stage, mat_path, tex=None, color=None, rough=0.6, metal=0.0, scale=(1.0, 1.0)):
    prim = stage.GetPrimAtPath(mat_path)
    if not prim.IsValid():
        return
    mat = UsdShade.Material(prim)
    sh = UsdShade.Shader.Define(stage, mat_path + "/LookPBR")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if tex:
        sh.CreateInput("diffuse_texture", A).Set(Sdf.AssetPath(f"{LOOK_DIR}/{tex}"))
    if color:   # with a texture the colour tints it (OmniPBR multiplies the texture by diffuse_tint)
        sh.CreateInput("diffuse_tint" if tex else "diffuse_color_constant", C3).Set(Gf.Vec3f(*color))
    sh.CreateInput("reflection_roughness_constant", F).Set(float(rough))
    sh.CreateInput("metallic_constant", F).Set(float(metal))
    sh.CreateInput("project_uvw", B).Set(True)
    sh.CreateInput("world_or_object", B).Set(True)
    sh.CreateInput("texture_scale", F2).Set(Gf.Vec2f(*scale))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateVolumeOutput("mdl"), mat.CreateDisplacementOutput("mdl")):
        o.ConnectToSource(out)


def _surface(stage, path, tex=None, color=(0.8, 0.8, 0.8), rough=0.6, metal=0.0):
    """A UsdPreviewSurface material; a texture is read through the mesh's own st coordinates."""
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():          # already defined: only a constant colour can change
        if not tex:
            UsdShade.Shader.Define(stage, path + "/s").CreateInput("diffuseColor", C3).Set(Gf.Vec3f(*color))
        return UsdShade.Material(prim)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", F).Set(float(rough))
    sh.CreateInput("metallic", F).Set(float(metal))
    if tex:
        rd = UsdShade.Shader.Define(stage, path + "/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", T).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", A).Set(Sdf.AssetPath(f"{LOOK_DIR}/{tex}"))
        tx.CreateInput("st", F2).ConnectToSource(rd.ConnectableAPI(), "result")
        tx.CreateInput("wrapS", T).Set("repeat")
        tx.CreateInput("wrapT", T).Set("repeat")
        tx.CreateInput("sourceColorSpace", T).Set("sRGB")
        sh.CreateInput("diffuseColor", C3).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", C3).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _dim(stage, path, intensity=0.0):
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        UsdLux.LightAPI(p).CreateIntensityAttr(float(intensity))


def _wall_front(stage):
    """World y of the front face of the bench wall as the scene actually built it: the most forward of the large
    thin panels standing on that wall (the backdrop's name and thickness differ between scene files)."""
    from pxr import Usd
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    best = None
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not p.IsA(UsdGeom.Gprim) or path.startswith(("/World/LabDressing", "/World/Franka", "/World/Lights")):
            continue
        if p.IsA(UsdGeom.Cube):
            # A cube renders from its size, but a flattened scene file can carry a stale extent; bound it from size.
            h = 0.5 * float(UsdGeom.Cube(p).GetSizeAttr().Get() or 2.0)
            box = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-h, -h, -h), Gf.Vec3d(h, h, h)),
                            UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            r = box.ComputeAlignedRange()
        else:
            r = bc.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        mn, mx = r.GetMin(), r.GetMax()
        if mx[0] - mn[0] > 1.0 and mx[2] - mn[2] > 1.0 and mx[1] - mn[1] < 0.3 and -0.9 < mx[1] < -0.6:
            best = mx[1] if best is None else max(best, mx[1])
    return WALL_Y if best is None else float(best)


def apply(stage, look=None):
    global WALL_Y
    WALL_Y = _wall_front(stage)
    look = look or _LOOK
    wall = WALLS[look["wall"]]
    # Surfaces: grey vinyl floor, painted wall, the instance's bench top, dark steel legs.
    _omnipbr(stage, "/World/Looks/visual_material", tex="floor_vinyl.png", rough=0.55, scale=(2.0, 2.0))
    _omnipbr(stage, "/World/Looks/visual_material_1", **BENCHES[look["bench"]])
    for i in (2, 3, 4, 5):
        _omnipbr(stage, f"/World/Looks/visual_material_{i}", color=(0.06, 0.06, 0.065), rough=0.4, metal=0.7)
    _omnipbr(stage, "/World/Looks/visual_material_6", tex="wall_paint.png", color=wall["wall"], rough=0.85, scale=(2.0, 2.0))
    # Lights: the old soft rig off; the room (HDRI) as a dim fill; two ceiling LED panels; a window sun.
    for path in ("/World/Lights/Key", "/World/Lights/BlueFill", "/World/Lights/WarmRim",
                 "/World/Lights/NeutralTopKey", "/World/HookeApparatus/env_light", "/World/Lights/PendulumFill",
                 "/World/Lights/PendulumKey"):
        _dim(stage, path, 0.0)
    _dim(stage, "/World/Lights/NeutralFrontFill", 150.0)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/NeutralDome")
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(f"{LOOK_DIR}/vintage_measuring_lab_2k.hdr"))
    dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(170.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 1.0))
    for k, (x, y) in enumerate(((0.45, -0.05), (1.35, -0.05))):
        panel = UsdLux.RectLight.Define(stage, f"/World/Lights/CeilingPanel{k}")
        panel.CreateWidthAttr(1.2)
        panel.CreateHeightAttr(0.3)
        panel.CreateIntensityAttr(13000.0)
        panel.CreateColorAttr(NEUTRAL)
        UsdGeom.XformCommonAPI(panel).SetTranslate(Gf.Vec3d(x, y, 2.75))
    dress_room(stage, wall)
    dress_bench_wall(stage, wall, look["shelf"])
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/WindowSun")
    sun.CreateIntensityAttr(2200.0 * look["sun_gain"])
    sun.CreateAngleAttr(0.8)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-35.0, look["sun_yaw"], 0.0))
    print(f"[scene_look] wall_y={WALL_Y:.3f} {look}", flush=True)


FLOOR_Z = -0.05
SHELF_Z = (0.15, 0.60, 1.05, 1.50)          # the wire rack's shelf tops
BOTTLES = ("duran_gl45_1000ml_amber.usda", "duran_gl45_500ml_clear.usda", "winchester_2500ml_amber.usda")


def _panel(stage, name, center, size, mat):
    """A visual-only box (no collider)."""
    c = UsdGeom.Cube.Define(stage, f"/World/LabDressing/{name}")
    c.CreateSizeAttr(1.0)
    x = UsdGeom.XformCommonAPI(c)
    x.SetTranslate(Gf.Vec3d(*center))
    x.SetScale(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def _prop(stage, name, usd, xyz, yaw=0.0):
    fresh = not stage.GetPrimAtPath(f"/World/LabDressing/{name}").IsValid()
    p = UsdGeom.Xform.Define(stage, f"/World/LabDressing/{name}")
    if fresh:
        p.GetPrim().GetReferences().AddReference(f"{LOOK_DIR}/props/{usd}")
    x = UsdGeom.XformCommonAPI(p)
    x.SetTranslate(Gf.Vec3d(*xyz))
    x.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw)))


def _wall_quad(stage, name, x0, x1, z0, z1, mat, off=0.004, rep=(1.0, 1.0)):
    """A visual-only rectangle on the bench wall facing the cameras (+y), with its own texture coordinates."""
    y = WALL_Y + off
    m = UsdGeom.Mesh.Define(stage, f"/World/LabDressing/{name}")
    m.CreatePointsAttr([Gf.Vec3f(x1, y, z0), Gf.Vec3f(x0, y, z0), Gf.Vec3f(x0, y, z1), Gf.Vec3f(x1, y, z1)])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateNormalsAttr([Gf.Vec3f(0, 1, 0)] * 4)
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    su, sv = rep
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(su, 0), Gf.Vec2f(su, sv), Gf.Vec2f(0, sv)])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def dress_room(stage, wall):
    """Room walls around the bench wall, and lab furniture beyond both table ends (ChemLab-Assets, CC BY-NC 4.0)."""
    path = "/World/Looks/look_wall"
    UsdShade.Material.Define(stage, path)
    _omnipbr(stage, path, tex="wall_paint.png", color=wall["wall"], rough=0.85, scale=(2.0, 2.0))
    mat = UsdShade.Material(stage.GetPrimAtPath(path))
    _panel(stage, "BackWall", (0.5, WALL_Y - 0.011, 1.575), (10.0, 0.02, 3.25), mat)   # flush with the backdrop
    _panel(stage, "SideWallPlusX", (4.4, 1.6, 1.575), (0.02, 4.8, 3.25), mat)
    _panel(stage, "SideWallMinusX", (-3.4, 1.6, 1.575), (0.02, 4.8, 3.25), mat)
    _prop(stage, "SafetyCabinet", "asecos_q90_195_120.usda", (3.55, -0.46, FLOOR_Z), 180.0)
    _prop(stage, "GasCylinder", "gas_cylinder_50l_200bar.usda", (2.65, -0.62, FLOOR_Z), 180.0)
    rx, ry = -2.3, -0.55
    _prop(stage, "WireRack", "metro_super_erecta_1848_4shelf.usda", (rx, ry, FLOOR_Z), 180.0)
    k = 0
    for level, z in enumerate(SHELF_Z[1:], 1):
        for i, dx in enumerate((-0.42, -0.24, -0.06, 0.14, 0.34)):
            if (level + i) % 3 == 0:
                continue
            k += 1
            _prop(stage, f"Bottle{k}", BOTTLES[(level * 2 + i) % 3], (rx + dx, ry + 0.02 * (i % 2), FLOOR_Z + z), 0.0)


def dress_bench_wall(stage, wall, shelf_seed):
    """What a lab bench wall carries, all beyond the arm's reach: a wainscot or tiled band, a socket trunking,
    a notice board (blank sheets: no text, nothing about any task), safety signs, a reagent shelf with a light."""
    x0, x1 = -1.9, 2.3
    if wall["band"] is None:
        band = _surface(stage, "/World/Looks/look_tile", tex="backsplash_tile.png", rough=0.2)
        _wall_quad(stage, "Band", x0, x1, BENCH_Z, 1.22, band, rep=((x1 - x0) / 0.6, 0.42 / 0.6))
    else:
        band = _surface(stage, "/World/Looks/look_band", color=wall["band"], rough=0.55)
        _wall_quad(stage, "Band", x0, x1, BENCH_Z, 1.22, band)
    trim = _surface(stage, "/World/Looks/look_trim", color=(0.72, 0.72, 0.70), rough=0.4, metal=0.3)
    _panel(stage, "BandTrim", ((x0 + x1) / 2, WALL_Y + 0.006, 1.225), (x1 - x0, 0.012, 0.014), trim)
    # Socket trunking with outlets.
    plastic = _surface(stage, "/World/Looks/look_trunking", color=(0.86, 0.86, 0.84), rough=0.45)
    _panel(stage, "Trunking", ((x0 + x1) / 2, WALL_Y + 0.018, 1.06), (x1 - x0, 0.036, 0.085), plastic)
    plate = _surface(stage, "/World/Looks/look_socket", tex="socket_plate.png", rough=0.35)
    for k, sx in enumerate((-1.45, -0.85, -0.25, 0.35, 0.95, 1.55, 2.1)):
        _wall_quad(stage, f"Socket{k}", sx - 0.036, sx + 0.036, 1.024, 1.096, plate, off=0.0375)
    # Notice board and safety signs (pictograms only).
    _wall_quad(stage, "NoticeBoard", 1.55, 2.15, 1.40, 1.80, _surface(stage, "/World/Looks/look_notice",
               tex="notice_board.png", rough=0.8), off=0.012)
    for name, tex, sx, sz in (("SignWarning", "sign_warning.png", 1.10, 1.55), ("SignGoggles", "sign_goggles.png", 0.0, 1.55),
                              ("SignEyewash", "sign_eyewash.png", -0.85, 1.50)):
        _wall_quad(stage, name, sx - 0.11, sx + 0.11, sz - 0.11, sz + 0.11,
                   _surface(stage, f"/World/Looks/look_{name}", tex=tex, rough=0.3), off=0.006)
    # Reagent shelf: white melamine board, steel brackets, an LED strip under it lighting the wall.
    sx0, sx1, sz, depth = -1.75, -0.35, 1.72, 0.24
    board = _surface(stage, "/World/Looks/look_shelf", color=(0.84, 0.84, 0.82), rough=0.5)
    _panel(stage, "WallShelf", ((sx0 + sx1) / 2, WALL_Y + depth / 2, sz - 0.0125), (sx1 - sx0, depth, 0.025), board)
    steel = _surface(stage, "/World/Looks/look_steel", color=(0.62, 0.63, 0.65), rough=0.35, metal=0.9)
    for k, bx in enumerate((sx0 + 0.1, sx1 - 0.1)):
        _panel(stage, f"ShelfBracket{k}", (bx, WALL_Y + 0.09, sz - 0.1), (0.015, 0.18, 0.17), steel)
    strip = UsdLux.RectLight.Define(stage, "/World/Lights/ShelfStrip")
    strip.CreateWidthAttr(sx1 - sx0 - 0.1)
    strip.CreateHeightAttr(0.02)
    strip.CreateIntensityAttr(6000.0)
    strip.CreateColorAttr(NEUTRAL)
    UsdGeom.XformCommonAPI(strip).SetTranslate(Gf.Vec3d((sx0 + sx1) / 2, WALL_Y + 0.05, sz - 0.03))
    rng = random.Random(shelf_seed)
    x, k = sx0 + 0.08, 0
    while x < sx1 - 0.08:
        k += 1
        usd = rng.choice(BOTTLES)
        _prop(stage, f"ShelfBottle{k}", usd, (x, WALL_Y + 0.11 + rng.uniform(-0.02, 0.02), sz), rng.uniform(0, 360))
        x += (0.16 if usd.startswith("winchester") else 0.12) + (0.18 if rng.random() < 0.2 else 0.0)
