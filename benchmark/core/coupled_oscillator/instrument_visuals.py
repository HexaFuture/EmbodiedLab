"""Encoder display geometry; no dynamics or robot policy."""
def build_visuals(stage,root):
    from pendulum_asset import _box,_text,_digits,_set_digits,_material,_FONT
    _FONT.update({"H":(17,17,17,31,17,17,17),"Z":(31,1,2,4,8,16,31)})
    from pxr import UsdGeom,Gf
    white=_material(stage,root+'/Materials/RulerWhite',(.95,.95,.89))
    black=_material(stage,root+'/Materials/RulerInk',(.006,.012,.02))
    colors=[_material(stage,root+'/Materials/Channel1',(.08,.72,1.),emission=.25),
            _material(stage,root+'/Materials/Channel2',(1.,.35,.06),emission=.25)]
    displays=[]
    for i,center in enumerate((.43,.97)):
        # Front faces point +Y, matching the two existing fixed cameras.
        ruler=root+f'/Ruler{i+1}'
        _box(stage,ruler+'/Board',(center,.292,1.015),(.228,.012,.085),white)
        for mm in range(-100,101,5):
            height=.020 if mm%20==0 else (.014 if mm%10==0 else .008)
            _box(stage,ruler+f'/Tick{mm+100}',(center+mm/1000,.299,1.047-height/2),(.0008,.001,height),black)
            if mm%20==0:
                _text(stage,ruler+f'/Number{mm+100}',str(mm),(center+mm/1000,.301,1.012),.009,black)
        _text(stage,ruler+'/Unit',f'CH{i+1} MM  ZERO=0',(center,.301,.984),.006,black)
        # Larger, non-colliding labels on both sides beside the grip post.
        for face,y in (('Front',-.251),('Back',-.129)):
            _text(stage,root+f'/Slider{i+1}/GripLabel{face}',f'CH{i+1}',
                  (0,y,.19),.025,colors[i])
        pointer=root+f'/Slider{i+1}/Pointer'
        _box(stage,pointer+'/Arm',(0,.22,.09),(.006,.18,.008),colors[i])
        mesh=UsdGeom.Mesh.Define(stage,pointer+'/Tip')
        mesh.CreatePointsAttr([(-.008,.309,.085),(.008,.309,.085),(0,.309,.058)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2]);mesh.CreateDoubleSidedAttr(True)
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(colors[i])
        _box(stage,root+f'/Slider{i+1}/Fiducial',(0,0,.087),(.075,.075,.003),colors[i])
        _box(stage,root+f'/Slider{i+1}/FiducialCenter',(0,0,.090),(.014,.014,.003),black)
        panel=root+f'/EncoderDisplay{i+1}'
        _box(stage,panel+'/Housing',(center,-.37,1.30),(.265,.025,.155),black)
        _text(stage,panel+'/Title',f'CH{i+1} POSITION MM',(center,-.355,1.36),.011,white)
        digits=_digits(stage,panel+'/Digits',(center,-.354,1.312),5,.041,colors[i],separator_after=4,separator_gap=.008)
        # Five cells: sign, hundreds, tens, units, tenths.
        spacing=.041*.68
        _box(stage,panel+'/Decimal',(center-1.5*spacing,-.353,1.292),(.004,.001,.004),colors[i])
        _text(stage,panel+'/Spec','120 HZ  0.1 MM',(center,-.354,1.252),.010,white)
        _set_digits(stage,digits,' 0000')
        displays.append(digits)
    return displays

# Keep segment geometry fixed. LED colors, rather than visibility or scale,
# carry the state so repeated on/off transitions do not leave stale geometry.
_DISPLAY_SHADER_CACHE = {}
_DISPLAY_LED_STATE = {}

def update_visuals(stage,displays,row):
    from pendulum_asset import _SEGMENTS,_material
    from pxr import UsdGeom,Gf,UsdShade,Sdf
    if row is None:return
    for channel,(path,key) in enumerate(zip(displays,('x1_mm','x2_mm'))):
        v=row[key]
        text='-----' if v is None else ('-' if v<0 else ' ')+f'{round(abs(v)*10):04d}'
        color=(.08,.72,1.) if channel==0 else (1.,.35,.06)
        for index,char in enumerate(text):
            for segment in 'abcdefg':
                prim_path=f'{path}/D{index}_{segment}'
                cache_key=(str(stage.GetRootLayer().identifier),prim_path)
                if cache_key not in _DISPLAY_SHADER_CACHE:
                    prim=stage.GetPrimAtPath(prim_path)
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Set('inherited')
                    material=_material(stage,prim_path+'/LEDMaterial',(.001,.002,.003))
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                    shader=UsdShade.Shader(stage.GetPrimAtPath(prim_path+'/LEDMaterial/Shader'))
                    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
                    _DISPLAY_SHADER_CACHE[cache_key]=shader
                shader=_DISPLAY_SHADER_CACHE[cache_key]
                enabled=segment in _SEGMENTS[char]
                if _DISPLAY_LED_STATE.get(cache_key) == enabled: continue
                _DISPLAY_LED_STATE[cache_key]=enabled
                shader.GetInput('diffuseColor').Set(Gf.Vec3f(*(color if enabled else (.001,.002,.003))))
                shader.GetInput('emissiveColor').Set(Gf.Vec3f(*(tuple(.25*v for v in color) if enabled else (0,0,0))))
