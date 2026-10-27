"""EmbodiedLab tool catalog: the bench tools of each experiment and the T1 perception and planning tools.

Bench tools are forwarded to the benchmark unchanged; the T1 tools run in the participant (participant.py).
bootstrap.exposed() decides which entries an environment shows and accepts.
"""

# Apparatus family of each task id.
FAMILIES = {
    'hooke_identification_v2': 'hooke', 'hooke_k': 'hooke', 'hooke_dynamic_k': 'hooke',
    'pendulum_gravity_v1': 'pendulum', 'pendulum_g': 'pendulum',
    'coupled_oscillator_modes_v1': 'coupled', 'coupled_modes': 'coupled', 'airtrack_collision': 'coupled',
    'torsion_inertia': 'torsion', 'torsion_centroid': 'torsion',
    'rc_capacitance': 'circuit', 'resistor_blackbox': 'circuit', 'circuit_blackbox3': 'circuit',
    'led_planck': 'circuit', 'battery_emf': 'circuit', 'rectifier_supply': 'circuit',
    'spectro_basic': 'chem', 'spectro_mixture': 'chem', 'dye_absorptivity': 'chem',
    'kinetics_crystal_violet': 'chem', 'titration_basic': 'chem', 'titration_weak_acid': 'chem',
    'lens_focal': 'chem', 'grating_wavelength': 'chem', 'slit_width': 'chem', 'plate_index': 'chem',
    'optical_relay': 'chem', 'calorimetry_specific_heat': 'chem', 'thermistor_beta': 'chem',
    'rod_expansion': 'chem', 'beam_compliance': 'chem',
    'projectile_landing': 'projectile',
    'incline_friction': 'friction',
}
LEGACY_OF = {'hooke': 'hooke_identification_v2', 'pendulum': 'pendulum_gravity_v1', 'coupled': 'coupled_oscillator_modes_v1'}


def family(task_id):
    return FAMILIES[task_id]


# Every experiment: read the task, observe, move the arm and the fingers, wait, submit.
BENCH = {
    'get_task_spec': ('()', 'Read the public experiment requirements and device calibration.'),
    'get_observation': ('()', 'Inspect the three calibrated RGB cameras (two fixed views and the wrist camera; calibration in device_calibration.cameras, no depth image). Images and rgb_ref tokens belong to the same physical state.'),
    'get_robot_state': ('()', 'Read joint positions, gripper width (the finger opening) and gripper_holding (whether the gripper holds a part).'),
    'execute_joint_trajectory': ('(expected_state_id: str, targets_rad: list[list[float]], velocities_rad_s: list[list[float]] | None = None)', 'Drive the arm along seven-joint position targets, one per physics tick (240 Hz), optionally with joint velocity feed-forward. Targets beyond the physical range of a joint are clamped to it. Nothing checks collisions. Every tick is charged to the budget. Omit velocities_rad_s when unused.'),
    'set_gripper': ('(width_m: float, expected_state_id: str)', 'Set the finger target width (0 to 0.08 m); advance simulation to actuate. Closing onto a part holds it once both fingers press on it and stop (a commanded width at least 2 mm narrower than the part); a commanded width more than 2 mm wider than the held part releases it (device_calibration.gripper_mechanism).'),
    'wait': ('(seconds: float)', 'Advance up to 30 simulated seconds, charged to the experiment budget.'),
}
SUBMIT = ('(**fields)', 'Submit once; arguments follow get_task_spec().submission_schema and the task card. The first submission closes the episode even if invalid; it is scored offline.')
SOLVE_IK = ('(target_pose: list[float], frame: str = "base", seed_joints: list[float] | None = None)', 'Kinematics only: seven joint angles within the joint limits that put the hand (ee_link panda_hand) at target_pose [qw,qx,qy,qz,x,y,z] in the robot base frame (frame="base") or the world frame (frame="world"), searched from seed_joints (default: the current joints). Returns converged, joint_positions_rad (only when converged; no approximate solution) and the position and orientation residuals. Does not move the robot, check collisions or plan a path.')

# Exposure level T1 adds these perception and planning tools (bootstrap.PRIMITIVES), in this order.
PRIMITIVES = {
    'plan_return_to_start': ('(execute: bool = False)', "Collision-aware transit (cuRobo) back to the initial public hand pose, planned against the bench's own collision model: every part as built, at its current pose, and the held object. execute=False only previews; execute=True replans from the current state and moves. A blocked destination is reported without a fallback."),
    'get_robot_pose': ('()', 'EmbodiedLab FK: current hand pose in world/base coordinates and calibrated tool points, computed from public joints and robot CAD.'),
    'mask_geometry': ('(mask_id: str)', 'EmbodiedLab vision: visible point-cloud bounds and median for a selected current SAM mask. These are estimates of visible surfaces, not hidden simulator geometry.'),
    'translate_contact': ('(delta_base_m: list[float])', 'Bounded contact servo: move the hand by delta_base_m (robot base frame, at most 5 cm) keeping its orientation, slowly enough to touch or withdraw from an object. Not collision-aware. Returns action_completed, tracking_error_m and the resulting robot state; a large tracking error means the hand was stopped by contact or a joint limit. Re-observe afterwards.'),
    'localize_rgbd_point': ('(rgb_ref: str, point_xy: list[float])', 'EmbodiedLab vision: backproject an Agent-selected pixel with same-frame calibrated depth. Inspect RGB first.'),
    'segment_sam3_point_prompt': ('(rgb_ref: str, point_xy: list[float])', 'SAM3 point2mask on the exact original RGB named by rgb_ref and the main Agent-selected [x,y] pixel. Returns scored mask candidates and visual overlays; choose a mask explicitly.'),
    'inspect_mask_image': ('(mask_id: str, padding_px: int = 32, scale: int = 4)', 'EmbodiedLab visual inspection: magnify an explicitly selected SAM candidate with surrounding context. Returns unmarked RGB and mask-overlay close-ups, original-pixel crop bounds and coordinate mapping. Use to inspect small or ambiguous candidates before selecting a contact pose. This does not identify the object, choose another mask, move the camera or robot, or add image detail. All SAM and motion inputs still use original RGB pixels.'),
    'move_hand': ('(target_pose: list[float])', "Collision-aware transit (cuRobo): plan and execute a path to the hand pose [qw,qx,qy,qz,x,y,z] in ROBOT BASE coordinates; convert world points with device_calibration.world_from_robot_base. It plans against the bench's own collision model: every part as built, at its current pose, and the held object. IK_FAIL means the pose is unreachable (or in the wrong frame); INVALID_START_STATE_WORLD_COLLISION means the hand is currently in contact, back out with translate_contact first."),
}

# Instrument records an experiment declares in its task tools.
INSTRUMENTS = {
    'export_encoder': ('(evidence_id: str | int)', 'Export a finished, immutable tray-encoder record of this episode (records are started with REC on the recorder panel): rows and CSV (sample_id, t_s, z_mm; z_mm is empty while the tray is outside the encoder range), t_s in simulated seconds from REC. evidence_id may also be the record number shown on the display (NO) or "latest". Does not advance time.'),
    'export_photogate': ('(evidence_id: str | int)', 'Export a finished, immutable photogate record of this episode (records are started with REC on the recorder panel): transit rows and CSV (transit_id, t_block_s, t_clear_s), times in simulated seconds from REC. evidence_id may also be the record number shown on the display (NO) or "latest". Does not advance time.'),
    'export_logger': ('(evidence_id: str)', 'A finished logger record of this episode (evidence_id <episode_id>:logger:<n>, n as NO on the logger display): CSV (sample_id, t_s, status, ch1_V, ch2_V) with its SHA-256, rate, length, start and end time; t_s is simulation time since the episode start. Does not advance time.'),
    'export_spectro_scans': ('()', 'Every SCAN of this episode so far: scan_id, t_s, wavelengths (nm) and absorbances against the factory reference, with CSV and SHA-256. Scans are survey spectra; they cannot be cited as readings. Does not advance time.'),
    'export_spectro_kinetics': ('()', 'Every timed KIN run of this episode so far: run, wavelength, blank wavelength and the readings (point, t_s, A_display), with CSV and SHA-256. Cite a run as <episode_id>:kinetics:<run>. Does not advance time.'),
    'export_temperature_log': ('()', "The recorder's records of this episode so far (a record runs from one press of the REC cap to the next; nothing is recorded otherwise): each with its record_id <episode_id>:thermo_rec:<n> (n as NO on the thermometer display), started_s, ended_s (null while it runs) and its rows (sample_id, t_s, T_C the water thermometer, heater_C the heater block thermometer) with CSV and SHA-256. Does not advance time."),
    'export_titration_log': ('()', 'Every recorder row of this episode so far: rows (sample_id, t_s, pH_display, stable), CSV with SHA-256, and the records (evidence_id <episode_id>:titration:<n>, first and last sample_id). Does not advance time.'),
    'export_flight': ('(evidence_id: str | None = None)', 'The flight timer memory. Without arguments: every completed shot record of this episode, oldest first ({records: [...]}). With evidence_id: that immutable record. A record holds record_number, evidence_id, the raw photogate edges (gate_events, gate_separation_m) and the light-curtain crossing (landing, distance_origin, receiver_center_z_m). Does not advance time.'),
}
EXPORT_RECORDING = {
    'coupled_modes': 'Export a finished, immutable two-channel record of this episode (records are started by the release post after REC): rows and CSV (t_s, x1_mm, x2_mm, status), t_s in simulated seconds from the release, status OK or OVER_RANGE, plus metadata and SHA. evidence_id may also be the record number shown on the display (NO) or "latest"; submit_result needs the full evidence id string. Does not advance time.',
    'airtrack_collision': 'A run of the two photogate timers (its evidence_id, the run number shown on the recorder display, or "latest"): run, status (recording or complete), length_s and every pass (gate, index, direction, beam times from the start of the run, dt_ms at 0.01 ms). The record does not say which glider passed. Does not advance time.',
    'incline_friction': 'A finished keypad record (evidence_id, record_number, length_s) with its samples: physics_tick, time_s, angle_deg, position_m (the position of the slider on the sensor track that is nearest the lower end of the ramp; null while no slider lies flat on the track). evidence_id may also be the record number shown on the display or "latest". Does not advance time.',
}
WRITES = {'execute_joint_trajectory', 'set_gripper', 'wait', 'submit_result', 'plan_return_to_start',
          'translate_contact', 'move_hand', 'export_flight'}

# Bench-specific wording of the observation, robot state and gripper tools.
OBSERVATION_FIXED = 'Inspect the three calibrated RGB cameras (calibration in device_calibration.cameras, no depth image). Images and rgb_ref belong to the same physical state.'
OBSERVATION_HOOKE = 'Inspect the three calibrated RGB cameras (calibration in device_calibration.cameras, no depth image), with same-frame rgb_ref tokens and an evidence_id bound to this physical state (its images are these three views).'
OBSERVATION_EVIDENCE = {
    'pendulum_g': ' Returns evidence_id (<episode_id>:obs:<n>): cite it for the timer readings this observation shows.',
    'hooke_dynamic_k': ' In this experiment the measurement evidence is the tray-encoder record (export_encoder).',
    'beam_compliance': ' Returns evidence_id (<episode_id>:obs:<n>): cite it for the indicator reading this observation shows.',
    **dict.fromkeys(('lens_focal', 'grating_wavelength', 'plate_index', 'slit_width', 'optical_relay'),
                    ' Returns evidence_id (<episode_id>:obs:<n>): cite it for the rail and screen readings this observation shows.'),
    **dict.fromkeys(('spectro_basic', 'spectro_mixture', 'dye_absorptivity', 'kinetics_crystal_violet',
                     'thermistor_beta', 'rod_expansion', 'resistor_blackbox', 'battery_emf', 'rc_capacitance',
                     'circuit_blackbox3', 'led_planck', 'rectifier_supply'),
                    ' Returns evidence_id (<episode_id>:obs:<n>): cite it for the display readings this observation shows.'),
}
ROBOT_STATE = {
    'pendulum': 'Read joint positions and gripper width.',
    'coupled': 'Read joint positions, gripper width and gripper_holding (both fingers have pressed on the same part for at least 0.1 s; it does not say which).',
    'torsion': 'Read joint positions, gripper width (the finger opening) and gripper_holding (whether the gripper holds an object).',
}
ROBOT_STATE_POSE = ' Also the hand pose by forward kinematics of the public robot calibration: hand_pose_base_wxyz_xyz and hand_pose_world_wxyz_xyz ([qw,qx,qy,qz,x,y,z] of the ee_link panda_hand), jaw_center_world_m and fingertip_center_world_m.'
# The pendulum and rail benches have no contact hold.
SET_GRIPPER_FRICTION = 'Set the finger target width (0 to 0.08 m); advance simulation to actuate. On this bench the fingers have no contact hold: a closed grip holds by finger friction only.'


def _entry(name, signature, description, effect=None):
    return {'name': name, 'signature': signature, 'description': description,
            'effect': effect or ('write' if name in WRITES else 'read'), 'approval_policy': 'allow'}


def catalog(task_id, task=None):
    """Every tool of this experiment at exposure level T1, in display order; bootstrap filters it for T0."""
    task = task or {}
    kind = family(task_id)
    allowed = set(task.get('tools') or ())
    text = {name: description for name, (_, description) in BENCH.items()}
    if kind in ('pendulum', 'coupled'):
        text['get_observation'] = OBSERVATION_FIXED
        text['set_gripper'] = SET_GRIPPER_FRICTION
    elif kind == 'hooke':
        text['get_observation'] = OBSERVATION_HOOKE
    text['get_observation'] += OBSERVATION_EVIDENCE.get(task_id, '')
    text['get_robot_state'] = ROBOT_STATE.get(kind, text['get_robot_state']) + ROBOT_STATE_POSE
    entries = [_entry(name, signature, text[name]) for name, (signature, _) in BENCH.items()]
    submit = _entry('submit_result', submit_signature(task), SUBMIT[1])
    if kind == 'coupled':
        entries += [_entry(name, *value) for name, value in PRIMITIVES.items()]
        if 'export_recording' in allowed:
            entries.append(_entry('export_recording', '(evidence_id: str | int)', EXPORT_RECORDING[task_id], 'read'))
        entries.append(submit)
    else:
        entries.append(submit)
        entries += [_entry(name, *value) for name, value in PRIMITIVES.items()]
        if 'export_recording' in allowed:
            entries.append(_entry('export_recording', '(evidence_id: str)', EXPORT_RECORDING[task_id], 'write'))
    entries += [_entry(name, *value) for name, value in INSTRUMENTS.items() if name in allowed]
    entries.append(_entry('solve_ik', *SOLVE_IK))
    return entries


def submit_signature(task):
    schema = task.get('submission_schema')
    if not schema:
        return SUBMIT[0]
    fields = ', '.join(f"{k}: {schema_type(v, schema)}" for k, v in schema['properties'].items()
                       if k in schema.get('required', []))
    optional = ', '.join(f"{k}: {schema_type(v, schema)} | None = None" for k, v in schema['properties'].items()
                         if k not in schema.get('required', []))
    return '(' + fields + (', ' + optional if optional else '') + ')'


KINDS={'number':'float','integer':'int','string':'str','array':'list','object':'dict','boolean':'bool','null':'None'}


def schema_type(prop,schema,depth=0):
 """Python-style type of a JSON-schema property for the submit_result signature ($ref, enum, const and unions resolved)."""
 if not isinstance(prop,dict) or depth>8:return 'Any'
 ref=prop.get('$ref')
 if isinstance(ref,str) and ref.startswith('#/'):
  target=schema
  for part in ref[2:].split('/'):target=target.get(part,{}) if isinstance(target,dict) else {}
  return schema_type(target,schema,depth+1)
 if 'const' in prop:return f'Literal[{prop["const"]!r}]'
 if isinstance(prop.get('enum'),list):return 'Literal['+', '.join(repr(v) for v in prop['enum'])+']'
 for key in ('anyOf','oneOf'):
  if isinstance(prop.get(key),list):
   kinds=[]
   for option in prop[key]:
    kind=schema_type(option,schema,depth+1)
    if kind not in kinds:kinds.append(kind)
   return ' | '.join(kinds)
 kind=prop.get('type')
 if isinstance(kind,list):return ' | '.join(KINDS.get(k,'Any') for k in kind)
 if kind=='array' and isinstance(prop.get('items'),dict):return f"list[{schema_type(prop['items'],schema,depth+1)}]"
 if kind is None and isinstance(prop.get('properties'),dict):return 'dict'
 return KINDS.get(kind,'Any')
