# Air track: the mass of an unknown glider B

**Goal.** Determine the mass mass_B_kg (kg) of the red glider B.

**Apparatus.** A horizontal air track; its guide bearings exert no friction along the track. The blue glider A has a mass of 0.200 kg; the mass of the red glider B is unknown. Collisions between the gliders are not guaranteed to be perfectly elastic. Velocities are positive for motion away from the launcher and negative for motion towards it. At the start both gliders are held in their start catches, the release lever is at HOLD, the launcher is uncocked and the recorder is idle.
- Gliders. Each glider carries an opaque flag on top and a yellow vertical handle post. A glider can only be slid along the track; it cannot be lifted off it. On this bench the fingers have no contact hold: a handle is held by finger friction only (the fingers close with at most 3 N each), and `device_calibration` has no `gripper_mechanism`. On this bench the two fingers are not coupled: each has its own drive (at most 3 N) toward half the commanded width from the hand centre, so a finger that meets something stops there or pushes it along, while the other keeps closing toward its own target; the hand itself does not move. `get_robot_state` reports `gripper_holding`: true while both fingers have pressed on the same handle or lever post with more than 0.02 N each for at least 0.1 s; it does not say which.
- Start catches. Each glider has a spring catch (a brass tongue) at its start mark, labelled A or B on the rail. While the release lever is at HOLD a catch is down: a glider brought to rest with its centre on its mark is caught and held there; a glider dragged a few millimetres off the mark comes free again. Pulling the release lever to RELEASE lifts both catches.
- Launcher. A spring launcher (housing labelled LAUNCHER) whose plunger, when fired, strikes A standing in its catch. Pushing its yellow knob towards the launcher housing pushes the plunger back. A pawl holds the plunger in notch 1, 2 or 3 (the scale beside the knob path marks 0, 1, 2 and 3; the knob stands over 0 when uncocked), whichever is the deepest notch passed; the knob can always be pushed further back. The pawl only engages while the release lever is at HOLD. Each notch is a different compression of the same spring; the speed it gives A is not published. Moving the lever to RELEASE fires the launcher; afterwards it rests uncocked.
- Release lever. A yellow post on a carriage in a slot marked HOLD and RELEASE (title RELEASE), with two detents; it is moved by pushing the post along the slot. Passing the midpoint towards RELEASE fires the launcher, lifts both catches and starts a READY recorder; a detent spring then holds it at RELEASE until it is pushed back to HOLD, which lowers the catches again.
- End bumpers. The track's two end stops, the launcher's plunger face and a black rubber bumper, absorb a glider's impact without rebound.
- The round dial on the air supply box only illustrates the ideal air bearing: it is not a control and gives no reading.

**Instrument.** Two double-beam photogates, labelled 1 and 2. The two beams of a gate are 20 mm apart. A pass starts when a flag blocks one beam while the other is clear and is complete when the flag blocks the other beam; its time is the interval between the two blocking edges (0.01 ms resolution, edges interpolated between physics steps); its direction is + for a flag moving away from the launcher and − for one moving towards it. A flag that turns back before the second beam makes no pass; after a pass both beams must clear before the next one. The gates do not know which glider passed.
- Recorder. A keypad (keys 0–9, CLR and a red REC key, labelled RECORDER) and a display titled RECORDER (state IDLE, READY, REC, DONE or ERR; SET S, the keyed record length; T S, seconds recorded; NO, the run number). Type the record length in whole seconds and press REC: 5–60 s is accepted (anything else shows ERR). REC clears both gate screens and makes the recorder READY; the run starts when the release lever fires and records for the keyed length. READY cannot be cancelled, and a running record runs its full length; while READY or recording the keys do nothing.
- Gate screens. Each gate has its own screen, titled GATE 1 and GATE 2. Each lists, in order, every pass through its gate in the current or last run: the pass index, its direction and its time in ms; with more than six passes the screen shows the latest six.
- `export_recording(evidence_id)` returns a run as the timers recorded it: `evidence_id` (`<episode_id>:collision:<n>`), `run`, `status` (`recording` or `complete`), `length_s`, and `passes` (gate, index, direction, `t_first_beam_s` and `t_second_beam_s` from the start of the run, `dt_ms`). `evidence_id` may also be the run number or `"latest"`. It does not advance time. A pass that is missing only means that no flag completed it; it does not mean a velocity was zero.

**Acceptance conditions** (the evaluator checks these; how you measure and compute is up to you):
- At least one valid collision. A run may be cited in only one row (a row repeated identically counts once).
- B is at rest before every collision. While a collision's velocities are measured, nothing but the other glider exerts a force on either glider (no robot contact, launcher, bumper or catch).
- Every row of `measurements` cites the `evidence_id` of a run of this episode and gives the three signed velocities of that run's collision: A before, A after and B after; each must agree within 0.0005 m/s with the evaluator's own reading of that run.
- Your submitted values must be self-consistent with momentum conservation in the collisions you recorded: the evaluator refits your rows with that model, and your submitted mass must agree with the refit within 0.5%.
- Leave the bench restored before you submit: both gliders held in their start catches and at rest on their marks (centre within 1 mm of the mark, moving at most 5 mm/s), the launcher uncocked, the release lever at HOLD, the recorder neither READY nor recording, and the robot holding nothing.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top).
- Submit exactly once; the first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- A catch takes a glider whose centre is within 1 mm of its mark and which moves at 0.05 m/s or less, and holds it with a spring of at most 6 N; dragged more than 4 mm off the mark, the glider comes free. A glider moving faster rides over a lowered catch.
- The pawl drops into a notch once the plunger is 0.6 mm past it. A key acts when its cap is pressed 4 mm down (of 9 mm travel) and acts again only after it has risen back above 2 mm.
- A run counts as a valid collision only if B was at rest (at most 1 mm/s) from the release until the collision, the gliders touched only once until their velocities after the collision had been measured, and nothing but the other glider exerted a force on either glider while its velocities were being measured.
- A glider moved more than 0.2 mm off its rail ends the episode. At submission the gripper must not hold a glider handle or the release lever post.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.** Relative error e_rel = |m_B,submitted − m_B| / m_B. Success when all conditions hold and e_rel ≤ {threshold_pct}%. Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments: `mass_B_kg`, `unit` ("kg"), `measurements` (one row or more, each with `evidence_id`, `v_A_before_m_s`, `v_A_after_m_s`, `v_B_after_m_s`), `model` and `uncertainty_statement`; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values). Scoring is done offline.

Example (placeholder values only; they are not a consistent solution):
```json
{"mass_B_kg": "<mass of glider B in kg, a number>", "unit": "kg",
 "measurements": [
  {"evidence_id": "<episode_id>:collision:<n>", "v_A_before_m_s": "<velocity of A before the collision in m/s, a number>", "v_A_after_m_s": "<velocity of A after the collision in m/s, a number>", "v_B_after_m_s": "<velocity of B after the collision in m/s, a number>"}],
 "model": "<how the mass follows from your rows>", "uncertainty_statement": "<your uncertainty estimate and its basis>"}
```
