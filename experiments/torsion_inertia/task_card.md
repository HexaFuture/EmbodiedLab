# Moment of inertia with a torsion pendulum

**Goal.** Determine the moment of inertia I (kg·m²) of the unknown object about its own symmetry axis.

**Apparatus.** A dark grey turntable turns about a vertical axis on a torsion spring; its own moment of inertia and the spring constant are not known. Its top is flat, and a magenta circle is painted around its axis. A yellow post standing on the turntable can be pushed to twist the turntable; it then oscillates freely. A red photogate fork straddles the turntable's rim. Two objects: a standard solid disc, coloured blue (mass {standard_disc_mass_kg} kg, moment of inertia {standard_disc_moment_of_inertia_kgm2} kg·m² about its axis), and the unknown object, coloured green (mass {unknown_object_mass_kg} kg, the same outer shape; its internal mass distribution is not known). Each carries the same handle on its axis: a bronze post under a round knob. In the submission, `load` names them `standard_disc` (blue) and `unknown_object` (green); `empty` is the turntable with nothing on it. Setting an object down or touching the turntable disturbs it; the spring's own damping is small.

**Storage box.** The objects start in a grey-green storage box fixed to the table. It has two seats, numbered 1 and 2 in white print; seat 1 holds the disc and seat 2 the unknown object.

**Damper lever.** An eddy-current damper acts on the turntable: on a dark pedestal, a carriage with a yellow post slides between two spring detents printed OFF and ON. A dark-red magnet on the carriage stands over a copper vane at ON. At ON the damper brakes the turntable strongly (a swing dies out within a few seconds); at OFF it has no effect; in between it brakes partly. It can be moved at any time; nothing locks it.

**Instrument.** A photogate at the turntable's rest angle; a flag on the rim interrupts its beam twice per oscillation period. It is operated from its recorder panel: a dark console with 12 push keys labelled 0 to 9, CLR and REC (REC is red), and a display. Digits enter the record length in whole seconds (at most three digits), CLR clears the entry, and REC takes it: {photogate_record_s_text} s starts a record of that length at once; any other entry shows ERR and does nothing. A record runs its full length and every key is ignored until it has ended. A record still running when the budget ends is lost. The display shows a state word (IDLE, REC, DONE or ERR) and three numbers: SET S (the entry being typed, or the length of the record running), T S (seconds recorded so far during REC, the record's length at DONE) and NO (the number of the record being made or last made). The display is read only in the camera images: `get_robot_state` returns the robot's own state and carries no instrument readings. Records are numbered 1, 2, … in the order they are made. `export_photogate(evidence_id)` returns a finished record and its CSV (`transit_id, t_block_s, t_clear_s`): the times at which the flag blocked and cleared the beam, in simulated seconds from REC, with {photogate_time_resolution_s} s resolution; `evidence_id` may also be the record's number or "latest".

At the start both objects are in their seats, the damper lever is at OFF, the turntable is at rest and the recorder is idle.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- At least {min_points} measurements marked `used_in_fit: true`, covering the empty turntable, the standard disc alone and the unknown object alone, each referencing a different photogate record of this episode. `load` must name what rested on the turntable during that record.
- Each cited record contains at least one complete oscillation period of consecutive clean transits, and its `period_s` agrees within {period_tol_pct}% with the period the evaluator measures on the longest run of clean transits of that record.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Tidy up: when you submit, both objects are back in seats of the storage box (any seat), no record is running and the damper lever is at OFF.
- Submit exactly once; the first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- An object counts as on the turntable when, over the last 1/3 s, it pressed on the platter with more than half its weight, the gripper did not hold it and no part of the robot touched it. The load is taken when REC starts a record and again when the record ends; both must be the load you name.
- A transit is clean when, from the end of the previous transit to its own end, no part of the robot touched the turntable (its yellow post included) or an object on the platter (one pressing on it then, or when the record started; touching an object elsewhere does not matter), the damper's magnet did not overlap the vane, and the set of objects pressing on the platter with more than 0.02 N has not changed since REC (nor was a held object among them). The record itself holds only the times; the evaluator keeps the bench's condition of every transit.
- The beam counts as blocked while the turntable is within ±0.0055 rad (0.32°) of its rest angle; the flag itself collides with nothing.
- A key acts once per press, when its cap is pressed down at least 4 mm (of its 9 mm travel); it acts again only after rising back above 2 mm.
- The damper lever's detent spring pulls it to the nearer detent; it counts as at a detent within 4 mm of it. The damper's viscous torque is in proportion to the magnet's overlap with the vane (full at ON, none at OFF).
- An object counts as in a seat of the storage box when the robot does not hold it, it stands upright (tilted at most {storage_box_seat_tilt_deg}° from vertical) and it is within 20 mm, horizontally and vertically, of where an object resting on that seat's floor stands.
- The display and the printed labels are visual only. The console, keys, display and its stand, lever, pedestal and storage box collide.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.** Relative error e_rel = |I_submitted − I| / I. Success when all conditions hold and e_rel ≤ {threshold_pct}%. Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
