# Spring launcher: predict the landing point and catch the ball

**Goal.** Predict where a ball fired from the launcher's target notch (plunger compression {target_notch_m} m) crosses the plane of the receiving light curtain, and have the cup there to catch it. The spring stiffness is unknown. No ball may be fired from the target notch before you submit; any other notch may be test-fired as often as you like.

**Apparatus.**
- Launcher: it launches at an elevation of 10°. A coil spring drives the plunger. Pushing the knob back along the launch axis compresses the spring; the pawl on the knob drops into the deepest tooth of the ratchet rack it has passed and holds that notch, and only the trigger button lifts it. A ball resting in the guide groove against the plunger face is launched when the trigger frees the plunger; the trigger with no ball releases the empty plunger.
- Trigger: a purple push-button cap, with a white plate labelled `FIRE` lying beside it.
- Notch scale: a white plate with one black tick per notch, numbered in mm (15 to 50), at the knob position that latches that notch. The green pointer on the knob shows the plunger compression against the ticks.
- Receiving table: a ball that has flown ends up in its V-shaped return groove.
- Cup: blue, square, with a yellow grip post on one wall. It does not start on the receiving table: it starts on its own dark pad.
- Ball: orange-red; it starts on the ball pad, a light pad with a shallow seat on top of a green column.
- The ball model, the masses and the launcher's published constants are in `get_task_spec().apparatus`; everything on the bench is seen through the cameras, and measurements come only from the public instruments.

**Instruments.** The flight timer keeps one record per shot: the raw edges of the two photogates the ball passes as it leaves the launcher and the point where the ball centre crosses the plane of the receiving light curtain. `export_flight()` returns every stored record of this episode, oldest first; `export_flight(evidence_id)` returns one. A record holds `record_number`, `evidence_id`, `gate_events` (`channel`, `edge`, `t_s`), `gate_separation_m`, `landing` (`distance_m` from the muzzle, `lateral_m`, `t_s`; `null` when the ball did not cross the plane), `distance_origin` (the name of the reference point `distance_m` is measured from: always `muzzle`, the ball centre as it leaves the ends of the launch rails) and `receiver_center_z_m` (the level of the curtain plane in the timer's own reference frame: a constant of the timer, the same in every record). The timer does not know the notch: read it from the scale.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- `predicted_distance_m` is the horizontal distance from the muzzle at which the ball centre crosses the plane of the receiving light curtain (not where the ball reaches the bottom of the cup). `measurement_evidence_ids` cite distinct raw records of this episode, made before the submission, that are complete and free of robot intervention: each has exactly one edge from each of the two photogates and a `landing` that is not `null`.
- No ball was fired from the target notch before the submission; every shot is audited.
- At submission the launcher is locked at the target notch with the ball in place, the cup stands released on the receiving table, and the arm touches neither the ball, the cup nor the plunger.
- After the submission the system releases the same pawl once (4 s, not charged to the budget; nothing places or pushes the ball: the spring launches it as in a shot of your own). It succeeds only if the measured crossing point is within {threshold_pct}% of the prediction, the ball enters the cup and stays in it for at least 0.5 s, and the cup stays still from launch to measurement (displacement ≤ 2 mm, rotation quaternion inner product ≥ 0.999).
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing. The ball is exempt from the tilt check.
- Submit exactly once; the first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- The light curtain is a fixed horizontal plane in space; it does not move with the cup.
- The trigger registers when the robot holds its cap at least 5 mm down for 0.04 s; presses during the 4 s recording of a shot do nothing.
- A shot is recorded only when a notch is latched and the ball lies on the rails: within 6 mm of the launch line, from 4 mm behind to 150 mm ahead of its rest point. It counts as seated only when the ball is within 3 mm of the plunger face and moves slower than 3 mm/s; valid records and the final shot need a seated ball. A trigger with no ball on the rails is not recorded.
- A shot record made before the submission is invalid if, from the trigger until the ball crosses the curtain, the robot touches the ball or the plunger, the gripper holds the ball, or the ball touches the cup (its grip post included); what happens after the crossing does not matter. On the final shot after the submission, the robot must not touch the ball, the cup or the plunger, nor the gripper hold anything, within 4 s of the trigger, and the ball must not touch the cup (its grip post included) before it crosses the curtain.
- Any recorded shot from the target notch before the submission fails the episode, even an invalid one.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.** Relative error e_rel = |d_predicted − d_measured| / d_measured, where d_measured is the crossing distance measured on the shot released after the submission. Success when all conditions hold and e_rel ≤ {threshold_pct}%. Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
