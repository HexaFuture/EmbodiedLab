# Spring stiffness (Hooke's law)

**Goal.** Determine the stiffness k (N/m) of the spring that supports the tray.

**Apparatus.** A tray hangs from a vertical spring and slides along a guide. A red pointer fixed to the tray moves along a vertical ruler graduated in {ruler_unit}. The tray is orange-brown; its public mass is {tray_mass_kg} kg. {n_weights} identical weights of {weight_mass_kg} kg each, red cylinders with a brass handle on top, stand in a grey-green weight box fixed to the table. The box has six seats, each numbered on the box and each holding one weight upright; the weights start in seats 1 to {n_weights} and the remaining seats start empty. The weights are identical and unmarked; they need not be told apart (any weight, any free seat). Full-load tray travel stays within {full_load_travel_max_m} m. Use g = {gravity_for_conversion_m_s2} m/s^2 to convert mass to force.

**Instrument.** The ruler is read against the red pointer. Every `get_observation` returns the three camera views and an `evidence_id` bound to that exact physical state. Readings are taken by you from those images.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- The measurements you mark `used_in_fit: true` cover at least {min_loads} different loads; each references an `evidence_id` from this episode (an `evidence_id` may appear in only one row) taken while the tray was at rest and the robot was not touching the tray or its load. `mass_kg` must equal the load actually on the tray at that moment.
- The evaluator refits the readings you mark `used_in_fit: true` with the standard physical model of this measurement; your submitted k must agree with that refit within {self_consistency_pct}%.
- At submission, the gripper is open and empty.
- Before submitting, put every weight back into a seat of the weight box (any free seat): at submission each weight stands in a seat.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Submit exactly once; the first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- A weight counts as in a seat of the weight box when the robot does not hold it, it stands upright (tilted at most 10° from vertical) and it is within 20 mm, horizontally and vertically, of where a weight resting on that seat's floor stands.
- A weight counts as on the tray only if it stands directly on the tray deck (its bottom within 4 mm of the deck top) and, over the last 1 s, its support force on the deck averaged 0.49 ± 0.06 N and never fell below 0.2 N. A weight resting on another weight does not count.
- The tray counts as at rest when, over the last 1 s, its position varied by less than 0.8 mm peak-to-peak and it never moved faster than 2 mm/s.
- "Not touching the tray or its load" means that during the last 1 s no part of the robot touched the tray or a weight standing on the tray deck, and no weight held by the robot pressed on the deck (support force above 0.1 N). A weight held clear of the tray does not matter.
- The load of a reading is the number of weights counted on the tray at that moment times {weight_mass_kg} kg; `mass_kg` is compared with it.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.** Relative error e_rel = |k_submitted − k| / k. Success when all conditions hold and e_rel ≤ {threshold_pct}%. Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
