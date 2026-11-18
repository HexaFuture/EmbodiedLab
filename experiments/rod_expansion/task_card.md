# Thermal: the linear expansion coefficient of a metal rod

**Goal.** Determine the linear expansion coefficient α (1/K) of the rod's alloy, and report the rod's length L0 at room temperature that you used.

**Apparatus.**
- **Dry block.** A grey block with an orange collar seat on top; `DRY BLOCK 25-80 C` is printed on the table beside it. It has a heater and no cooler. Its controller drives the block toward the set point: it heats at up to about {block_heating_K_per_min} K per minute and then settles at the set point; with the set point below the block temperature the heater stays off and the block cools by itself, slowly, toward the room. The set point starts at {block_start_set_C} °C and moves by {block_set_step_C} °C per press, within {block_set_min_C}–{block_set_max_C} °C.
- **Controller.** The screen on the block's face shows `SET` (the set point, whole °C) and `BLOCK` (the block's built-in thermometer, resolution {meters_thermometer_resolution_C} °C, reading noise {meters_thermometer_noise_C} °C standard deviation, refreshed at {meters_refresh_hz} Hz). The red `UP` cap raises the set point, the purple `DN` cap lowers it; they are labelled `SET +5 C` and `SET -5 C` on the table (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm).
- **Heating jacket and rod.** A black jacket, open on top (`HEATING JACKET` is printed beside it), fed from the block by two black hoses; it follows the block's temperature. The aluminium-coloured rod lies in it with both ends out: one end against a fixed dark blue-grey stop, the other against the plunger of the digital indicator. The rod does not take up the jacket's temperature instantly.
- **Steel rule.** A millimetre steel rule lies along the rod, numbered every 10 mm; its zero is at the fixed stop's face, where the rod's end rests.
- **Digital indicator.** A dark grey head labelled `INDICATOR` and `mm` on the rod's free end. Its LCD shows the plunger position in mm (resolution {meters_indicator_resolution_mm} mm, reading noise {meters_indicator_noise_mm} mm standard deviation, refreshed at {meters_refresh_hz} Hz); its zero is arbitrary, and the reading increases as the rod lengthens.
- **Thermometer probe.** A dark grey stainless probe with a white handle on top. At the start it stands in the teal collar of its grey stand (`THERMOMETER PROBE` is printed on the table beside it). A part stands in a seat only when its square foot rests on the well floor. In the block's well it takes up the block's temperature; elsewhere it relaxes to the room. It does not respond instantly.
- **Digital thermometer.** A grey box on a post labelled `PROBE`, wired to the probe. Its LCD shows the probe's temperature in °C (resolution {meters_thermometer_resolution_C} °C, reading noise {meters_thermometer_noise_C} °C standard deviation, refreshed at {meters_refresh_hz} Hz).

**Readings.** Every reading is read from the displays and the rule in the camera images. Each `get_observation` returns an `evidence_id` (`<episode_id>:obs:<n>`) for the moment its images show.

**Acceptance conditions** (the evaluator checks these):
- `points` lists your measured points (`used_in_fit: true` for those in your fit; at least one point must be marked `used_in_fit: true`). Each point cites the observation it was read from (`evidence_id`, a different observation for each point); its `T_C` (the probe's digital thermometer, labelled `PROBE`, or the controller's `BLOCK`) and `indicator_mm` (the digital indicator) are what the displays showed in that observation.
- At the observation of a point used in the fit whose `T_C` is the digital thermometer's reading (not the controller's `BLOCK`), the thermometer probe stands in the block's well.
- How many points you use, and at which temperatures, is your choice.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Tidy up: when you submit, the probe stands in its own stand, the set point is back at {block_set_min_C} °C and the hand is empty.
- Submit exactly once. The first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- The probe follows the block only while it stands in the block's well not held, its release complete, tilted no more than 10°, and within 6 mm horizontally and 3 mm vertically of the well floor. Anywhere else, held in the well included, it relaxes toward the room temperature with a time constant of 60 s.
- The rod follows the block at all times, wherever the probe is; the indicator depends on the rod's temperature only.
- The jacket, the rod, the stop, the rule and the indicator have no collision.
- Each `UP` or `DN` press changes the set point once: the cap must reach the switch depth, and rise to within 2 mm of rest before the next press counts. Presses beyond the set-point range are ignored.
- Displays: the controller, the digital thermometer and the digital indicator refresh at {meters_refresh_hz} Hz and show the latest refresh until the next, rounded to the display resolution: `SET` whole °C, `BLOCK` and the thermometer {meters_thermometer_resolution_C} °C, the indicator {meters_indicator_resolution_mm} mm.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing is free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.**
- e_rel = |α − α_true| / α_true.
- Success when all conditions hold and e_rel ≤ {threshold_pct}%.
- Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
