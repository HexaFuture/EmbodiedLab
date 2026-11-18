# Thermal: the B constant of an NTC thermistor

**Goal.** Determine the B constant (K) and the resistance at 25 °C, R25 (Ω), of the thermistor probe. B is the constant of the usual NTC exponential relation between resistance and absolute temperature, referred to T25 = 298.15 K.

**Apparatus.**
- **Dry block.** A grey block with an orange collar seat on top; `DRY BLOCK 25-80 C` is printed on the table beside it. It has a heater and no cooler. Its controller drives the block toward the set point: it heats at up to about {block_heating_K_per_min} K per minute and then settles at the set point; with the set point below the block temperature the heater stays off and the block cools by itself, slowly, toward the room. The set point starts at {block_start_set_C} °C and moves by {block_set_step_C} °C per press, within {block_set_min_C}–{block_set_max_C} °C.
- **Controller.** The screen on the block's face shows `SET` (the set point, whole °C) and `BLOCK` (the block's built-in reference thermometer, resolution {meters_thermometer_resolution_C} °C, reading noise {meters_thermometer_noise_C} °C standard deviation, refreshed at {meters_refresh_hz} Hz). The red `UP` cap raises the set point, the purple `DN` cap lowers it; they are labelled `SET +5 C` and `SET -5 C` on the table (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm).
- **Thermistor probe.** A dark grey stainless probe with a white handle on top. At the start it stands in the teal collar of its grey stand (`THERMISTOR PROBE` is printed on the table beside it). A part stands in a seat only when its square foot rests on the well floor. In the block's well it takes up the block's temperature; elsewhere it relaxes to the room. It does not respond instantly.
- **Ohmmeter.** A grey box on a post labelled `OHMMETER`, wired to the probe. Its 4½-digit LCD ({ohmmeter_counts} counts) shows the probe's resistance with the unit `Ohm` or `kOhm` and, in autorange, `AUTO`; `OL` means the resistance is above the range. It measures and refreshes at {meters_refresh_hz} Hz; each measurement has a relative noise of {ohmmeter_noise_rel} (standard deviation). Ranges, resolution and test currents (data sheet): 200 Ω (0.01 Ω) at 1 mA, 2 kΩ (0.1 Ω) at 1 mA, 20 kΩ (1 Ω) at 100 µA, 200 kΩ (10 Ω) at 10 µA. It starts in autorange, which uses the most sensitive range the resistance fits. The blue `RNG` cap (labelled `RANGE`) leaves autorange and holds the present range; each further press steps to the next range (after 200 kΩ back to 200 Ω). The green `AUTO` cap returns to autorange (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm). The test current flows through the probe all the time.

**Readings.** Every reading is read from the displays in the camera images. Each `get_observation` returns an `evidence_id` (`<episode_id>:obs:<n>`) for the moment its images show.

**Acceptance conditions** (the evaluator checks these):
- `points` lists your measured points (`used_in_fit: true` for those in your fit; at least one point must be marked `used_in_fit: true`). Each point cites the observation it was read from (`evidence_id`, a different observation for each point); its `T_C` (the controller's `BLOCK` reading) and `R_ohm` (the ohmmeter reading, in ohms) are what the displays showed in that observation.
- At the observation of a point used in the fit, the probe stands in the block's well.
- How many points you use, and at which temperatures, is your choice.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Tidy up: when you submit, the probe stands in its own stand, the set point is back at {block_set_min_C} °C, the ohmmeter is back in autorange (`AUTO`) and the hand is empty.
- Submit exactly once. The first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- The probe follows the block only while it stands in the block's well not held, its release complete, tilted no more than 10°, and within 6 mm horizontally and 3 mm vertically of the well floor. Anywhere else, held in the well included, it relaxes toward the room temperature with a time constant of 60 s.
- The ohmmeter's test current heats the probe as a real thermistor's: by the power it dissipates in the probe, divided by the probe's dissipation constant (not published), with the probe's own lag.
- Each `UP`, `DN`, `RNG` or `AUTO` press acts once: the cap must reach the switch depth, and rise to within 2 mm of rest before the next press counts. Set-point presses beyond the range are ignored.
- Displays: the controller and the ohmmeter refresh at {meters_refresh_hz} Hz and show the latest refresh until the next, rounded to the display resolution: `SET` whole °C, `BLOCK` {meters_thermometer_resolution_C} °C, the ohmmeter the resolution of its present range (data sheet above).

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing is free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.**
- e = max(|B − B_true| / B_true, |R25 − R25_true| / R25_true).
- Success when all conditions hold and e ≤ {threshold_pct}%.
- Reward = G × max(0, 1 − e / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
