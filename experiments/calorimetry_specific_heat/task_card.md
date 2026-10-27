# Thermal: the specific heat of a metal sample

**Goal.** Determine the specific heat capacity c_s, in J/(kg K), of the metal sample.

**Apparatus.**
- **Heater block.** A dark red block with an orange collar seat on top and a red power lamp; `HEATER 25-95 C` is printed on the table beside it. It has a heater and no cooler. Its controller drives the block toward the set point: it heats at up to about {heater_heating_K_per_min} K per minute and then settles at the set point; with the set point below the block temperature the heater stays off and the block cools by itself, slowly, toward the room. The set point starts at {heater_start_set_C} °C and moves by {heater_set_step_C} °C per press, within {heater_set_min_C}–{heater_set_max_C} °C.
- **Heater controls.** The controller screen on the block's face shows `SET` (the set point, whole °C) and `BLOCK` (the block's own thermometer, resolution {heater_resolution_C} °C, reading noise {heater_noise_C} °C standard deviation, refreshed at {thermometer_refresh_hz} Hz). The red `UP` cap raises the set point, the purple `DN` cap lowers it; they are labelled `SET +5 C` and `SET -5 C` on the table (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm).
- **Sample.** A {m_s_g} g block of black-painted alloy with a white handle on top; its mass is printed on the table beside the heater. At the start it stands in the heater's well, at room temperature. A part stands in a seat only when its square foot rests on the well floor. The sample takes up the block's temperature in the well, but not instantly.
- **Calorimeter cup.** An insulated white cup with a teal seat collar on its floor, under {m_w_g} g of clear water; the water line is drawn as a pale blue ring on the inner wall. The cup's own heat capacity is {c_cup_J_K} J/K. The water mass and the cup's heat capacity are printed on the table beside the cup. The water's specific heat is {water_c_J_kgK} J/(kg K). A thin thermometer probe and a dark stirrer shaft dip into the water.
- **Stirrer.** It runs from the start. The green `STIR` cap switches it on or off (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm); the green lamp labelled `STIR ON` beside it is lit while the stirrer runs.
- **Digital thermometer.** A grey box on a post labelled `WATER`, wired to the probe in the cup. Its LCD shows the water temperature in °C (resolution {thermometer_resolution_C} °C, reading noise {thermometer_noise_C} °C standard deviation, refreshed at {thermometer_refresh_hz} Hz).
- **Recorder.** Built into the digital thermometer and linked to the heater's controller. The blue `REC` cap (`RECORD` is printed on the table beside it) starts a record; the next press ends it (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm). While a record runs, the thermometer's LCD shows `REC` and the record's number (`NO`), and every display refresh of the water thermometer and of the heater's `BLOCK` thermometer is recorded. Nothing is recorded while no record runs; at the start none runs.
- **Heat exchange.** The room is at {t_room_C} °C, and the water does not start at room temperature. The cup exchanges heat with the room, and the sample cools in the air whenever it is not standing in the heater well.

**Tool.**
- `export_temperature_log()`: the recorder's records so far, each with its `record_id` (`<episode_id>:thermo_rec:<n>`, n as `NO` on the LCD), its start and end time `started_s`, `ended_s` (null while it runs) and its rows (`sample_id`, `t_s`, `T_C` the water thermometer, `heater_C` the heater's block thermometer), as CSV with SHA-256. It does not advance time.
- Everything else is read from the displays in the camera images.

**Acceptance conditions** (the evaluator checks these):
- `readings` gives the three readings you used, each citing the record it comes from (`record_id`) and a window `t_from_s`–`t_to_s` of it: `heater` (the heater's block thermometer before the sample left the heater), `water_before` (the water before the sample entered it) and `water_after` (the water after it). Each `T_C` lies within the recorded values of its column in its window, as your latest export of that record before submitting returned them.
- The sample entered the water at a moment between the `water_before` and `water_after` windows and after the `heater` window, and the sample stays in the water until the end of the `water_after` window.
- `T_sample_C`, `T_water0_C` and `T_final_C` are the temperatures your result uses; how you obtain them from the readings is your choice.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Tidy up: when you submit, the sample stands in a seat (the heater well or the cup's collar), the stirrer is off, the recorder is not recording (no record runs), the heater set point is back at {heater_set_min_C} °C and the hand is empty.
- Submit exactly once. The first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- The sample takes up the block's temperature only while it stands in the heater well not held, its release complete, tilted no more than 10°, and within 6 mm horizontally and 3 mm vertically of the well floor. Anywhere else, gripped in the well included, it cools as in air.
- The sample is in the water while the centre of its metal body is inside the cup and below the water line, whatever its tilt, held in the gripper or not; there it exchanges heat with the water fully, and anywhere else not at all.
- The cup wall, the thermometer probe, the stirrer shaft and the water have no collision; the water exerts no buoyancy or drag and its level does not rise. Only the cup floor and its teal collar are solid.
- The sample has its published mass in the robot's hand and on the table.
- The thermometer reads the well-mixed water wherever the sample is. The stirrer changes the sample–water exchange and the probe lag. Each `STIR`, `UP`, `DN` or `REC` press acts once: the cap must reach the switch depth, and rise to within 2 mm of rest before the next press counts. Presses beyond the set-point range are ignored.
- Displays: the controller and the water thermometer refresh at {thermometer_refresh_hz} Hz and show the latest refresh until the next, rounded to the display resolution: `SET` whole °C, `BLOCK` {heater_resolution_C} °C, the water {thermometer_resolution_C} °C. While a record runs, each refresh of the two thermometers is one recorded row.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.**
- e_rel = |c_s − c_true| / c_true.
- Success when all conditions hold and e_rel ≤ {threshold_pct}%.
- Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
