# Mechanics: tune a vibration absorber on a shaken cantilever

**Goal.** A steel strip, clamped at one end, is shaken by an exciter while the analyzer sweeps the drive frequency. Tune the dynamic absorber on the strip (its mass, its blade length and its damper valve) so that the largest vibration amplitude of the strip's tip over the sweep is as small as possible. Report the PEAK the analyzer shows for your final sweep.

**Apparatus.**
- **Cantilever.** A steel strip clamped in a cast-iron vise; a small accelerometer at its tip is cabled to the analyzer. An electrodynamic exciter drives the strip through a thin push rod. Its power amplifier has no controls.
- **Absorber.** Clamped on the strip; it cannot be removed. A short spring-steel blade runs from the absorber's mount. A sliding carrier clamps the blade; it carries a tray with {tray_sockets} square sockets and an orange push handle. Each socket has a hold-down: a spring slide clamp on a short rail, with a red push tab. Pushed fully forward, the slide drops into a notch at the end of its rail: its forked nose comes down on the top of the body of the weight in that socket, its spring presses the weight into the socket, and the notch keeps the slide closed. Pushed back, the slide climbs out of the notch; pushed back to the end of its rail, its nose clears the body of the weight. A slide stays where it is left. At the start every hold-down is closed. Push the handle along the blade: an index pin clicks the carrier into detent holes every {blade_detent_step_mm} mm, the blade's free length (from its root clamp to the carrier) printed at each hole, from {blade_detent_first_mm} to {blade_detent_last_mm} mm. Rubber stops limit the carrier's travel up and down (nominal gap ±{stops_gap_mm} mm from the unloaded blade). A small hydraulic damper connects the carrier to the mount; its cylinder carries the valve keys `−` and `+` and a one-digit display showing the valve setting, 0 to 9 (each press moves it one step).
- **Weights.** {weight_count} brass weights, {weights_g_text} g, each with its mass printed on it (square foot, round body, white grip). They start in the dark weight rack. A weight stands in a tray socket or a rack seat foot first.

**Instrument.**
- **Vibration analyzer.** Press the green `SWEEP` key (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm): the exciter drives the strip with a force of nominal amplitude {exciter_force_n} N while its frequency rises steadily from {sweep_f_lo_hz} to {sweep_f_hi_hz} Hz over {sweep_duration_s} s. The screen shows the RUN lamp during a sweep, PEAK (the largest displacement of the tip from its rest position so far, mm, {analyzer_peak_decimals} decimals) and FREQ (the drive frequency at that peak, Hz, {analyzer_hz_decimals} decimal); after the sweep they stay until the next one. The screen refreshes {analyzer_refresh_hz} times per simulated second while a sweep runs.
- A sweep is void if anything touches the strip or the absorber while it runs, or if the absorber's configuration changes: the screen shows `Err`. `SWEEP` pressed while the carrier is not at rest in a detent shows `Err` and starts nothing. The valve keys do nothing while a sweep runs.
- **Observations and export.** Every `get_observation` is evidence of the state it shows: it returns `evidence_id` (`<episode_id>:obs:<n>`). `export_sweeps()` returns what the analyzer stores of every sweep of this episode that has ended: its `evidence_id` (`<episode_id>:sweep:<n>`), its status (`complete` or `void`), its start and end times, and the amplitude-frequency curve (the largest tip displacement in each 0.1 Hz band of the drive frequency). PEAK and FREQ are shown on the analyzer's screen only, the valve setting on the damper's display only: read them in the camera images.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- `sweep_evidence` cites a complete sweep of this episode, and the absorber has not been changed since that sweep started.
- `peak_reading` cites an observation (`evidence_id` from `get_observation`) whose images show the analyzer's screen with that sweep's PEAK after the sweep has ended, and `peak_amplitude_mm` is that PEAK within 0.002 mm (two counts of its last digit).
- `configuration` is the absorber as it stands when you submit: the total mass of the weights in the tray (g, within {mass_tol_g} g), the blade length of the carrier's detent (mm) and the valve setting.
- Leave the bench in order: when you submit, no loose item is tipped over or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Tidy up: when you submit, no sweep is running, and every weight not in the tray stands in a rack seat.
- Submit exactly once. The first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- A weight is in the tray when it is not held and stands upright in a socket, its foot down on the socket's floor; only such weights count for the absorber's mass. The carrier is at rest in a detent when its index pin is within 0.5 mm of the hole and it moves slower than 2 mm/s (position and speed both averaged over half a second).
- After you submit, the evaluator runs one more sweep of the absorber as it stands, starting from rest; that sweep's peak is A.
- The carrier takes about {carrier_detent_n} N along the blade to push it out of a detent and about {carrier_friction_n} N to slide it between detents.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing is free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.**
- A is the tip's peak amplitude over the evaluator's sweep of your absorber; A* is a reference value: the smallest such peak that an offline calculation over the configurations this absorber allows (any set of the weights in the tray, any detent, any valve setting) found; an A smaller than A* counts as A*.
- e = max((max(A, A*) / A* − 1) / {threshold}, |peak_amplitude_mm − A| / ({reported_threshold} · A)).
- Success when all conditions hold and e ≤ 1: A is at most (1 + {threshold}) × A*, and the reported peak is within {reported_threshold_pct}% of A.
- Reward = G × max(0, 1 − e).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
