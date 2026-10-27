# Mechanics: bending stiffness of a strip and the compliance of its clamp

**Goal.** A thin metal strip (material unknown) is clamped at one end and sticks out horizontally as a cantilever. The clamp is not perfectly rigid: under the moment at the strip's root it turns by an angle proportional to that moment. Determine:
- the strip's bending stiffness EI, in N m²;
- the clamp's rotational stiffness k_θ (moment per angle), in N m/rad.

**Apparatus.**
- **Strip.** It leaves a grey clamp labelled CLAMP and sticks out horizontally. Its own weight is already carried in the unloaded position. The strip is solid: things can stand on it anywhere, and it bends under whatever presses on it — a weight set on it, a saddle, the robot's fingers. A rubber stop under its free end lets that end go down by at most 30 mm.
- **Knife-edge saddles.** Four load saddles ride on the strip; each one's distance from the clamp edge is printed beside it. Each is a level grey block on a steel knife edge laid across the strip, with an orange collar on top; everything a saddle carries presses on the strip through its knife edge. Every seat here (saddles and rack) is a raised collar with a square well and a chamfered mouth; a weight stands in a seat when its foot rests on the well floor.
- **Weights.** Three steel weights of {weights_g_0} g, {weights_g_1} g and {weights_g_2} g start in the teal collars of a dark rack, one per collar; the label on top of each weight shows its mass. Weights of equal mass are identical and need not be told apart. Each weight, from the bottom: a grey square foot, a steel cylinder, a dark stem, then a white handle (a stem under a rim disc). Take g = {g_m_s2} m/s².

**Instrument.**
- **Digital indicator on a rail.** A rail with detent holes; each hole's distance from the clamp edge is printed beside it on a white strip. The indicator rides on a carriage on this rail. An arm from the carriage reaches under the strip, and the plunger at its end touches the strip's underside at the distance of the hole the carriage stands in (the carriage's red index line points at that hole's number). The indicator reads the strip's height against the rail there, downward = + (plunger pushed in = +; resolution {gauge_resolution_mm} mm), on a seven-segment LCD on the carriage that shows the present reading in mm with three decimals. The rail and the unloaded strip are not perfectly straight, so the unloaded reading differs from hole to hole. It starts at an arbitrary reading.
- **Moving the carriage.** Push the orange handle on the carriage along the rail. A ball detent clicks the carriage into a hole when it comes within a few millimetres of it, holds it there, and lets it go when pushed on. The green round `ZERO` cap on the carriage sets the reading to 0.000 where the carriage stands when it is pressed; at another hole the reading is then the strip's height there less its height where `ZERO` was pressed (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm).
- **Observations.** Every `get_observation` is evidence of the state it shows: it returns `evidence_id` (`<episode_id>:obs:<n>`). Readings are taken from the LCD in its images. The indicator has no logger and no export.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- `readings` lists the indicator readings you used. Each row cites a different `evidence_id` of this episode, the hole the carriage stood in (`hole_mm`, the printed number) and the reading the LCD showed in that observation's images (`reading_mm`). At that observation the carriage must stand in that hole, at rest (not between holes, not moving), and the reading must equal what the LCD showed then within {tol_mm} mm.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Tidy up: when you submit, all three weights stand in rack collars (any collar), nothing rests on or presses on the strip or its saddles, the indicator carriage stands in a hole (any hole), and the robot holds nothing.
- Submit exactly once. The first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- The strip is loaded by every contact on it, whatever the body: a contact with the strip by its force normal to the strip at the point of contact, a contact with a saddle by its vertical force at that saddle's knife edge. A weight resting in a saddle loads the strip there with its weight; one standing on the bare strip loads it where it stands. The indicator's plunger applies no force to the strip (it touches nothing physically and loads nothing).
- Weights have their labelled mass everywhere (in the hand, on the table, on the strip).
- After a load change the deflection follows with a first-order lag of 0.25 s (within 0.1% after about 1.7 s).
- The LCD refreshes {gauge_refresh_hz} times per simulated second; each value it shows carries its own Gaussian noise of 0.0015 mm (1σ). An observation's images show the LCD as it is at that moment.
- The carriage stands in a hole when its index is within {hole_tol_mm} mm of the hole and it moves slower than {max_speed_mm_s} mm/s. It takes about {carriage_detent_n} N along the rail to push it out of a hole and about {carriage_friction_n} N to slide it between holes.
- `ZERO` registers once when its cap reaches the switch depth, and again only after the cap has risen to within 2 mm of rest.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing is free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.**
- e = max(|EI − EI_true| / EI_true, |k_θ − k_θ,true| / k_θ,true).
- Success when all conditions hold and e ≤ {threshold_pct}%.
- Reward = G × max(0, 1 − e / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
