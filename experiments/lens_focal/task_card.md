# Focal length of a converging lens

**Goal.** Determine the focal length f (mm) of the converging lens `L`.

**Apparatus.**
- **Light box.** A dark box whose face shows an illuminated letter F while it is on: that face is the object plane.
- **Rail.** A dark grey optical rail carries two riders.
- **Rail scale.** A white scale strip along the rail, graduated in millimetres from 0 to 680 mm: black ticks every 1 mm, longer every 5 mm and every 10 mm, and the number of every 10 mm mark. The scale reads distance from the object plane (0 mm).
- **Riders.** Two grey riders, A and S, each with a black rubber bumper on both ends. A rider's position is that of its index mark: a thin red line across the block top and, at the tip of a dark pointer reaching over the scale's ticks, a red index line on the scale. Each rider has an orange push handle on a grey arm, its letter on top. A rider is moved by pushing it along the rail with the robot: the riders have no motor, no clamp and no lock; friction holds a rider wherever it is released. They cannot pass each other: their bumpers stop them with the index marks {min_gap_mm} mm apart. Travel: A {travel_mm_A_text} mm, S {travel_mm_S_text} mm on the scale (end stops).
- **Rider A** carries a lens seat (orange collar) at its index mark. **Rider S** carries the screen: a white card in a light frame, printed with a faint millimetre grid (lines every 1 mm, darker every 5 mm and darkest every 10 mm, counted from the card's centre). The screen surface lies at S's index mark.
- **Lens.** Lens `L` (a pale blue disc, clear aperture {lens_aperture_mm} mm in diameter) stands in its holder in rider A's seat. Where the lens's optical centre sits relative to A's index mark is not stated (holders and lens thickness differ).
- **Light box switch.** A red round POWER push button on the light box and a green pilot lamp that lights while the light box is on. The light box starts off; each press switches it on or off (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm). With the light box off the F is dark and nothing is projected on the screen.
- **Holders and seats.** A lens holder (50 g) has a dark grey square foot, a thin post carrying the element and a white handle on top (a stem under a rim disc with a small cap); `L` is printed on the rim. Seated on rider A, its element is on the optical axis. Every seat here is a raised collar with a square well and a chamfered mouth; a part stands in a seat only when its foot rests on the well floor. A lens stand (a dark base with three dark teal seat collars) stands on the table; its seats are empty.

**Instrument.** The rail scale, the riders' index lines, the light box and the screen are seen in the three camera views (the wrist camera shows the scale and the screen in detail when brought close). Every `get_observation` returns the three camera views and an `evidence_id` bound to that exact physical state. Readings are taken by you from those images.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- Each row of `measurements` references a different `evidence_id` from this episode. Its `lens_mm` and `screen_mm` are the positions of riders A and S at that observation, read from the scale; each must lie within {position_tol_mm} mm of the rider's actual index mark at that moment. The readings of a row not marked `used_in_fit: true` are not checked.
- At least one row is marked `used_in_fit: true`, and every row so marked was observed with lens L seated on rider A, the light box on, and the image of the F on the screen in focus (blur radius no larger than {max_blur_mm} mm).
- Before submitting, switch the light box off and put every lens holder upright into a seat (a rider's seat or a lens stand seat): at submission the light box is off and each holder stands in a seat.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Submit exactly once; the first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- An element counts as seated on rider A only when its holder stands in the rider's seat not held, its release complete, tilted no more than 10°, and within 6 mm horizontally and 3 mm vertically of the well floor (4 mm horizontally in the lens stand's seats). Once an element counts as seated on a rider, it keeps counting while its holder stays within 9 mm horizontally and 9 mm vertically of the well floor; the other conditions still apply.
- A seated element sits exactly at rider A's index mark plus its fixed offset: where it stands inside the well, its tilt and its rotation make no difference.
- The screen image is computed from the elements seated on the riders only. A holder on the stand, a held element, the arm or anything else in the beam neither blocks nor changes it.
- A rider moves only when pushed: it takes {rail_friction_N} N along the rail to slide one, and a rider released while sliding stops within about 1 mm. Seating or lifting a holder does not move a rider.
- The screen image is updated whenever the light box is switched, an element is seated or lifted, or a rider has moved more than 0.1 mm, at most 5 times per simulated second; at every `get_observation` it shows exactly the state of that moment.
- The lens disc, the screen card, the index pointers and lines and the illuminated F have no collision; the holder's foot, post and handle (except its small top cap), the riders with their bumpers, arms, handles and the screen's frame, the rail, the scale strip, the light box and its button, and the lens stand do.
- A button registers once when its cap reaches the switch depth, and again only after the cap has risen to within 2 mm of rest.
- Blur is the radius of the blur disc on the screen, set by the lens's clear aperture.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.** e_rel = |f − f_true| / f_true. Success when all conditions hold and e_rel ≤ {threshold_pct}%. Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
