# Refractive index of a glass block

**Goal.** Determine the refractive index n of the glass block `P`.

**Apparatus.**
- **Light box.** A dark box whose face shows an illuminated letter F while it is on: that face is the object plane.
- **Rail.** A dark grey optical rail carries three riders.
- **Rail scale.** A white scale strip along the rail, graduated in millimetres from 0 to 680 mm: black ticks every 1 mm, longer every 5 mm and every 10 mm, and the number of every 10 mm mark. The scale reads distance from the object plane (0 mm).
- **Riders.** Three grey riders, A, B and S, each with a black rubber bumper on both ends. A rider's position is that of its index mark: a thin red line across the block top and, at the tip of a dark pointer reaching over the scale's ticks, a red index line on the scale. Each rider has an orange push handle on a grey arm, its letter on top. A rider is moved by pushing it along the rail with the robot: the riders have no motor, no clamp and no lock; friction holds a rider wherever it is released. They cannot pass each other: A and B stop with their index marks {min_gap_mm_A_B} mm apart (rider B's handle arm meets rider A's bumper), B and S with their index marks {min_gap_mm_B_S} mm apart (bumper to bumper). Travel: A {travel_mm_A_text} mm, B {travel_mm_B_text} mm, S {travel_mm_S_text} mm on the scale (end stops).
- **Riders A and B** each carry a holder seat (orange collar) at their index mark. **Rider S** carries the screen: a white card in a light frame, printed with a faint millimetre grid (lines every 1 mm, darker every 5 mm and darkest every 10 mm, counted from the card's centre). The screen surface lies at S's index mark.
- **Lens.** A converging lens `L` (a round pale blue disc, clear aperture {lens_aperture_mm} mm in diameter) stands in its holder in rider A's seat. Where the lens's optical centre sits relative to A's index mark is not stated. Rider B's seat is empty.
- **Glass block.** The block `P` (a rectangular amber plane-parallel glass block, {plate_thickness_mm} mm long along the beam when seated) stands in its holder in a seat of the lens stand.
- **Light box switch.** A red round POWER push button on the light box and a green pilot lamp that lights while the light box is on. The light box starts off; each press switches it on or off (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm). With the light box off the F is dark and nothing is projected on the screen.
- **Holders and seats.** A holder (50 g) has a dark grey square foot, a thin post carrying the element and a white handle on top (a stem under a rim disc with a small cap); the element's name is printed on the rim (`L` on the lens's holder, `P` on the block's). Seated on a rider, its element is on the optical axis. Every seat here is a raised collar with a square well and a chamfered mouth; a part stands in a seat only when its foot rests on the well floor. The lens stand (a dark base with three dark teal seat collars, numbered 1, 2 and 3 on white labels on their tops) stands on the table.

**Instrument.** The rail scale, the riders' index lines, the light box and the screen are seen in the three camera views (the wrist camera shows the scale and the screen in detail when brought close). Every `get_observation` returns the three camera views and an `evidence_id` bound to that exact physical state. Readings are taken by you from those images.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- The submission has two groups of measurements, `no_plate` and `plate`, each with at least one row. Each row references a different `evidence_id` from this episode (no observation is cited twice, within a group or across the groups). Its `lens_mm` and `screen_mm` are the positions of riders A and S at that observation, read from the scale; each must lie within {position_tol_mm} mm of the rider's actual index mark at that moment.
- Every cited observation was made with lens L seated on rider A, the light box on, and the image of the F on the screen in focus (blur radius no larger than {max_blur_mm} mm): in the `no_plate` rows with the block on no rider, in the `plate` rows with the block seated on rider B.
- Before submitting, switch the light box off and put every holder upright into a seat (a rider's seat or a lens stand seat): at submission the light box is off and each holder stands in a seat.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Submit exactly once; the first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- An element counts as seated on a rider only when its holder stands in the rider's seat not held, its release complete, tilted no more than 10°, and within 6 mm horizontally and 3 mm vertically of the well floor (4 mm horizontally in the lens stand's seats). Once an element counts as seated on a rider, it keeps counting while its holder stays within 9 mm horizontally and 9 mm vertically of the well floor; the other conditions still apply.
- A seated element sits exactly at its rider's index mark (the lens plus its fixed offset): where it stands inside the well, its tilt and its rotation make no difference. A block seated on B lies wholly between the lens and the screen.
- The screen image is computed from the elements seated on the riders only (paraxial optics). A holder on the stand, a held element, the arm or anything else in the beam neither blocks nor changes it.
- A rider moves only when pushed: it takes {rail_friction_N} N along the rail to slide one, and a rider released while sliding stops within about 1 mm. Seating or lifting a holder does not move a rider.
- The screen image is updated whenever the light box is switched, an element is seated or lifted, or a rider has moved more than 0.1 mm, at most 5 times per simulated second; at every `get_observation` it shows exactly the state of that moment.
- The lens disc, the glass block, the screen card, the index pointers and lines and the illuminated F have no collision; the holder's foot, post and handle (except its small top cap), the riders with their bumpers, arms, handles and the screen's frame, the rail, the scale strip, the light box and its button, and the lens stand do.
- A button registers once when its cap reaches the switch depth, and again only after the cap has risen to within 2 mm of rest.
- Blur is the radius of the blur disc on the screen, set by the lens's clear aperture.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.** e_rel = |n − n_true| / n_true. Success when all conditions hold and e_rel ≤ {threshold_pct}%. Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values). `no_plate` and `plate` are objects, each with a list `measurements`:
```json
{submission_example}
```
