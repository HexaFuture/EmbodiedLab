# Slit width from single-slit diffraction

**Goal.** Determine the width a (mm) of the slit.

**Apparatus.**
- **Laser module.** A dark box printed `LASER`; while it is on, its beam runs along the rail. Its wavelength is {laser_nm} nm.
- **Rail.** A dark grey optical rail carries two riders.
- **Rail scale.** A white scale strip along the rail, graduated in millimetres from 0 to 680 mm: black ticks every 1 mm, longer every 5 mm and every 10 mm, and the number of every 10 mm mark. The scale reads distance from the laser's exit aperture (0 mm).
- **Riders.** Two grey riders, A and S, each with a black rubber bumper on both ends. A rider's position is that of its index mark: a thin red line across the block top and, at the tip of a dark pointer reaching over the scale's ticks, a red index line on the scale. Each rider has an orange push handle on a grey arm, its letter on top. A rider is moved by pushing it along the rail with the robot: the riders have no motor, no clamp and no lock; friction holds a rider wherever it is released. They cannot pass each other: their bumpers stop them with the index marks {min_gap_mm} mm apart. Travel: A {travel_mm_A_text} mm, S {travel_mm_S_text} mm on the scale (end stops).
- **Rider A** carries a holder seat (orange collar) at its index mark. **Rider S** carries the screen: a white card in a light frame, printed with a horizontal millimetre scale whose 0 is the card's centre, on the beam: ticks every 1 mm, longer every 5 mm and every 10 mm, from −55 to +55 mm, numbered −50 to 50 every 10 mm. The screen surface lies at S's index mark.
- **Slit.** A single vertical slit in a blackened plate across the beam (the light line drawn on the plate is wider than the slit) stands in its holder in rider A's seat. Where the slit plane sits relative to A's index mark is not stated. The diffraction pattern spreads sideways (horizontally) on the screen: a bright central maximum with dark fringes on either side. The dark fringe of order k is the k-th dark fringe counted outward from the central maximum, k positive on the positive side of the screen scale and negative on the negative side.
- **Laser switch.** A red round POWER push button on the laser module and a green pilot lamp that lights while the laser is on. The laser starts off; each press switches it on or off (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm). With the laser off nothing is projected on the screen.
- **Holders and seats.** The holder (50 g) has a dark grey square foot, a thin post carrying the element and a white handle on top (a stem under a rim disc with a small cap); `SL` is printed on the rim. Seated on rider A, its element is on the beam. Every seat here is a raised collar with a square well and a chamfered mouth; a part stands in a seat only when its foot rests on the well floor. A holder stand (a dark base with three dark teal seat collars, numbered 1, 2 and 3 on white labels on their tops) stands on the table; its seats are empty.

**Instrument.** The rail scale, the riders' index lines, the laser module and the screen are seen in the three camera views (the wrist camera shows the scale and the screen in detail when brought close). Every `get_observation` returns the three camera views and an `evidence_id` bound to that exact physical state. Readings are taken by you from those images.

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- Each row of `measurements` references a different `evidence_id` from this episode. Its `slit_mm` and `screen_mm` are the positions of riders A and S at that observation, read from the rail scale, each within {position_tol_mm} mm of the rider's actual index mark at that moment. Each entry of its `dark_fringes` gives a dark fringe's order and its position `x_mm` on the screen scale at that observation, a fringe on the card at least 1 mm inside its edge, within {spot_tol_mm} mm of the fringe's actual position; no order twice in a row. The readings of a row not marked `used_in_fit: true` are not checked.
- At least one row is marked `used_in_fit: true`, and every row so marked was observed with the slit seated on rider A, the laser on, and a dark fringe showing on each side of the central maximum.
- Before submitting, switch the laser off and put the holder upright into a seat (rider A's seat or a holder stand seat): at submission the laser is off and the holder stands in a seat.
- Leave the bench in order: when you submit, no loose item (anything not fixed to the apparatus) is tipped over (the direction that pointed up when the episode started now more than 45° from vertical) or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Submit exactly once; the first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- An element counts as seated on rider A only when its holder stands in the rider's seat not held, its release complete, tilted no more than 10°, and within 6 mm horizontally and 3 mm vertically of the well floor (4 mm horizontally in the holder stand's seats). Once an element counts as seated on a rider, it keeps counting while its holder stays within 9 mm horizontally and 9 mm vertically of the well floor; the other conditions still apply.
- A seated element sits exactly at rider A's index mark plus its fixed offset: where it stands inside the well, its tilt and its rotation make no difference.
- The light on the screen is computed from the elements seated on the riders only. A holder on the stand, a held element, the arm or anything else in the beam neither blocks nor changes it.
- A rider moves only when pushed: it takes {rail_friction_N} N along the rail to slide one, and a rider released while sliding stops within about 1 mm. Seating or lifting a holder does not move a rider.
- The light on the screen is updated whenever the laser is switched, an element is seated or lifted, or a rider has moved more than 0.1 mm, at most 5 times per simulated second; at every `get_observation` it shows exactly the state of that moment.
- The slit plate, the screen card, the index pointers and lines and the laser aperture have no collision; the holder's foot, post and handle (except its small top cap), the riders with their bumpers, arms, handles and the screen's frame, the rail, the scale strip, the laser module and its button, and the holder stand do.
- A button registers once when its cap reaches the switch depth, and again only after the cap has risen to within 2 mm of rest.
- The central maximum over-exposes the card. A dark fringe counts as showing on the screen when it lies at least 1 mm inside the card's edge and the bright band just beyond it lights the card at least 5 % as strongly as the over-exposed centre.

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing and exporting records are free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.** e_rel = |a − a_true| / a_true. Success when all conditions hold and e_rel ≤ {threshold_pct}%. Reward = G × max(0, 1 − e_rel / {threshold}).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
