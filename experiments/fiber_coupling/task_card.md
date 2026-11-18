# Optics: couple a laser into a single-mode fiber

**Goal.** A helium-neon laser's beam is steered by two mirrors into a fiber coupler, where a lens focuses it onto the end of a single-mode fiber; the fiber's far end goes to a power meter. Make the power coupled into the fiber as high as this bench allows: choose the coupling lens, align the beam and focus the fiber. Report the meter's reading of your final state.

**Apparatus.**
- **Laser.** A helium-neon laser, {laser_nm} nm, {laser_power_mw} mW (its label), on a post; nominal beam waist radius {laser_waist_radius_mm} mm (1/e² intensity). Its power supply has a `POWER` push button (travel {buttons_travel_mm} mm, switch at {buttons_switch_mm} mm) and a pilot lamp; each press switches the laser on or off. It starts off.
- **Mirrors.** Two mirrors, each in a kinematic mount fixed to the bench. Each mount has two knurled adjuster knobs: one tilts the mirror about a vertical axis, the other about a horizontal one. Each adjuster is a fine screw, {mirror_pitch_mm_per_turn} mm of travel per turn, acting {mirror_lever_mm} mm from the mount's pivot. The knobs turn on their own axes and move nothing else.
- **Fiber coupler.** Fixed to the bench. The beam enters it through a seat for a lens cartridge, whose lens focuses it onto the fiber's end; the fiber's connector sits on a focus stage, whose knob moves the fiber's end along the beam by {focus_mm_per_turn} mm per turn. A black alignment cap with a white crosshair and concentric circles stands in a slot on the coupler; while it is there no light enters the fiber, and the beam shows as a spot on its face. It starts on.
- **Alignment target.** A white plate with a crosshair and a {target_hole_mm} mm hole at its centre, on a stem with a handle. A slot on the bench holds it across the beam; there the beam shows as a spot on its face, and the part of the beam that falls into the hole goes on.
- **Lens box.** Five lens cartridges in slots, focal lengths {lenses_text} mm, each printed on the cartridge's top and engraved on one face; the box also has places for the cap and the target. The target starts in the box. A cartridge is a block with a white handle holding its lens in a ring; it stands in a slot of the box or in the coupler's seat either way round. Each cartridge is made to put its lens's focus on the fiber's end (within a fraction of a millimetre) when it stands in the coupler with its engraved face toward the incoming beam.
- **Fiber.** Single-mode fiber for {laser_nm} nm, nominal mode field diameter {fiber_mfd_um} µm, cladding diameter {fiber_cladding_um} µm, a short yellow patch cord to the meter.
- **Power meter.** Shows the optical power leaving the fiber's far end in µW (below 1 µW in nW), {meter_digits} significant digits, refreshed every {meter_refresh_s} s: the light guided in the fiber's core and, with this short patch cord, some light carried in the cladding while the focused spot lies on it.

**Instrument.**
- **Observations.** The meter has no data connection: read it in the camera images. Every `get_observation` is evidence of the state it shows: it returns `evidence_id` (`<episode_id>:obs:<n>`).

**Acceptance conditions** (the evaluator checks these; how you meet them is up to you):
- `power_evidence` cites an observation of this episode, and the light path has not changed since: the laser, the cartridge in the coupler and its orientation, the cap and the target stand as they did then, and no adjuster screw (mirrors, focus) has moved by more than {screw_tol_deg}° of knob rotation.
- `lens` is the focal length printed on the cartridge in the coupler (mm). When you submit, the cap is not on the coupler and the laser is on.
- Leave the bench in order: when you submit, no loose item is tipped over or off the bench (more than 0.10 m below the table top), and the robot holds nothing.
- Tidy up: when you submit, every cartridge not in the coupler stands in a slot of the lens box, and the cap is in the box or set down elsewhere off the coupler.
- Submit exactly once. The first `submit_result` closes the episode, even if it is invalid.

**Simulation settings** (how this simulation decides; the same at every exposure level):
- A cartridge, the cap or the target stands in a seat or slot when it is not held, upright, and down on the seat's floor; only then does it act on the beam as such. Anything else in the beam (the robot, an item set down in it) blocks it.
- After you submit, the evaluator computes, from the simulation's optics, the light path as it stands: η, the fraction of the laser's power coupled into the fiber's core (the target, if still in the beam, passes only what falls into its hole), and P, the power the meter receives (µW).

**Budget.** {sim_s} simulated seconds; every physics tick is charged (observing is free). A command that needs more simulated time than remains is refused and nothing moves; once no time remains, a further physical action ends the episode and nothing is accepted after that, not even a submission, so submit before the budget runs out. The wall-clock limit is in README.md. One submission.

**Scoring.**
- η* is a reference value: the highest core coupling that an offline calculation for this bench found (the best of the five lenses, standing the right way round, with the beam and the fiber's focus at their best); an η larger than η* counts as η*.
- e = max(max(0, 1 − η / η*) / {threshold}, |power_uW − P| / ({reported_threshold} · P)).
- Success when all conditions hold and e ≤ 1: η is at least (1 − {threshold}) × η*, and the reported power is within {reported_threshold_pct}% of P.
- Reward = G × max(0, 1 − e).

**Submission** (`submit_result` arguments; JSON schema in `submission_schema`; the `<…>` entries are placeholders, replace them with your own values):
```json
{submission_example}
```
