# Experiment log

Use this log to record decisions, measurements, failures, and improvements. Results
belong here only after they have actually been observed.

## Experiment 001 — Initial hand-landmark smoke test

- **Status:** Partially complete
- **Question:** Does the webcam pipeline detect zero, one, and two hands and
  produce structurally valid landmark data?
- **Independent variables:** Lighting, distance from camera, background, one hand
  versus two hands, static pose versus movement.
- **Measurements to record:** Negotiated camera resolution, reported camera FPS,
  processing FPS shown in the overlay, missed hands, visibly unstable landmarks.
- **Procedure:**
  1. Run the default 1280×720 configuration.
  2. Hold one hand still for 10 seconds, then move it slowly for 10 seconds.
  3. Repeat with two hands.
  4. Repeat once with poorer lighting or a busier background.
  5. Save one representative snapshot from each condition with the `S` key.
- **Results:** The live prototype reported 0, 1, and 2 detected hands correctly in
  an initial user test. An earlier throwaway sequence sample contained 39 frames
  over 1.509 seconds, had 100% hand detection, increasing timestamps, and 21 image
  plus 21 world landmarks per hand. That deliberately mislabelled sample was not
  valid training data. MediaPipe predicted `Left` with mean 97.6% confidence while
  the user reported using their physical right hand in an unusual gesture. An
  earlier resolution experiment measured:

  | Resolution | Measured FPS |
  |---|---:|
  | 640 × 480 | 21.46 |
  | 1280 × 720 | 15.88 |
  | 1920 × 1080 | 14.39 |

  All requested resolutions were delivered. These exploratory measurements
  included camera startup and per-frame terminal printing, so they should be
  repeated with the current rolling FPS counter before drawing a firm conclusion.
- **Conclusion and next change:** Core landmark extraction works. Camera-mode
  measurements currently favour 640 × 480, but need a controlled repeat.
  Handedness also needs an open-palm calibration before it is trusted.

## Experiment 002 — Handedness calibration

- **Status:** Ready for hardware test
- **Question:** With mirrored input, do MediaPipe handedness predictions match the
  user's physical hand, reverse consistently, or vary by pose?
- **Controlled pose:** One open palm facing the camera; no second hand visible.
- **Measurements:** 100 accepted frames for the physical right hand and 100 for
  the physical left hand, prediction counts, mean confidence, and agreement rate.
- **Procedure:**
  1. Run `python calibrate_handedness.py`.
  2. Show only the requested physical hand and press `Space` for each stage.
  3. Copy the aggregate report values here.
  4. If results are inconsistent, repeat in different lighting before changing
     any handedness logic.
- **Results:** Pending.
- **Conclusion and next change:** Pending.

## Experiment 003 — Static collector acceptance test

- **Status:** Ready for hardware test
- **Question:** Does the collector reject movement and save exactly one valid,
  correctly labelled static pose?
- **Procedure:**
  1. Run `python collect_data.py --participant p001`.
  2. Use the UCL BSL SignBank reference to hold letter `A` correctly.
  3. Move during one attempted capture and verify that no sample is saved until
     the pose becomes stable.
  4. Save one correct sample and inspect its JSON and audit image.
  5. Verify `sample_type` is `static_pose`, `hand_count` is 2, the quality score
     is at or below its threshold, and there is no `frames` field.
  6. Test `R` once and confirm the pair moves from `data/raw/` to
     `data/rejected/`.
- **Results:** Pending.
- **Conclusion and next change:** Pending.
