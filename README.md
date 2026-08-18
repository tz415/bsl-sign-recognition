# BSL sign recognition

A Python project for recognising British Sign Language (BSL) 
signs from a webcam and displaying them as text.

## Current milestone

The core webcam and two-hand landmark pipeline has passed an initial hardware
smoke test. It uses MediaPipe Hand Landmarker to display 21 landmarks per hand,
reports the measured processing rate, and can save annotated experiment
snapshots.

The current milestone is a hardware check of the refactored static-pose collector
and handedness calibration. The collector requires two hands to remain stable
across a short quality-check window, but it saves only one landmark pose. It does
not store a motion sequence or video.

This pretrained MediaPipe model is a **landmark extractor**, not the BSL
classifier. A later stage will train and evaluate a separate classifier on
labelled BSL data.

## Quick start

Python 3.12 is recommended.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py
python main.py
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py
python main.py
```

The model downloader uses a versioned official MediaPipe model URL and verifies
the downloaded file with SHA-256 before installing it.

## Using the webcam prototype

The defaults request 1280×720 at 30 FPS, a sensible starting point for a Logitech
C922. Camera drivers can negotiate different values; the program prints the
actual resolution and frame rate reported by the driver.

Controls:

- `Q` or `Esc`: quit
- `S`: save an annotated snapshot under `captures/`

Useful options:

```bash
# Use a second connected camera
python main.py --camera 1

# Test the C922's 1080p mode
python main.py --width 1920 --height 1080 --fps 30

# Test the C922's higher-frame-rate 720p mode
python main.py --width 1280 --height 720 --fps 60

# Process the unmirrored camera image
python main.py --no-mirror

# List every option
python main.py --help
```

If the camera cannot be opened, close Logitech Capture, Discord, browser video
calls, or any other program that may be using it. Then try `--camera 1` if the
computer also has a built-in camera.

## Pipeline

```text
webcam frame
    → colour conversion
    → MediaPipe two-hand landmark tracking
    → 21 (x, y, z) landmarks per detected hand
    → on-screen skeleton and diagnostics
```

The next data milestone collects one static two-hand landmark pose per sample,
starting with five BSL fingerspelling letters. Each capture also stores a local
hand-region audit image so labels can be checked before training. Raw participant
data stays out of Git. Motion sequences are a later, separate milestone.

First measure handedness behaviour with a normal open palm:

```bash
python calibrate_handedness.py
```

The program guides you through 100 accepted frames of your physical right hand,
then 100 of your physical left hand. It saves only aggregate prediction counts
and confidence values under `data/calibration/`; it does not save camera frames.
This determines whether the previous wrong-hand result was gesture-specific,
consistently reversed, or inconsistent.

Run the static collector with:

```bash
python collect_data.py --participant p001
```

Use `1`–`5` to select `A`, `B`, `C`, `L`, or `V`, hold the pose, and press
`Space`. Use the [UCL BSL SignBank two-handed fingerspelling
reference](https://bslsignbank.ucl.ac.uk/spell/twohanded.html) rather than guessing
the handshape. A capture is saved only after exactly two hands remain below the
movement threshold for 12 consecutive frames. The middle frame becomes the one
static sample; surrounding frames are discarded.

Controls:

- `1`–`5`: select `A`, `B`, `C`, `L`, or `V`
- `Space`: start a capture
- `R`: move the most recent bad capture from `data/raw/` to `data/rejected/`
- `Q` or `Esc`: quit

Every JSON sample contains `sample_type: "static_pose"`, two sets of 21 image and
world landmarks, capture metadata, and the measured stability score. It contains
no `frames` array or recording duration. MediaPipe handedness remains explicitly
marked as unverified metadata until the calibration is reviewed.

For the first hardware check, save only one correct `A` sample. Confirm that its
JSON and cropped audit image agree before collecting a pilot dataset. Audit images
can still contain identifiable imagery if hands are held near a face, so review
them locally and never commit raw participant data.

## Planned stages

- [x] Verify core webcam capture and live hand-landmark detection
- [x] Select five static two-handed fingerspelling classes for the pilot
- [ ] Hardware-test handedness calibration and the refactored static collector
- [ ] Collect and evaluate labelled static poses with consent
- [ ] Train a baseline classifier
- [ ] Add landmark sequences and temporal features for moving signs
- [ ] Display stable real-time predictions
- [ ] Evaluate on unseen signers and document limitations

## Reproducibility

Run the dependency-free tests with:

```bash
python -m unittest discover -s tests -v
```

Hardware observations and experiments are recorded in
[`docs/experiment-log.md`](docs/experiment-log.md). Recording failed attempts and
measured trade-offs is part of the project, not something to hide.

## Scope and limitations

The initial system will recognise only a deliberately small vocabulary of isolated
BSL signs. It is not a complete BSL translator, is not intended to replace
interpreters, and should not be relied on for important communication. Performance
must be tested across signers, lighting, backgrounds, camera positions, and signing
styles before any claims are made.