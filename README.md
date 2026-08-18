# BSL sign recognition

A Python project exploring recognition of British Sign Language (BSL) signs from a webcam.

I'm currently working with MediaPipe to track hand landmarks and collect labelled data, with the aim of training a classifier to recognise a small set of signs.

## Current progress

The webcam pipeline can currently:
- detect and track two hands
- extract 21 landmarks from each hand
- display the landmarks in real time
- show the processing frame rate
- save annotated snapshots

I've also started building the data collection pipeline. The first dataset will use five static two-handed fingerspelling letters: `A`, `B`, `C`, `L`, and `V`.

Before collecting the dataset, I'm testing MediaPipe's handedness predictions and checking that the collector saves poses consistently.

## Setup

Python 3.12 is recommended.

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py
python main.py
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py
python main.py
```

## Webcam

Run the webcam prototype with:

```bash
python main.py
```

The default camera settings are 1280×720 at 30 FPS. These can be changed from the command line:

```bash
python main.py --width 1920 --height 1080 --fps 30
```

Use another connected camera with:

```bash
python main.py --camera 1
```

Other options can be found with:

```bash
python main.py --help
```

### Controls

- `Q` or `Esc` — quit
- `S` — save an annotated snapshot to `captures/`

## Hand tracking pipeline

```text
webcam frame
    → MediaPipe Hand Landmarker
    → 21 (x, y, z) landmarks per hand
    → landmark data / on-screen skeleton
```

## Handedness calibration

Before collecting data, I use a short calibration test to check how MediaPipe labels my left and right hands:

```bash
python calibrate_handedness.py
```

The test records MediaPipe's predictions for each hand and saves the results to `data/calibration/`.

## Collecting samples

Start the static-pose collector with:

```bash
python collect_data.py --participant p001
```

The current pilot uses the letters `A`, `B`, `C`, `L`, and `V`.

I use the [UCL BSL SignBank two-handed fingerspelling reference](https://bslsignbank.ucl.ac.uk/spell/twohanded.html) when checking the handshapes.

### Controls

- `1`–`5` — select `A`, `B`, `C`, `L`, or `V`
- `Space` — capture a sample
- `R` — reject the most recent capture
- `Q` or `Esc` — quit

A sample is accepted once both hands have remained sufficiently stable for several frames. The collector then saves the landmarks from a single frame along with information about the capture.

For now I'm only collecting static poses. Motion-based signs will require sequences of landmarks and will be explored later. Data that I collected personally and audit images are kept out of the Git repository.

## Project plan

- [x] Set up webcam capture
- [x] Add live hand-landmark detection
- [x] Choose an initial set of static fingerspelling signs
- [ ] Test handedness calibration and data collection
- [ ] Collect a pilot dataset
- [ ] Train and evaluate a baseline classifier
- [ ] Investigate moving signs using landmark sequences
- [ ] Add stable real-time predictions
- [ ] Test performance on unseen signers

## Tests

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

Notes from hardware tests and experiments are kept in [docs/experiment-log.md](docs/experiment-log.md).

## Limitations

This is an experimental project using a small set of isolated BSL signs rather than a complete BSL translation system.

One of the main things I want to explore is how far can a model trained on landmark data works on different people and different recording conditions.