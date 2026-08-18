"""Collect labelled, single-frame BSL fingerspelling landmark samples."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, deque
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from bsl_sign_recognition.cli import DEFAULT_MODEL_PATH, positive_int
from bsl_sign_recognition.static_samples import (
    CollectorConfig,
    Landmark,
    build_sample,
    max_pairwise_displacement,
    serialize_hands,
)
from bsl_sign_recognition.webcam import HAND_CONNECTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
WINDOW_NAME = "Static BSL data collector"

# These are two-handed BSL fingerspelling letters, chosen as a controlled
# static-pose milestone. They are not English word signs.
STATIC_LABELS = {
    ord("1"): "a",
    ord("2"): "b",
    ord("3"): "c",
    ord("4"): "l",
    ord("5"): "v",
}


def _safe_identifier(value: str) -> str:
    """Validate an identifier before using it as part of a directory path."""
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_"
    )
    if not value or any(character not in allowed for character in value):
        raise argparse.ArgumentTypeError(
            "use only letters, numbers, hyphens, and underscores"
        )
    return value


def _positive_float(value: str) -> float:
    """Return a positive floating-point value for argparse."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the collector."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect one static two-hand BSL fingerspelling pose per capture."
        )
    )
    parser.add_argument(
        "--participant",
        type=_safe_identifier,
        default="p001",
        help="anonymous participant identifier (default: p001)",
    )
    parser.add_argument(
        "--session",
        type=_safe_identifier,
        default=None,
        help="session identifier (default: generated from the current time)",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
    )
    parser.add_argument(
        "--countdown",
        type=_positive_float,
        default=2.0,
        help="seconds between pressing Space and capture (default: 2)",
    )
    parser.add_argument(
        "--stability-frames",
        type=positive_int,
        default=12,
        help="consecutive stable frames required before saving (default: 12)",
    )
    parser.add_argument(
        "--max-mean-displacement",
        type=_positive_float,
        default=0.015,
        metavar="VALUE",
        help=(
            "maximum mean landmark movement between stable frames "
            "(default: 0.015)"
        ),
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not mirror the input before landmark detection",
    )
    return parser


def _open_camera(config: CollectorConfig) -> cv2.VideoCapture:
    """Open and configure the requested camera."""
    camera = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"could not open camera {config.camera_index}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    camera.set(cv2.CAP_PROP_FPS, config.fps)
    return camera


def _create_landmarker(config: CollectorConfig) -> mp.tasks.vision.HandLandmarker:
    """Create the continuously tracked two-hand landmarker."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def save_sample(
    *,
    label: str,
    hands: list[dict[str, Any]],
    frame: cv2.typing.MatLike,
    config: CollectorConfig,
    stability_displacement: float,
) -> tuple[Path, Path]:
    """Save one landmark JSON file and one local audit image."""
    captured_at = datetime.now(timezone.utc)
    height, width = frame.shape[:2]
    sample = build_sample(
        label=label,
        hands=hands,
        config=config,
        captured_at=captured_at,
        width=width,
        height=height,
        stability_displacement=stability_displacement,
    )

    folder = (
        config.data_directory
        / config.participant_id
        / config.session_id
        / label
    )
    folder.mkdir(parents=True, exist_ok=True)

    stem = captured_at.strftime("%Y%m%dT%H%M%S-%fZ")
    json_path = folder / f"{stem}.json"
    image_path = folder / f"{stem}.jpg"
    sample["audit_image"] = image_path.name

    audit_image = _crop_hand_region(frame, hands)
    if not cv2.imwrite(str(image_path), audit_image):
        raise RuntimeError(f"could not save audit image to {image_path}")

    try:
        json_path.write_text(
            json.dumps(sample, indent=2),
            encoding="utf-8",
        )
    except OSError:
        image_path.unlink(missing_ok=True)
        raise

    return json_path, image_path


def _crop_hand_region(
    frame: cv2.typing.MatLike,
    hands: list[dict[str, Any]],
    padding: int = 36,
) -> cv2.typing.MatLike:
    """Crop an audit image around both hands to reduce unnecessary imagery."""
    height, width = frame.shape[:2]
    x_values = [
        min(max(float(point[0]), 0.0), 1.0)
        for hand in hands
        for point in hand["image_landmarks"]
    ]
    y_values = [
        min(max(float(point[1]), 0.0), 1.0)
        for hand in hands
        for point in hand["image_landmarks"]
    ]
    left = max(int(min(x_values) * width) - padding, 0)
    right = min(int(max(x_values) * width) + padding, width)
    top = max(int(min(y_values) * height) - padding, 0)
    bottom = min(int(max(y_values) * height) + padding, height)
    if right <= left or bottom <= top:
        raise ValueError("could not calculate a valid hand-region crop")
    return frame[top:bottom, left:right]


def reject_sample(
    *,
    json_path: Path,
    image_path: Path,
    label: str,
    config: CollectorConfig,
) -> tuple[Path, Path]:
    """Move the latest bad capture out of the raw training-data directory."""
    destination = (
        config.data_directory.parent
        / "rejected"
        / config.participant_id
        / config.session_id
        / label
    )
    destination.mkdir(parents=True, exist_ok=True)
    rejected_json = destination / json_path.name
    rejected_image = destination / image_path.name

    json_path.replace(rejected_json)
    try:
        image_path.replace(rejected_image)
    except OSError:
        rejected_json.replace(json_path)
        raise
    return rejected_json, rejected_image


def _landmark_to_pixel(
    landmark: Landmark,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert one normalised landmark into a clamped pixel position."""
    x = min(max(float(landmark.x), 0.0), 1.0)
    y = min(max(float(landmark.y), 0.0), 1.0)
    return min(int(x * width), width - 1), min(int(y * height), height - 1)


def draw_result(frame: cv2.typing.MatLike, result: Any) -> None:
    """Draw all detected hand landmarks onto a display frame."""
    height, width = frame.shape[:2]
    for landmarks in result.hand_landmarks:
        points = [
            _landmark_to_pixel(point, width, height)
            for point in landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (80, 220, 120),
                2,
                cv2.LINE_AA,
            )
        for point in points:
            cv2.circle(frame, point, 4, (30, 80, 255), -1, cv2.LINE_AA)


def _draw_status(
    frame: cv2.typing.MatLike,
    *,
    selected_label: str,
    detected_hands: int,
    phase: str,
    remaining_seconds: int | None,
    stability_progress: int,
    stability_target: int,
    latest_displacement: float | None,
    saved_counts: Counter[str],
) -> None:
    """Draw collector instructions and state."""
    if phase == "idle":
        state = "READY - hold pose, then press Space"
    elif phase == "countdown":
        state = f"HOLD STILL - capturing in {remaining_seconds}"
    elif latest_displacement is None:
        state = "CHECKING STABILITY - keep both hands visible"
    elif latest_displacement > 0:
        state = f"CHECKING STABILITY - movement {latest_displacement:.4f}"
    else:
        state = "CHECKING STABILITY"

    counts = " ".join(
        f"{label.upper()}:{saved_counts[label]}"
        for label in STATIC_LABELS.values()
    )

    lines = (
        f"Selected letter: {selected_label.upper()}",
        state,
        f"Hands detected: {detected_hands}/2",
        f"Stable frames: {stability_progress}/{stability_target}",
        f"Saved: {counts}",
        "1 A | 2 B | 3 C | 4 L | 5 V",
        "Space: capture | R: reject last | Q/Esc: quit",
    )

    cv2.rectangle(frame, (8, 8), (625, 204), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        colour = (120, 255, 120) if detected_hands == 2 else (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (18, 32 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            1,
            cv2.LINE_AA,
        )


def run_collector(config: CollectorConfig) -> None:
    """Run the live collector until the user quits."""
    camera = _open_camera(config)
    selected_label = "a"
    phase = "idle"
    countdown_started: float | None = None
    candidates: deque[tuple[list[dict[str, Any]], cv2.typing.MatLike]] = deque(
        maxlen=config.stability_frames
    )
    latest_displacement: float | None = None
    saved_counts: Counter[str] = Counter()
    last_saved: tuple[Path, Path, str] | None = None
    video_started_ns = time.monotonic_ns()
    consecutive_read_failures = 0

    print("Static collector: 1-5 select a letter, Space captures, Q quits.")
    print(
        f"Participant={config.participant_id} Session={config.session_id}"
    )
    print("MediaPipe handedness is stored as unverified metadata for now.")

    try:
        with _create_landmarker(config) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 10:
                        raise RuntimeError(
                            "the camera repeatedly failed to return frames"
                        )
                    continue
                consecutive_read_failures = 0

                if config.mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_pipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                timestamp_ms = (
                    time.monotonic_ns() - video_started_ns
                ) // 1_000_000
                result = landmarker.detect_for_video(
                    media_pipe_image,
                    timestamp_ms,
                )

                now = time.perf_counter()
                remaining_seconds: int | None = None
                if phase == "countdown" and countdown_started is not None:
                    elapsed = now - countdown_started
                    remaining_seconds = max(
                        math.ceil(config.countdown_seconds - elapsed),
                        0,
                    )

                    if elapsed >= config.countdown_seconds:
                        phase = "stabilising"
                        countdown_started = None
                        remaining_seconds = None
                        candidates.clear()
                        latest_displacement = None

                if phase == "stabilising":
                    if len(result.hand_landmarks) != 2:
                        candidates.clear()
                        latest_displacement = None
                    else:
                        hands = serialize_hands(result)
                        candidates.append((hands, frame.copy()))
                        if len(candidates) >= 2:
                            latest_displacement = max_pairwise_displacement(
                                [candidate[0] for candidate in candidates]
                            )

                        if (
                            len(candidates) == config.stability_frames
                            and latest_displacement is not None
                            and latest_displacement
                            <= config.max_mean_displacement
                        ):
                            candidate_hands, candidate_frame = list(candidates)[
                                len(candidates) // 2
                            ]
                            json_path, image_path = save_sample(
                                label=selected_label,
                                hands=candidate_hands,
                                frame=candidate_frame,
                                config=config,
                                stability_displacement=latest_displacement,
                            )
                            saved_counts[selected_label] += 1
                            last_saved = (
                                json_path,
                                image_path,
                                selected_label,
                            )
                            print(
                                f"Saved static {selected_label.upper()} sample: "
                                f"{json_path}"
                            )
                            print(f"Audit image: {image_path}")
                            phase = "idle"
                            candidates.clear()
                            latest_displacement = None

                display_frame = frame.copy()
                draw_result(display_frame, result)
                _draw_status(
                    display_frame,
                    selected_label=selected_label,
                    detected_hands=len(result.hand_landmarks),
                    phase=phase,
                    remaining_seconds=remaining_seconds,
                    stability_progress=len(candidates),
                    stability_target=config.stability_frames,
                    latest_displacement=latest_displacement,
                    saved_counts=saved_counts,
                )
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if phase == "idle" and key in STATIC_LABELS:
                    selected_label = STATIC_LABELS[key]
                if phase == "idle" and key == ord(" "):
                    phase = "countdown"
                    countdown_started = time.perf_counter()
                if phase == "idle" and key in (ord("r"), ord("R")):
                    if last_saved is None:
                        print("No capture from this run is available to reject.")
                    else:
                        json_path, image_path, label = last_saved
                        rejected_json, rejected_image = reject_sample(
                            json_path=json_path,
                            image_path=image_path,
                            label=label,
                            config=config,
                        )
                        saved_counts[label] -= 1
                        last_saved = None
                        print(f"Rejected sample moved to: {rejected_json}")
                        print(f"Rejected audit image: {rejected_image}")

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments, run the collector, and return an exit code."""
    args = build_parser().parse_args(argv)
    if args.camera < 0:
        print("Error: --camera must be zero or greater.", file=sys.stderr)
        return 2
    if args.stability_frames < 2:
        print("Error: --stability-frames must be at least 2.", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(
            "Error: hand-landmarker model missing. Run `python download_model.py`.",
            file=sys.stderr,
        )
        return 2

    session_id = args.session or datetime.now().astimezone().strftime(
        "s%Y%m%d-%H%M%S"
    )
    config = CollectorConfig(
        participant_id=args.participant,
        session_id=session_id,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model.resolve(),
        data_directory=args.data_directory.resolve(),
        countdown_seconds=args.countdown,
        stability_frames=args.stability_frames,
        max_mean_displacement=args.max_mean_displacement,
        mirror=not args.no_mirror,
    )

    try:
        run_collector(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    return 0
