"""Run an interactive MediaPipe handedness calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from bsl_sign_recognition.cli import DEFAULT_MODEL_PATH, positive_int
from bsl_sign_recognition.handedness import (
    HandednessAccumulator,
    build_calibration_report,
)
from bsl_sign_recognition.static_collector import draw_result

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "calibration"
WINDOW_NAME = "MediaPipe handedness calibration"


@dataclass(frozen=True)
class CalibrationConfig:
    """Runtime settings for handedness calibration."""

    camera_index: int
    width: int
    height: int
    fps: int
    model_path: Path
    output_directory: Path
    frames_per_hand: int
    mirror: bool


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for calibration."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure MediaPipe handedness predictions for known physical hands."
        )
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--frames",
        type=positive_int,
        default=100,
        help="accepted single-hand frames per physical hand (default: 100)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not mirror the input before landmark detection",
    )
    return parser


def _open_camera(config: CalibrationConfig) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"could not open camera {config.camera_index}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    camera.set(cv2.CAP_PROP_FPS, config.fps)
    return camera


def _create_landmarker(
    config: CalibrationConfig,
) -> mp.tasks.vision.HandLandmarker:
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _draw_status(
    frame: cv2.typing.MatLike,
    *,
    accumulator: HandednessAccumulator | None,
    collecting: bool,
    latest_prediction: str,
) -> None:
    if accumulator is None:
        lines = ("CALIBRATION COMPLETE", "The report will be saved locally.")
    else:
        state = "COLLECTING" if collecting else "PRESS SPACE TO START"
        lines = (
            f"Show PHYSICAL {accumulator.physical_hand.upper()} hand only",
            "Use an open palm facing the camera",
            state,
            (
                f"Accepted frames: {accumulator.observed_frames}/"
                f"{accumulator.target_frames}"
            ),
            f"Latest model output: {latest_prediction}",
            "Space: start | Q/Esc: cancel",
        )

    cv2.rectangle(frame, (8, 8), (590, 174), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 32 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def run_calibration(config: CalibrationConfig) -> dict[str, Any] | None:
    """Collect right- and left-hand predictions, or return None if cancelled."""
    camera = _open_camera(config)
    accumulators = [
        HandednessAccumulator("Right", config.frames_per_hand),
        HandednessAccumulator("Left", config.frames_per_hand),
    ]
    stage_index = 0
    collecting = False
    latest_prediction = "none"
    video_started_ns = time.monotonic_ns()
    consecutive_read_failures = 0
    actual_width = config.width
    actual_height = config.height
    completed = False

    print("Handedness calibration uses one open physical hand at a time.")
    print("Press Space to start each stage. Press Q or Esc to cancel.")

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
                actual_height, actual_width = frame.shape[:2]

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

                accumulator = (
                    accumulators[stage_index]
                    if stage_index < len(accumulators)
                    else None
                )
                if collecting and accumulator is not None:
                    if len(result.hand_landmarks) == 1:
                        handedness = result.handedness[0][0]
                        latest_prediction = (
                            f"{handedness.category_name} "
                            f"{handedness.score:.1%}"
                        )
                        accumulator.add(
                            handedness.category_name,
                            float(handedness.score),
                        )
                        if accumulator.complete:
                            summary = accumulator.summary()
                            print(
                                f"{accumulator.physical_hand}: "
                                f"{summary['model_prediction_counts']}"
                            )
                            collecting = False
                            stage_index += 1
                            latest_prediction = "none"
                            if stage_index == len(accumulators):
                                completed = True

                display_frame = frame.copy()
                draw_result(display_frame, result)
                current_accumulator = (
                    accumulators[stage_index]
                    if stage_index < len(accumulators)
                    else None
                )
                _draw_status(
                    display_frame,
                    accumulator=current_accumulator,
                    collecting=collecting,
                    latest_prediction=latest_prediction,
                )
                cv2.imshow(WINDOW_NAME, display_frame)

                if completed:
                    cv2.waitKey(250)
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    return None
                if key == ord(" ") and not collecting:
                    collecting = True
                    latest_prediction = "waiting for one hand"

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    return None
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return build_calibration_report(
        accumulators,
        mirrored_before_detection=config.mirror,
        camera_width=actual_width,
        camera_height=actual_height,
    )


def save_report(report: dict[str, Any], output_directory: Path) -> Path:
    """Save one calibration report and return its path."""
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    destination = output_directory / f"handedness-{timestamp}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments, run calibration, and return an exit code."""
    args = build_parser().parse_args(argv)
    if args.camera < 0:
        print("Error: --camera must be zero or greater.", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(
            "Error: hand-landmarker model missing. Run `python download_model.py`.",
            file=sys.stderr,
        )
        return 2

    config = CalibrationConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model.resolve(),
        output_directory=args.output_directory.resolve(),
        frames_per_hand=args.frames,
        mirror=not args.no_mirror,
    )

    try:
        report = run_calibration(config)
        if report is None:
            print("Calibration cancelled; no report was saved.")
            return 0
        destination = save_report(report, config.output_directory)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130

    interpretation = report["interpretation"]
    messages = {
        "labels_match_physical_hands": (
            "MediaPipe labels matched both physical hands consistently."
        ),
        "labels_are_reversed": (
            "MediaPipe labels were consistently reversed in this setup."
        ),
        "labels_are_inconsistent": (
            "MediaPipe labels were inconsistent; do not treat them as truth."
        ),
    }
    print(f"Interpretation: {messages[interpretation]}")
    print(f"Calibration report saved: {destination}")
    return 0
