"""Command-line interface for the live webcam prototype."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"


def positive_int(value: str) -> int:
    """Return a positive integer or raise an argparse-friendly error."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def confidence(value: str) -> float:
    """Return a confidence threshold in the inclusive range [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the application's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Open a webcam and display MediaPipe landmarks for up to two hands."
        )
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="camera device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=1280,
        help="requested capture width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=720,
        help="requested capture height (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=30,
        help="requested capture frame rate (default: 30)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"hand-landmarker model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=confidence,
        default=0.5,
        metavar="VALUE",
        help="minimum initial hand-detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--min-presence-confidence",
        type=confidence,
        default=0.5,
        metavar="VALUE",
        help="minimum hand-presence confidence (default: 0.5)",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=confidence,
        default=0.5,
        metavar="VALUE",
        help="minimum between-frame tracking confidence (default: 0.5)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not mirror the webcam image",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the prototype, and return a process exit code."""
    args = build_parser().parse_args(argv)

    if args.camera < 0:
        print("Error: --camera must be zero or greater.", file=sys.stderr)
        return 2

    if not args.model.is_file():
        print(
            "Error: the hand-landmarker model is missing.\n"
            "Run `python download_model.py`, then try again.",
            file=sys.stderr,
        )
        return 2

    try:
        from bsl_sign_recognition.webcam import WebcamConfig, run_webcam
    except ImportError as exc:
        print(
            "Error: a required package is not installed.\n"
            "Run `python -m pip install -r requirements.txt`.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1

    config = WebcamConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model.resolve(),
        mirror=not args.no_mirror,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    try:
        run_webcam(config)
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    return 0
