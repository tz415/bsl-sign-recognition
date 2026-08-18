"""Download and verify the pinned MediaPipe hand-landmarker model."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> str:
    """Copy a binary stream while calculating its SHA-256 digest."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
        destination.write(chunk)
        digest.update(chunk)
    return digest.hexdigest()


def download_model(
    destination: Path = DEFAULT_DESTINATION, force: bool = False
) -> Path:
    """Download the model atomically, verifying its pinned checksum."""
    if destination.exists() and not force:
        if sha256_file(destination) == MODEL_SHA256:
            print(f"Model is already present and verified: {destination}")
            return destination
        raise RuntimeError(
            f"{destination} exists but has the wrong checksum. "
            "Re-run with --force to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    try:
        print(f"Downloading model from {MODEL_URL}")
        with urlopen(MODEL_URL, timeout=60) as source, partial.open("wb") as output:
            downloaded_digest = _copy_and_hash(source, output)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"model download failed: {exc}") from exc

    if downloaded_digest != MODEL_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "downloaded model failed checksum verification; the file was not installed."
        )

    partial.replace(destination)
    print(f"Model downloaded and verified: {destination}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build the model-downloader argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and verify the MediaPipe hand-landmarker model."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"model destination (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing model file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Download the model and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        download_model(args.destination.resolve(), force=args.force)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
