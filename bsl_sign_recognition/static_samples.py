"""Pure data helpers for versioned static landmark samples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


class Landmark(Protocol):
    """Coordinate fields used from one MediaPipe-like landmark."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CollectorConfig:
    """Runtime settings for collecting static landmark samples."""

    participant_id: str
    session_id: str
    camera_index: int
    width: int
    height: int
    fps: int
    model_path: Path
    data_directory: Path
    countdown_seconds: float
    stability_frames: int
    max_mean_displacement: float
    mirror: bool


def landmarks_to_list(landmarks: Sequence[Landmark]) -> list[list[float]]:
    """Convert landmarks into JSON-serialisable xyz triples."""
    return [
        [float(point.x), float(point.y), float(point.z)]
        for point in landmarks
    ]


def serialize_hands(result: Any) -> list[dict[str, Any]]:
    """Serialise one result and give its hands a deterministic screen order."""
    hands: list[dict[str, Any]] = []

    for index, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[index][0]
        hands.append(
            {
                "wrist_x": float(landmarks[0].x),
                "model_handedness": handedness.category_name,
                "model_handedness_score": float(handedness.score),
                "image_landmarks": landmarks_to_list(landmarks),
                "world_landmarks": landmarks_to_list(
                    result.hand_world_landmarks[index]
                ),
            }
        )

    hands.sort(key=lambda hand: hand["wrist_x"])
    for index, hand in enumerate(hands):
        hand["screen_order"] = index
        del hand["wrist_x"]

    return hands


def validate_hands(hands: list[dict[str, Any]]) -> None:
    """Validate one serialised two-hand pose."""
    if len(hands) != 2:
        raise ValueError("a static BSL fingerspelling sample requires two hands")

    screen_orders: set[int] = set()
    for hand in hands:
        screen_order = hand.get("screen_order")
        if screen_order not in (0, 1):
            raise ValueError("screen_order must be 0 or 1")
        screen_orders.add(screen_order)

        score = hand.get("model_handedness_score")
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("model handedness score must be between 0 and 1")

        for coordinate_set in ("image_landmarks", "world_landmarks"):
            landmarks = hand.get(coordinate_set)
            if not isinstance(landmarks, list) or len(landmarks) != 21:
                raise ValueError(f"each hand requires 21 {coordinate_set}")
            if any(
                not isinstance(point, list)
                or len(point) != 3
                or any(not math.isfinite(float(value)) for value in point)
                for point in landmarks
            ):
                raise ValueError(
                    f"each {coordinate_set} point requires three finite values"
                )

    if screen_orders != {0, 1}:
        raise ValueError("the two hands require distinct screen_order values")


def mean_image_displacement(
    previous_hands: list[dict[str, Any]],
    current_hands: list[dict[str, Any]],
) -> float:
    """Return mean two-dimensional landmark movement between two poses."""
    validate_hands(previous_hands)
    validate_hands(current_hands)

    previous_hands = sorted(previous_hands, key=lambda hand: hand["screen_order"])
    current_hands = sorted(current_hands, key=lambda hand: hand["screen_order"])
    total = 0.0
    count = 0
    for previous_hand, current_hand in zip(previous_hands, current_hands):
        for previous, current in zip(
            previous_hand["image_landmarks"],
            current_hand["image_landmarks"],
        ):
            total += math.hypot(
                float(current[0]) - float(previous[0]),
                float(current[1]) - float(previous[1]),
            )
            count += 1
    return total / count


def max_pairwise_displacement(
    poses: Sequence[list[dict[str, Any]]],
) -> float:
    """Return the largest mean displacement between any two poses."""
    if len(poses) < 2:
        raise ValueError("at least two poses are required")
    return max(
        mean_image_displacement(previous, current)
        for previous_index, previous in enumerate(poses[:-1])
        for current in poses[previous_index + 1 :]
    )


def build_sample(
    *,
    label: str,
    hands: list[dict[str, Any]],
    config: CollectorConfig,
    captured_at: datetime,
    width: int,
    height: int,
    stability_displacement: float,
) -> dict[str, Any]:
    """Build a versioned single-pose sample ready to save as JSON."""
    if not label:
        raise ValueError("label cannot be empty")
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    if (
        not math.isfinite(stability_displacement)
        or stability_displacement < 0
    ):
        raise ValueError("stability displacement must be finite and non-negative")

    validate_hands(hands)

    return {
        "schema_version": 2,
        "sample_type": "static_pose",
        "label_kind": "bsl_fingerspelling_letter",
        "label": label,
        "participant_id": config.participant_id,
        "session_id": config.session_id,
        "captured_at_utc": captured_at.isoformat(),
        "hand_count": len(hands),
        "handedness_status": "mediapipe_prediction_unverified",
        "quality": {
            "stability_frames": config.stability_frames,
            "max_pairwise_mean_image_displacement": stability_displacement,
            "accepted_displacement_threshold": config.max_mean_displacement,
        },
        "camera": {
            "width": width,
            "height": height,
            "mirrored_before_detection": config.mirror,
        },
        "hands": hands,
    }
