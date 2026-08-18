from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

def canonical_hand(value: str) -> str:
    """Return a canonical hand label."""
    normalised = value.strip().lower()
    if normalised == "left":
        return "Left"
    if normalised == "right":
        return "Right"
    return "Unknown"

@dataclass
class HandednessAccumulator:
    """Collect prediction counts for one known physical hand."""

    physical_hand: str
    target_frames: int
    counts: Counter[str] = field(default_factory=Counter)
    confidence_total: float = 0.0

    def __post_init__(self) -> None:
        self.physical_hand = canonical_hand(self.physical_hand)
        if self.physical_hand == "Unknown":
            raise ValueError("physical hand must be left or right")
        if self.target_frames <= 0:
            raise ValueError("target frames must be positive")

    @property
    def observed_frames(self) -> int:
        """Return the number of accepted single-hand frames."""
        return sum(self.counts.values())

    @property
    def complete(self) -> bool:
        """Return whether the requested number of frames has been observed."""
        return self.observed_frames >= self.target_frames

    def add(self, model_hand: str, confidence: float) -> None:
        """Add one model prediction."""
        if self.complete:
            raise ValueError("calibration stage is already complete")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.counts[canonical_hand(model_hand)] += 1
        self.confidence_total += confidence

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready summary for this physical hand."""
        observed = self.observed_frames
        if observed == 0:
            raise ValueError("cannot summarise an empty calibration stage")
        dominant = max(self.counts, key=self.counts.get)
        return {
            "physical_hand": self.physical_hand,
            "observed_frames": observed,
            "model_prediction_counts": dict(sorted(self.counts.items())),
            "dominant_model_prediction": dominant,
            "dominant_prediction_rate": self.counts[dominant] / observed,
            "physical_label_agreement_rate": (
                self.counts[self.physical_hand] / observed
            ),
            "mean_model_confidence": self.confidence_total / observed,
        }

def evaluate_calibration(
    results: list[dict[str, Any]],
    threshold: float = 0.9,
) -> str:
    """Classify a completed left-and-right calibration."""
    by_hand = {result["physical_hand"]: result for result in results}
    if set(by_hand) != {"Left", "Right"}:
        raise ValueError("calibration requires left and right results")
    if not 0.5 < threshold <= 1.0:
        raise ValueError("threshold must be greater than 0.5 and at most 1")

    labels_match = all(
        result["dominant_model_prediction"] == physical_hand
        and result["dominant_prediction_rate"] >= threshold
        for physical_hand, result in by_hand.items()
    )
    if labels_match:
        return "labels_match_physical_hands"

    labels_reversed = (
        by_hand["Left"]["dominant_model_prediction"] == "Right"
        and by_hand["Right"]["dominant_model_prediction"] == "Left"
        and by_hand["Left"]["dominant_prediction_rate"] >= threshold
        and by_hand["Right"]["dominant_prediction_rate"] >= threshold
    )
    if labels_reversed:
        return "labels_are_reversed"
    return "labels_are_inconsistent"

def build_calibration_report(
    accumulators: list[HandednessAccumulator],
    *,
    mirrored_before_detection: bool,
    camera_width: int,
    camera_height: int,
) -> dict[str, Any]:
    """Build a versioned calibration report."""
    if {item.physical_hand for item in accumulators} != {"Left", "Right"}:
        raise ValueError("calibration requires one left and one right stage")
    if any(not accumulator.complete for accumulator in accumulators):
        raise ValueError("all calibration stages must be complete")
    if camera_width <= 0 or camera_height <= 0:
        raise ValueError("camera dimensions must be positive")
    results = [accumulator.summary() for accumulator in accumulators]
    target_frames = {item.target_frames for item in accumulators}
    if len(target_frames) != 1:
        raise ValueError("calibration stages must use the same frame target")
    return {
        "schema_version": 1,
        "calibration_type": "mediapipe_handedness",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_frames_per_hand": target_frames.pop(),
        "camera": {
            "width": camera_width,
            "height": camera_height,
            "mirrored_before_detection": mirrored_before_detection,
        },
        "results": results,
        "interpretation": evaluate_calibration(results),
    }