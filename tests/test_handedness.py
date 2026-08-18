"""Tests for pure handedness-calibration helpers."""

import unittest

from bsl_sign_recognition.handedness import (
    HandednessAccumulator,
    build_calibration_report,
    evaluate_calibration,
)


def _completed_accumulator(
    physical_hand: str,
    predictions: list[str],
) -> HandednessAccumulator:
    accumulator = HandednessAccumulator(physical_hand, len(predictions))
    for prediction in predictions:
        accumulator.add(prediction, 0.95)
    return accumulator


class HandednessAccumulatorTests(unittest.TestCase):
    def test_summary_measures_physical_label_agreement(self) -> None:
        accumulator = _completed_accumulator(
            "Right",
            ["Right"] * 9 + ["Left"],
        )

        summary = accumulator.summary()

        self.assertEqual(summary["dominant_model_prediction"], "Right")
        self.assertAlmostEqual(summary["physical_label_agreement_rate"], 0.9)
        self.assertTrue(accumulator.complete)

    def test_add_rejects_invalid_confidence(self) -> None:
        accumulator = HandednessAccumulator("Left", 1)
        with self.assertRaises(ValueError):
            accumulator.add("Left", 1.1)


class CalibrationEvaluationTests(unittest.TestCase):
    def test_report_recognises_matching_labels(self) -> None:
        accumulators = [
            _completed_accumulator("Right", ["Right"] * 10),
            _completed_accumulator("Left", ["Left"] * 10),
        ]

        report = build_calibration_report(
            accumulators,
            mirrored_before_detection=True,
            camera_width=640,
            camera_height=480,
        )

        self.assertEqual(
            report["interpretation"],
            "labels_match_physical_hands",
        )
        self.assertEqual(report["target_frames_per_hand"], 10)

    def test_report_recognises_reversed_labels(self) -> None:
        accumulators = [
            _completed_accumulator("Right", ["Left"] * 10),
            _completed_accumulator("Left", ["Right"] * 10),
        ]
        results = [accumulator.summary() for accumulator in accumulators]

        self.assertEqual(
            evaluate_calibration(results),
            "labels_are_reversed",
        )

    def test_report_rejects_incomplete_stage(self) -> None:
        accumulators = [
            HandednessAccumulator("Right", 10),
            _completed_accumulator("Left", ["Left"] * 10),
        ]

        with self.assertRaises(ValueError):
            build_calibration_report(
                accumulators,
                mirrored_before_detection=True,
                camera_width=640,
                camera_height=480,
            )

    def test_mixed_predictions_are_inconsistent(self) -> None:
        accumulators = [
            _completed_accumulator(
                "Right",
                ["Right"] * 6 + ["Left"] * 4,
            ),
            _completed_accumulator("Left", ["Left"] * 10),
        ]
        results = [accumulator.summary() for accumulator in accumulators]

        self.assertEqual(
            evaluate_calibration(results),
            "labels_are_inconsistent",
        )


if __name__ == "__main__":
    unittest.main()
