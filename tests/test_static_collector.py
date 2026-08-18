"""tests for the static data collector's pure data helpers"""

import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bsl_sign_recognition.static_samples import (
    CollectorConfig,
    build_sample,
    max_pairwise_displacement,
    serialize_hands,
)


def _landmarks(wrist_x: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(x=wrist_x + index / 1000, y=0.4, z=-0.01)
        for index in range(21)
    ]


def _serialised_hands(offset: float = 0.0) -> list[dict[str, Any]]:
    hands = []
    for screen_order, base_x in enumerate((0.2, 0.7)):
        image_landmarks = [
            [base_x + offset + index / 1000, 0.4 + offset, -0.01]
            for index in range(21)
        ]
        hands.append(
            {
                "screen_order": screen_order,
                "model_handedness": "Left" if screen_order == 0 else "Right",
                "model_handedness_score": 0.9,
                "image_landmarks": image_landmarks,
                "world_landmarks": [point.copy() for point in image_landmarks],
            }
        )
    return hands


class StaticSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CollectorConfig(
            participant_id="p001",
            session_id="test-session",
            camera_index=0,
            width=640,
            height=480,
            fps=30,
            model_path=Path("model.task"),
            data_directory=Path("data/raw"),
            countdown_seconds=2.0,
            stability_frames=12,
            max_mean_displacement=0.015,
            mirror=True,
        )

    def test_serialize_hands_sorts_by_wrist_x(self) -> None:
        right_on_screen = _landmarks(0.8)
        left_on_screen = _landmarks(0.2)
        result = SimpleNamespace(
            hand_landmarks=[right_on_screen, left_on_screen],
            hand_world_landmarks=[right_on_screen, left_on_screen],
            handedness=[
                [SimpleNamespace(category_name="Left", score=0.9)],
                [SimpleNamespace(category_name="Right", score=0.8)],
            ],
        )

        hands = serialize_hands(result)

        self.assertEqual(hands[0]["screen_order"], 0)
        self.assertEqual(hands[0]["model_handedness"], "Right")
        self.assertEqual(len(hands[0]["image_landmarks"]), 21)
        self.assertEqual(len(hands[1]["world_landmarks"]), 21)

    def test_build_sample_is_single_pose_not_sequence(self) -> None:
        hand = {
            "screen_order": 0,
            "model_handedness": "Left",
            "model_handedness_score": 0.9,
            "image_landmarks": [[0.0, 0.0, 0.0]] * 21,
            "world_landmarks": [[0.0, 0.0, 0.0]] * 21,
        }
        sample = build_sample(
            label="a",
            hands=[hand, {**hand, "screen_order": 1}],
            config=self.config,
            captured_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
            width=640,
            height=480,
            stability_displacement=0.003,
        )

        self.assertEqual(sample["sample_type"], "static_pose")
        self.assertEqual(sample["hand_count"], 2)
        self.assertIn("hands", sample)
        self.assertNotIn("frames", sample)
        self.assertNotIn("duration_ms", sample)
        self.assertEqual(sample["quality"]["stability_frames"], 12)

    def test_build_sample_rejects_missing_hand(self) -> None:
        with self.assertRaises(ValueError):
            build_sample(
                label="a",
                hands=[],
                config=self.config,
                captured_at=datetime.now(timezone.utc),
                width=640,
                height=480,
                stability_displacement=0.003,
            )

    def test_build_sample_rejects_wrong_landmark_count(self) -> None:
        malformed_hand = {
            "screen_order": 0,
            "model_handedness": "Left",
            "model_handedness_score": 0.9,
            "image_landmarks": [[0.0, 0.0, 0.0]] * 20,
            "world_landmarks": [[0.0, 0.0, 0.0]] * 21,
        }
        with self.assertRaises(ValueError):
            build_sample(
                label="a",
                hands=[
                    malformed_hand,
                    {**malformed_hand, "screen_order": 1},
                ],
                config=self.config,
                captured_at=datetime.now(timezone.utc),
                width=640,
                height=480,
                stability_displacement=0.003,
            )

    def test_stable_pose_has_small_displacement(self) -> None:
        displacement = max_pairwise_displacement(
            [
                _serialised_hands(0.0),
                _serialised_hands(0.001),
                _serialised_hands(0.002),
            ]
        )
        self.assertLess(displacement, 0.004)

    def test_moving_pose_exceeds_default_threshold(self) -> None:
        displacement = max_pairwise_displacement(
            [_serialised_hands(0.0), _serialised_hands(0.03)]
        )
        self.assertGreater(displacement, self.config.max_mean_displacement)

    def test_slow_drift_across_window_exceeds_threshold(self) -> None:
        displacement = max_pairwise_displacement(
            [
                _serialised_hands(0.0),
                _serialised_hands(0.01),
                _serialised_hands(0.02),
            ]
        )
        self.assertGreater(displacement, self.config.max_mean_displacement)

    def test_build_sample_rejects_duplicate_screen_order(self) -> None:
        hands = _serialised_hands()
        hands[1]["screen_order"] = 0
        with self.assertRaises(ValueError):
            build_sample(
                label="a",
                hands=hands,
                config=self.config,
                captured_at=datetime.now(timezone.utc),
                width=640,
                height=480,
                stability_displacement=0.003,
            )

    def test_build_sample_rejects_non_finite_coordinate(self) -> None:
        hands = _serialised_hands()
        hands[0]["image_landmarks"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            build_sample(
                label="a",
                hands=hands,
                config=self.config,
                captured_at=datetime.now(timezone.utc),
                width=640,
                height=480,
                stability_displacement=0.003,
            )


if __name__ == "__main__":
    unittest.main()
