"""Live webcam capture and MediaPipe hand-landmark visualisation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import mediapipe as mp

WINDOW_NAME = "BSL hand landmarks"
CAPTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "captures"

# MediaPipe hand-landmark indexes joined in the standard hand skeleton.
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class WebcamConfig:
    """Runtime configuration for the webcam prototype."""

    camera_index: int
    width: int
    height: int
    fps: int
    model_path: Path
    mirror: bool
    min_detection_confidence: float
    min_presence_confidence: float
    min_tracking_confidence: float


class FpsCounter:
    """Calculate a stable frame-rate estimate over a short rolling window."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def update(self) -> float:
        """Record the current time and return the rolling frames per second."""
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class Landmark(Protocol):
    """The coordinate fields used from a MediaPipe normalized landmark."""

    x: float
    y: float


def _open_camera(config: WebcamConfig) -> cv2.VideoCapture:
    """Open and configure a camera, or raise a useful runtime error."""
    camera = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"could not open camera {config.camera_index}. "
            "Close other camera apps or try `--camera 1`."
        )

    # Camera drivers may negotiate nearby values rather than exact requests.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    camera.set(cv2.CAP_PROP_FPS, config.fps)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(
        "Camera opened: "
        f"index={config.camera_index}, "
        f"resolution={actual_width}x{actual_height}, "
        f"reported_fps={actual_fps:.1f}"
    )
    return camera


def _landmark_to_pixel(landmark: Landmark, width: int, height: int) -> tuple[int, int]:
    """Convert one normalized MediaPipe landmark to a clamped pixel point."""
    x = min(max(float(landmark.x), 0.0), 1.0)
    y = min(max(float(landmark.y), 0.0), 1.0)
    return min(int(x * width), width - 1), min(int(y * height), height - 1)


def _draw_hand(
    frame: cv2.typing.MatLike,
    landmarks: list[Landmark],
    label: str,
) -> None:
    """Draw one hand skeleton and its handedness label onto a BGR frame."""
    height, width = frame.shape[:2]
    points = [_landmark_to_pixel(item, width, height) for item in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (80, 220, 120), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (30, 80, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 5, (255, 255, 255), 1, cv2.LINE_AA)

    label_x = max(min(point[0] for point in points), 6)
    label_y = max(min(point[1] for point in points) - 12, 24)
    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_status(
    frame: cv2.typing.MatLike,
    detected_hands: int,
    measured_fps: float,
) -> None:
    """Draw live diagnostics and keyboard controls."""
    height, width = frame.shape[:2]
    lines = (
        f"Hands: {detected_hands}/2",
        f"Processing: {measured_fps:.1f} FPS",
        f"Frame: {width}x{height}",
        "Q / Esc: quit   S: snapshot",
    )
    cv2.rectangle(frame, (8, 8), (350, 112), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 32 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def _save_snapshot(frame: cv2.typing.MatLike) -> Path:
    """Save an annotated snapshot and return its path."""
    CAPTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    destination = CAPTURE_DIRECTORY / f"landmarks-{timestamp}.jpg"
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"could not save snapshot to {destination}")
    return destination


def _create_landmarker(config: WebcamConfig) -> mp.tasks.vision.HandLandmarker:
    """Create a two-hand MediaPipe landmarker in tracking-enabled video mode."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=config.min_detection_confidence,
        min_hand_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def run_webcam(config: WebcamConfig) -> None:
    """Run the live camera loop until the user quits."""
    camera = _open_camera(config)
    fps_counter = FpsCounter()
    start_ns = time.monotonic_ns()
    consecutive_read_failures = 0

    print("Controls: press Q or Esc to quit; press S to save a snapshot.")
    try:
        with _create_landmarker(config) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 10:
                        raise RuntimeError(
                            "the camera opened but repeatedly failed to return frames."
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
                timestamp_ms = (time.monotonic_ns() - start_ns) // 1_000_000
                result = landmarker.detect_for_video(media_pipe_image, timestamp_ms)

                for index, hand_landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[index][0]
                    name = handedness.category_name or "Hand"
                    label = f"{name} {handedness.score:.0%}"
                    _draw_hand(frame, hand_landmarks, label)

                _draw_status(frame, len(result.hand_landmarks), fps_counter.update())
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    print(f"Snapshot saved: {_save_snapshot(frame)}")

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
