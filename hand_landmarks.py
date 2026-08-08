import cv2 as cv
import mediapipe as mp
import time
from pathlib import Path
from webcam import open_camera

drawing_utils = mp.tasks.vision.drawing_utils
drawing_styles = mp.tasks.vision.drawing_styles
hand_connections = mp.tasks.vision.HandLandmarksConnections

model_path = Path(__file__).resolve().parent / "hand_landmarker.task"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

camera = open_camera()
start_time = time.perf_counter()

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("frame could not be captured")

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            for hand_landmarks in result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hand_connections.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )
            cv.imshow("Hand Landmarks", frame)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

finally:
    camera.release()
    cv.destroyAllWindows()