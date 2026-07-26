import cv2 as cv
import time

camera = cv.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("could not open camera 0")

# Webcam dimensions
requested_width = 640
requested_height = 480

camera.set(cv.CAP_PROP_FRAME_WIDTH, requested_width)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, requested_height)

# Webcam frames
requested_fps = 30
camera.set(cv.CAP_PROP_FPS, requested_fps)

frame_count = 0
measurement_duration = 5
start_time = time.perf_counter()

try:
    while True:
        success, frame = camera.read()

        if not success:
            raise RuntimeError("frame could not be captured")

        frame_count += 1
        elapsed_time = time.perf_counter() - start_time

        cv.imshow("Webcam (Logitech C922)", frame)

        # deducing actual height and width
        actual_height, actual_width = frame.shape[:2]
        print(actual_width, actual_height)
        # deducing the actual fps
        actual_fps = frame_count / elapsed_time
        print(actual_fps)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

        if elapsed_time >= measurement_duration:
            actual_height, actual_width = frame.shape[:2]
            actual_fps = frame_count / elapsed_time

            print(f"actual resolution: {actual_width} × {actual_height}")
            print(f"measured FPS: {actual_fps:.2f}") # .2f it because it's too long 
            break
finally:
    camera.release()
    cv.destroyAllWindows()