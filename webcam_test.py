import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("could not open camera 0")

success, frame = camera.read()
camera.release()

if not success:
    raise RuntimeError("camera opened, but no frame was captured")

height, width, channels = frame.shape
print(f"captured frame: {width}x{height}, {channels} channels")