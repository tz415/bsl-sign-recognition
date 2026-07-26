import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("could not open camera 0")

try:
    while True:
        success, frame = camera.read()

        if not success:
            raise RuntimeError("frame could not be captured")

        cv2.imshow("C922 Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()