import cv2 as cv

def open_camera(width=640, height=480, fps=30):
    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("could not open camera 0")

    camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv.CAP_PROP_FPS, fps)

    return camera

def main():
    camera = open_camera()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("frame could not be captured")

            cv.imshow("Webcam (Logitech C922)", frame)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()