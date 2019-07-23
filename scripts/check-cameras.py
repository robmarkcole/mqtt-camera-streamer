"""
Check which inputs are a valid camera.
"""
import cv2
import time

for i in range(0, 5):
    cap = cv2.VideoCapture(i)
    is_camera = cap.isOpened()
    if is_camera:
        print(f"Input {i} is a valid camera value for VIDEO_SOURCE")
        cap.release()
        time.sleep(3)
