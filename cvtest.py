import cv2
import os
video_capture = cv2.VideoCapture("video2.mp4")
if video_capture.isOpened():
    print("yes")
else:
    print("No")
    
# print(os.path.isfile("/home/crc/nv-codec-headers/Makefile"))