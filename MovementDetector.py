from collections import defaultdict

import cv2
import numpy as np


class MovementDetector:
    def __init__(self):
        self.background_images     = {}
        self.kernel                = np.ones((3, 3), np.uint8)
        self.time_of_last_movement = defaultdict(int)

    def detect_movement(self, frame_metadata):
        # return True
        frame       = frame_metadata.frame
        timestamp   = frame_metadata.timestamp
        camera_name = frame_metadata.camera_name

        # get smaller, gray version of frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frame = cv2.resize(gray_frame, (0, 0), fx=1.0 / 5.0, fy=1.0 / 5.0, interpolation=cv2.INTER_NEAREST)
        gray_frame = gray_frame.astype(np.float64)

        # if this is the first frame from this camera, save it in memory
        if camera_name not in self.background_images:
            self.background_images[camera_name] = gray_frame

        # get the difference between this frame and the background image. We only care about
        # significant (>= 20) differences
        diff_frame = np.abs(gray_frame - self.background_images[camera_name])
        diff_frame[diff_frame < 20]  = 0
        diff_frame[diff_frame >= 20] = 255

        # erode the image to remove noise, then dilate the image to emphasize big differences
        # diff_frame = cv2.erode(diff_frame,  self.kernel, iterations=1)
        # diff_frame = cv2.dilate(diff_frame, self.kernel, iterations=3)

        # find contours in the difference image
        contours, _ = cv2.findContours(diff_frame.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = False

        # look for large areas of the difference image
        for contour in contours:
            contour_area = cv2.contourArea(contour)

            if contour_area > 250:
                movement_detected                       = True
                self.time_of_last_movement[camera_name] = timestamp

                break

        if not movement_detected:
            if (timestamp - self.time_of_last_movement[camera_name]) < 5000:
                movement_detected = True

        # update background image
        cv2.accumulateWeighted(gray_frame, self.background_images[camera_name], 0.1)

        return movement_detected
