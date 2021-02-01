import bz2
import os
import tempfile
import urllib.request

import cv2
import dlib
import numpy as np

from Face import Face
from HiddenPrints import HiddenPrints
from face_detector.inference_usbCam_face import TensoflowFaceDector


class FaceDetector:
    def __init__(self, face_threshold=0.5, require_frontal_face=False):
        self.face_threshold       = face_threshold
        self.require_frontal_face = require_frontal_face

        if self.require_frontal_face:
            shape_predictor_model_file = 'shape_predictor_5_face_landmarks.dat'

            if not os.path.isfile(shape_predictor_model_file):
                print('Face shape predictor model not found, downloading now...')

                model_file_url = 'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2'
                output_file    = os.path.join(tempfile.gettempdir(), os.path.basename(model_file_url))

                urllib.request.urlretrieve(model_file_url, output_file)

                print('Extracting shape predictor model...')
                with bz2.BZ2File(output_file, 'rb') as f, open(shape_predictor_model_file, 'wb') as o:
                    for data_block in iter(lambda: f.read(100 * 1024), b''):
                        o.write(data_block)

                print('Shape predictor model retrieved')

            self.frontal_face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor       = dlib.shape_predictor(shape_predictor_model_file)

        self.detector = TensoflowFaceDector()

    def find_faces(self, frame):
        """Find all faces in the given frame"""
        faces = []

        # we resize the frame to detect faces more quickly
        resized_frame = cv2.resize(frame, (0, 0), fx=1.0 / 2.0, fy=1.0 / 2.0, interpolation=cv2.INTER_CUBIC)

        with HiddenPrints():
            boxes, scores, classes, num_detections = self.detector.run(resized_frame)

        # get all boxes with confidence scores > face_threshold
        boxes = boxes[scores > self.face_threshold]

        frame_height, frame_width = frame.shape[:2]

        for box in boxes:
            # scale each box to match the resolution of the original frame
            box[0] *= frame_height
            box[1] *= frame_width
            box[2] *= frame_height
            box[3] *= frame_width

            box = box.astype(np.int)

            y, x, y_max, x_max = box

            width  = x_max - x
            height = y_max - y

            pad_multiplier = 1 if self.require_frontal_face else 0.25

            # pad the crop
            x_pad = int(pad_multiplier * width)
            y_pad = int(pad_multiplier * height)

            x      -= x_pad
            y      -= y_pad
            width  += 2 * x_pad
            height += 2 * y_pad

            # get the crop of the face
            crop = frame[y:y_max+y_pad, x:x_max+x_pad, :]

            if self.require_frontal_face:
                # find a frontal face in this crop
                frontal_faces = self.frontal_face_detector(crop, 0)

                if len(frontal_faces) > 0:
                    frontal_face = frontal_faces[0]

                    # align this face in the crop
                    shape = self.shape_predictor(crop, frontal_face)
                    crop  = dlib.get_face_chip(crop, shape, size=150, padding=0.25)

                    x      = x + frontal_face.left()
                    y      = y + frontal_face.top()
                    width  = frontal_face.right()  - frontal_face.left()
                    height = frontal_face.bottom() - frontal_face.top()

                    # package up the face and associated metadata
                    face = Face(x, y, width, height, crop)

                    faces.append(face)
            else:
                # package up the face and associated metadata
                face = Face(x, y, width, height, crop)

                faces.append(face)

        return faces
