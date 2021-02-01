import bz2
import os
import tempfile
import urllib.request

import cv2
import dlib
import numpy as np


class FaceFeatureGenerator:
    def __init__(self):
        generator_model_file = 'dlib_face_recognition_resnet_model_v1.dat'

        if not os.path.isfile(generator_model_file):
            print('Face feature generator model not found, downloading now...')

            model_file_url = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'
            output_file    = os.path.join(tempfile.gettempdir(), os.path.basename(model_file_url))

            urllib.request.urlretrieve(model_file_url, output_file)

            print('Extracting generator model...')
            with bz2.BZ2File(output_file, 'rb') as f, open(generator_model_file, 'wb') as o:
                for data_block in iter(lambda: f.read(100 * 1024), b''):
                    o.write(data_block)

            print('Generator model retrieved')

        self.generator = dlib.face_recognition_model_v1(generator_model_file)

    def generate_features(self, face):
        face_crop = cv2.resize(face.crop, (150, 150))

        features = self.generator.compute_face_descriptor(face_crop, num_jitters=5)

        return np.asarray(features)
