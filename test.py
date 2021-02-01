import cv2
import datetime
import traceback
from FaceDetector import FaceDetector
from FaceFeatureGenerator import FaceFeatureGenerator


class FrameMetadata:
    def __init__(self, camera_name=None, frame=None, timestamp=None, is_live=None):
        self.camera_name = camera_name
        self.frame       = frame
        self.timestamp   = timestamp
        self.is_live     = is_live

camera_stream = "F:/Datasets/image/face/test/example.mp4"
if __name__ == '__main__':

    video_capture = cv2.VideoCapture(camera_stream)
    frame_index = 0

    start_time = None
    camera_name = "Test"
    is_live = True

    detector = FaceDetector(require_frontal_face=False)
    feature_generator = FaceFeatureGenerator()


    while True:
        try:
            if video_capture.isOpened():
                status, frame = video_capture.read()

                if status:
                    if start_time is None:
                        timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
                    else:
                        timestamp = int(start_time + video_capture.get(cv2.CAP_PROP_POS_MSEC))

                    frame_metadata = FrameMetadata()
                    frame_metadata.frame       = frame#= cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)
                    frame_metadata.camera_name = camera_name
                    frame_metadata.timestamp   = timestamp
                    frame_metadata.is_live     = is_live

                    print('Processing camera %s %d' % (frame_metadata.camera_name, frame_index))
                    
                    faces = detector.find_faces(frame_metadata.frame)
                    # faces = tracker.match(faces, frame_metadata)

                    if len(faces) > 0:
                        print('Detected Faces %d' % len(faces))

                    frame_copy = frame_metadata.frame.copy()

                    del frame_metadata.frame

                    for face in faces:
                        face.location  = frame_metadata.camera_name
                        face.timestamp = frame_metadata.timestamp

                        x1 = face.x
                        y1 = face.y
                        x2 = face.x + face.width
                        y2 = face.y + face.height

                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.namedWindow(frame_metadata.camera_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(frame_metadata.camera_name, 800, 600)

                        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

                        cv2.imshow(frame_metadata.camera_name, frame_copy)
                        cv2.waitKey(1)

                        face.features = feature_generator.generate_features(face)

                        print(face.features.shape)


                    frame_index += 1
                else:
                    video_capture.release()
                    break
        except:
            traceback.print_exc()
            video_capture.release()
            break

