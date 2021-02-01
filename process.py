import datetime
import multiprocessing
import queue
import sys
import threading

import cv2
import pytz
import urllib.request

import aws
from VideoStreamer import VideoStreamer
from MovementDetector import MovementDetector
from DoorHandler import DoorHandler
from FaceDetector import FaceDetector
from FaceFeatureGenerator import FaceFeatureGenerator
from FaceComparer import FaceComparer

# the names of cameras we want to process for historical data
camera_names = [
    '3rd Floor',
    'Back Entrance',
    'Back Entrance - Entry Camera',
    'Back Entrance - Exit Camera',
    'Main Entrance',
    'Main Entrance - Entry Camera',
    'Main Entrance - Exit Camera',
    # 'Roof 1',
    # 'Roof 2',
    'Side Entrance',
    'Side Entrance - Entry Camera',
    'Side Entrance - Exit Camera'
]

# this maps the name of the files saved to s3 to the pretty name
# e.g. 'Main_Entrance___Entry_Camera' -> 'Main Entrance - Entry Camera'
camera_name_map = {camera_name.replace(' ', '_').replace('-', '_'): camera_name for camera_name in camera_names}

# devices we want to process if we're not doing historical predictions.
# the rest of the stream string will be filled in based on if the given
# string starts with rtsp://
cameras = [
    #('Main', 'Main_Entrance___Entry_Camera_01_20201123084117.mp4'),
    #('Side', 'Side_Entrance___Entry_Camera_01_20201123124307.mp4'),
    #('Back', 'Back_Entrance___Entry_Camera_01_20201123091410.mp4'),
    # ('3rd Floor', '3rd_Floor_01_20201211111746.mp4')
    # ('Main', 'rtsp://admin:330andrew@192.168.1.207:554//h264Preview_01_main'),
    # ('Side', 'rtsp://admin:330andrew@192.168.1.200:554//h264Preview_01_main'),
    # ('Back', 'rtsp://admin:330andrew@192.168.1.203:554//h264Preview_01_main')
    ('Main', 'video2.mp4')
]

def process_historical(update_aws):
    import aws

    processed_videos = aws.get_processed_videos()

    start_after = None

    while True:
        objects = aws.list_objects(bucket='228byers', prefix='2020/', start_after=start_after)

        if 'Contents' not in objects:
            break

        keys = [c['Key'] for c in objects['Contents']]
        keys = [k for k in keys if k.endswith('.mp4')]

        for key in keys:
            if key in processed_videos:
                continue

            file_name = key.split('/')[-1]
            split     = file_name.split('_01_')

            camera_name = split[0]

            if camera_name not in camera_name_map:
                continue

            # get the start time of the video
            date_string = split[1].split('.')[0]

            year   = int(date_string[ 0:4 ])
            month  = int(date_string[ 4:6 ])
            day    = int(date_string[ 6:8 ])
            hour   = int(date_string[ 8:10])
            minute = int(date_string[10:12])
            second = int(date_string[12:14])

            # convert from EST to UTC
            start_time = datetime.datetime(year, month, day, hour, minute, second)
            start_time = pytz.timezone('US/Eastern').localize(start_time)
            start_time = int(start_time.timestamp() * 1000)

            camera_name   = camera_name_map[camera_name]
            camera_stream = aws.get_url('228byers', key)

            try:
                print('Downloading %s...' % file_name)
                urllib.request.urlretrieve(camera_stream, file_name)
            except:
                continue

            camera_stream = file_name

            print('Processing %s...' % file_name)
            video_streamer.start_stream(camera_name, camera_stream, start_time=start_time, delete_file=True, threaded=False)

            if update_aws:
                aws.set_processed_video(key)

            processed_videos.add(key)

        start_after = keys[-1]


def _preprocess_worker(input_queue, output_queue):

    movement_detector = MovementDetector()

    while True:
        frame_metadata = input_queue.get(block=True)

        if frame_metadata is None:
            output_queue.put(None)
            break

        movement_detected = movement_detector.detect_movement(frame_metadata)

        if movement_detected:
            output_queue.put(frame_metadata)

def _face_detection_worker(input_queue, output_queue, require_frontal_face=False, show_preview=False):
    # import gpu
    # gpu.init_gpus()
    # gpu.enable_mixed_precision()

    detector = FaceDetector(require_frontal_face=require_frontal_face)
    # tracker  = Tracker()

    while True:
        frame_metadata = input_queue.get(block=True)

        if frame_metadata is None:
            output_queue.put(None)
            break

        faces = detector.find_faces(frame_metadata.frame)
        # faces = tracker.match(faces, frame_metadata)

        print('%d faces are detected from frame taken %d' % (len(faces), frame_metadata.timestamp))

        frame_copy = None

        if show_preview:
            frame_copy = frame_metadata.frame.copy()

        del frame_metadata.frame

        for face in faces:
            face.location  = frame_metadata.camera_name
            face.timestamp = frame_metadata.timestamp

            output_queue.put((frame_metadata, face))

            if show_preview:
                x1 = face.x
                y1 = face.y
                x2 = face.x + face.width
                y2 = face.y + face.height

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if show_preview:
            cv2.namedWindow(frame_metadata.camera_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(frame_metadata.camera_name, 800, 600)

            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

            cv2.imshow(frame_metadata.camera_name, frame_copy)
            cv2.waitKey(1)

def _face_features_worker(input_queue, output_queue):
    # import gpu
    # gpu.init_gpus()
    # gpu.enable_mixed_precision()

    feature_generator = FaceFeatureGenerator()

    while True:
        item = input_queue.get(block=True)

        if item is None:
            output_queue.put(None)

            break

        frame_metadata, face = item

        face.features = feature_generator.generate_features(face)

        print('Extracted 128d feature vector of face:%s of frame %d' % (face.id, frame_metadata.timestamp))

        output_queue.put((frame_metadata, face))

def _face_comparison_worker(input_queue, door_open_queue, aws_update_queue=None, debug_logs=False):
    known_people = {}

    aws.get_known_people(known_people)

    face_comparer = FaceComparer()

    print('made face comparer')

    while True:
        item = input_queue.get(block=True)

        if item is None:
            if aws_update_queue is not None:
                aws_update_queue.put(None)

            break

        frame_metadata, face = item

        print('Matching face %s' % face.id)
        face = face_comparer.match_face(face, known_people)

        if face.is_new_person:
            if debug_logs:
                print('New person found, giving ID %s' % face.person_id)

            # make a copy to prevent reference leak
            known_people[face.person_id] = (-1, face.features.copy())

            # update known people asynchronously
            # aws.get_known_people(known_people)
        else:
            if debug_logs:
                print('Found person with ID %s' % face.person_id)

            if not frame_metadata.is_live:
                entity_id, _ = known_people[face.person_id]

                print("Entity id is %d" % entity_id)
                entity_id = 1
                if entity_id != -1:
                    door_open_queue.put((face.person_id, frame_metadata.camera_name))

        if aws_update_queue is not None:
            aws_update_queue.put(face)

if __name__ == '__main__':
    debug_logs            = True
    show_preview          = True
    update_aws            = False
    live                  = True
    historical            = False
    require_frontal_face  = False
    threaded_live_streams = True

    # limit queue size to prevent memory overflow
    preprocess_queue      = queue.Queue(maxsize=32)
    face_detection_queue  = queue.Queue(maxsize=32)
    face_feature_queue    = multiprocessing.Queue(maxsize=32)
    face_comparison_queue = multiprocessing.Queue(maxsize=32)
    door_open_queue = multiprocessing.Queue(maxsize=32)

    aws_update_queue  = None
    aws_update_thread = None

    if update_aws:
        aws_update_queue = multiprocessing.Queue(maxsize=32)

        aws_update_thread = threading.Thread(target=aws.aws_update_worker, args=(aws_update_queue,))
        aws_update_thread.start()

    video_streamer = VideoStreamer(preprocess_queue, debug_logs=debug_logs)

    preprocess_worker      = threading.Thread(target=_preprocess_worker,             args=(preprocess_queue,      face_detection_queue))
    face_detection_worker  = threading.Thread(target=_face_detection_worker,         args=(face_detection_queue,  face_feature_queue, require_frontal_face, show_preview))
    face_features_worker   = multiprocessing.Process(target=_face_features_worker,   args=(face_feature_queue,    face_comparison_queue))
    face_comparison_worker = multiprocessing.Process(target=_face_comparison_worker, args=(face_comparison_queue, door_open_queue, aws_update_queue, debug_logs))

    preprocess_worker.start()
    face_detection_worker.start()
    face_features_worker.start()
    face_comparison_worker.start()

    door_handler  = DoorHandler()
    door_handler.start_door_open_thread(door_open_queue)

    if live:
        for camera_name, camera_stream in cameras:
            video_streamer.start_stream(camera_name, camera_stream, start_time=None, delete_file=False, threaded=threaded_live_streams)

    if historical:
        process_historical(update_aws)

    # process until all streams finish
    video_streamer.join()

    # let the processors know they can exit now
    preprocess_queue.put(None)

    face_comparison_worker.join()

    if update_aws:
        aws_update_thread.join()
