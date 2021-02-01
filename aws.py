import io
import os
import sys
import tempfile
import threading

import boto3
from boto3.dynamodb.types import Binary
import cv2
from dotenv import load_dotenv
import numpy as np

# load aws credentials from .env file
load_dotenv()

if 'AWS_ACCESS_KEY_ID' not in os.environ or 'AWS_SECRET_ACCESS_KEY' not in os.environ:
    print('\nAWS credentials not found! Create a .env file with the following info:')
    print('\tAWS_ACCESS_KEY_ID=<key_id>')
    print('\tAWS_SECRET_ACCESS_KEY=<secret_key>')
    print('\tREGION=<region>')
    sys.exit(1)

# set up aws session and various resources
aws_session = boto3.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name=os.environ['REGION'])

dynamodb               = aws_session.resource('dynamodb')
faces_table            = dynamodb.Table('faces')
people_table           = dynamodb.Table('people')
processed_videos_table = dynamodb.Table('processed_videos')

s3 = aws_session.resource('s3')


def get_known_people(known_people):
    # get_known_people_thread = threading.Thread(target=_get_known_people, args=(known_people,))
    # get_known_people_thread.start()
    _get_known_people(known_people)


def _get_known_people(known_people):
    print("Getting known people from aws")
    # people = people_table.scan()['Items']
    people = []

    for person in people:
        known_people[person['id']] = (person['entity_id'], _bytes_to_numpy(person['face_vector'].value))
    
    print ("Known peoples are: ")
    print(known_people)


def get_processed_videos():
    processed_videos = set()

    videos = processed_videos_table.scan()['Items']

    for video in videos:
        processed_videos.add(video['key'])

    return processed_videos


def set_processed_video(key):
    processed_videos_table.put_item(Item={'key': key})


def aws_update_worker(face_queue):
    while True:
        face = face_queue.get(block=True)

        if face is None:
            break

        if not face.is_new_person:
            save_face_to_faces_table(face)
        else:
            save_face_to_people_table(face)


def save_face_to_faces_table(face):
    # upload face image to s3
    image_file = os.path.join(tempfile.gettempdir(), '%s.png' % face.id)
    s3_key     = 'faces/%s.png' % face.id

    cv2.imwrite(image_file, face.crop[:, :, ::-1])
    s3.meta.client.upload_file(image_file, Bucket='228byers', Key=s3_key)
    os.remove(image_file)

    # convert numpy array to bytes
    feature_bytes = _numpy_to_bytes(face.features)

    # save item to dynamodb
    faces_table.put_item(Item={
        'id': face.id,
        'location': face.location,
        'timestamp': face.timestamp,
        's3_key': s3_key,
        'face_vector': Binary(feature_bytes),
        'person_id': face.person_id
    })


def save_face_to_people_table(face):
    # convert numpy array to bytes
    feature_bytes = _numpy_to_bytes(face.features)

    # save item to dynamodb
    people_table.put_item(Item={
        'id': face.person_id,
        'entity_id': -1,
        'first_seen': face.timestamp,
        'face_vector': Binary(feature_bytes)
    })


def list_objects(bucket, prefix, start_after=None):
    kwargs = {}

    if start_after is not None:
        kwargs['StartAfter'] = start_after

    return s3.meta.client.list_objects_v2(Bucket=bucket, Prefix=prefix, **kwargs)


def get_url(bucket, key):
    return s3.meta.client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600)


# noinspection PyTypeChecker
def _numpy_to_bytes(array):
    bbytes = io.BytesIO()
    np.save(bbytes, array, allow_pickle=True)

    return bbytes.getvalue()


# noinspection PyTypeChecker
def _bytes_to_numpy(bbytes):
    return np.load(io.BytesIO(bbytes), allow_pickle=True)
