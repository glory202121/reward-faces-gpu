from collections import defaultdict

import numpy as np


class Tracker:
    def __init__(self, valid_track_length=1, track_timeout_ms=100):
        self.valid_track_length = valid_track_length
        self.track_timeout_ms   = track_timeout_ms

        self.tracks = defaultdict(list)

    def match(self, faces, face_metadata):
        camera_name = face_metadata.camera_name
        timestamp   = face_metadata.timestamp

        these_tracks = self.tracks[camera_name]

        for face in faces:
            best_match = None
            best_iou   = 0

            for i in range(len(these_tracks)):
                latest_face = these_tracks[i][-1]

                iou = face.iou(latest_face)

                if iou > best_iou:
                    best_match = i
                    best_iou   = iou

            if best_match is not None:
                these_tracks[best_match].append(face)
            else:
                these_tracks.append([face])

        tracks_to_remove = []
        valid_faces      = []

        # find tracks we haven't seen in > 1 second. we delete old tracks at
        # this point, while also sending out the average face for those tracks
        for i in range(len(these_tracks)):
            latest_timestamp = these_tracks[i][-1].timestamp
            this_track       = these_tracks[i]

            if (timestamp - latest_timestamp) > self.track_timeout_ms:
                tracks_to_remove.append(i)

                if len(this_track) >= self.valid_track_length:
                    latest_face          = this_track[-1]
                    latest_face.features = self._get_average_face_vector(this_track)

                    valid_faces.append(latest_face)

        # prune old tracks
        for track_to_remove in reversed(tracks_to_remove):
            del these_tracks[track_to_remove]

        return valid_faces

    @staticmethod
    def _get_average_face_vector(track):
        all_features = []

        for face in track:
            all_features.append(face.features)

        all_features = np.array(all_features)

        return np.average(all_features, axis=0)
