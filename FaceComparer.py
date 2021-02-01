import uuid

import numpy as np


class FaceComparer:
    def __init__(self, min_distance=0.6):
        self.min_distance = min_distance

    def match_face(self, face, known_people):
        closest_match    = None
        closest_distance = float('infinity')

        # calculate distance between this face and all others
        for person_id in known_people:
            _, person_features = known_people[person_id]

            distance = np.linalg.norm(face.features - person_features)

            if distance < self.min_distance and distance < closest_distance:
                closest_match    = person_id
                closest_distance = distance

        if closest_match is not None:
            face.person_id = closest_match
        else:
            face.is_new_person = True
            face.person_id     = str(uuid.uuid4())

        return face
