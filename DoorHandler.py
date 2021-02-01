import json
import time
from threading import Thread

import requests


class DoorHandler:
    unlock_url = 'https://dev.reward.com/crc-door/crc-lock/%d/open/%s?token=pPba48mkqm2RvtTy4PSXNPD8e95D63ajtX57Vfgb'

    door_ids = {
        'Main': 7,
        'Back': 6,
        'Side': 8,
        'Test':0
    }

    def start_door_open_thread(self, door_open_queue):
        door_open_thread        = Thread(target=self._door_open_worker, args=(door_open_queue,))
        door_open_thread.daemon = True
        door_open_thread.start()

    def _door_open_worker(self, door_open_queue):
        while True:
            item = door_open_queue.get(block=True)

            if item is None:
                break

            person_id, camera_name = item

            self._send_open_request(person_id, camera_name)

    def _send_open_request(self, person_id, camera_name):
        if camera_name not in self.door_ids:
            return

        door_id = self.door_ids[camera_name]

        requests.get(self.unlock_url % (door_id, person_id))

        print('Sent door open request for door %s and person %s' % (camera_name, person_id))

