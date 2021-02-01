import datetime
import os
import threading
import traceback

import cv2


class VideoStreamer:
    def __init__(self, output_queue, frame_interval=3, max_streams=10, debug_logs=True):
        # we process every frame_interval frames to save processing time.
        # most adjacent frames are near identical, so processing every
        # frame is often redundant
        self.output_queue   = output_queue
        self.frame_interval = frame_interval
        self.max_streams    = max_streams
        self.debug_logs     = debug_logs

        self.stream_threads = []

    def start_stream(self, camera_name, camera_stream, start_time, delete_file, threaded):
        """Start processing a stream"""
        file_to_delete = None
        auto_restart   = False
        is_live        = False

        if camera_stream.startswith('rtsp://'):
            auto_restart = True
            is_live      = True

            camera_stream = 'rtspsrc location="%s" latency=0 drop-on-latency=true ! queue2 max-size-buffers=2 ! decodebin ! videoconvert ! appsink' % camera_stream
        else:
            if delete_file:
                file_to_delete = camera_stream

            camera_stream = 'filesrc location="%s" ! queue2 max-size-buffers=2 ! decodebin ! videoconvert ! appsink' % camera_stream

        if threaded:
            if len(self.stream_threads) == self.max_streams:
                self.stream_threads[0].join()

            stream_thread = threading.Thread(target=self._process_stream, args=(camera_name, camera_stream, is_live, start_time, file_to_delete, auto_restart))
            stream_thread.start()

            self.stream_threads.append(stream_thread)
        else:
            self._process_stream(camera_name, camera_stream, is_live, start_time, file_to_delete, auto_restart)

    def join(self):
        """Block until all streams are finished"""
        while True:
            has_running_threads = False

            for stream_thread in self.stream_threads:
                if stream_thread.is_alive():
                    has_running_threads = True
                    break

            if has_running_threads:
                for stream_thread in self.stream_threads:
                    stream_thread.join()
            else:
                break

    def _process_stream(self, camera_name, camera_stream, is_live, start_time, file_to_delete, auto_restart):
        video_capture = cv2.VideoCapture(camera_stream, cv2.CAP_GSTREAMER)
        # video_capture = cv2.VideoCapture("/home/crc/reward-faces/video2.mp4")

        frame_index = 0

        while True:
            try:
                if video_capture.isOpened():
                    status, frame = video_capture.read()

                    if status:
                        if (frame_index % self.frame_interval) == 0:
                            if start_time is None:
                                timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
                            else:
                                timestamp = int(start_time + video_capture.get(cv2.CAP_PROP_POS_MSEC))

                            frame_metadata = FrameMetadata()
                            frame_metadata.frame       = frame#= cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)
                            frame_metadata.camera_name = camera_name
                            frame_metadata.timestamp   = timestamp
                            frame_metadata.is_live     = is_live

                            if self.debug_logs:
                                print('Queue size: %d' % self.output_queue.qsize())
                                print('Processing camera %s' % frame_metadata.camera_name)
                                print()

                            self.output_queue.put(frame_metadata)

                        frame_index += 1
                    else:
                        video_capture.release()
                        break
                else:
                    print("Camera is not opened.")
                    break
            except:
                traceback.print_exc()
                video_capture.release()
                break

        if auto_restart:
            self._process_stream(camera_name, camera_stream, is_live, start_time, file_to_delete, auto_restart)

        if file_to_delete is not None:
            os.remove(file_to_delete)


class FrameMetadata:
    def __init__(self, camera_name=None, frame=None, timestamp=None, is_live=None):
        self.camera_name = camera_name
        self.frame       = frame
        self.timestamp   = timestamp
        self.is_live     = is_live
        self.faces       = []
