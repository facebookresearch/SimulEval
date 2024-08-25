# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
import threading
import time
from queue import Queue

import wave
import numpy as np
import pyaudio

from simuleval.data.segments import Segment, segment_from_json_string, SpeechSegment
from simuleval.evaluator import SentenceLevelEvaluator
import requests

logger = logging.getLogger("simuleval.remote_evaluator")


class RemoteEvaluator:
    def __init__(self, evaluator: SentenceLevelEvaluator) -> None:
        self.evaluator = evaluator
        self.address = evaluator.args.remote_address
        self.port = evaluator.args.remote_port
        self.source_segment_size = evaluator.args.source_segment_size
        self.base_url = f"http://{self.address}:{self.port}"

    def send_source(self, segment: Segment):
        url = f"{self.base_url}/input"
        requests.put(url, data=segment.json())

    def receive_prediction(self) -> Segment:
        url = f"{self.base_url}/output"
        r = requests.get(url)
        return segment_from_json_string(r.text)

    def system_reset(self):
        requests.post(f"{self.base_url}/reset")

    def results(self):
        return self.evaluator.results()

    def remote_eval(self):
        for instance in self.evaluator.iterator:
            self.system_reset()
            while not instance.finish_prediction:
                self.send_source(instance.send_source(self.source_segment_size))
                # instance.py line 275, returns a segment object with all the floats in the 500 ms range

                output_segment = self.receive_prediction()
                # gets the prediction in text! like "This"...
                # refreshes each time. "This" for the 1st, "is" for the second

                instance.receive_prediction(output_segment)
                # instance.py line 190
                # processes data, gets in a prediction list with ["This", "is"] on 2nd iteration
            self.evaluator.write_log(instance)

        self.evaluator.dump_results()


class DemoRemote(RemoteEvaluator):
    def __init__(self, evaluator: SentenceLevelEvaluator) -> None:
        super().__init__(evaluator)
        self.float_array = np.asarray([])
        self.sample_rate = 22050
        self.finished = False
        self.queue = Queue(maxsize=0)

    def record_audio(self, counter):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1 if sys.platform == "darwin" else 2
        RATE = self.sample_rate
        RECORD_SECONDS = self.source_segment_size / 1000

        with wave.open(f"output{counter}.wav", "wb") as wf:
            p = pyaudio.PyAudio()
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

            for _ in range(0, round(RATE // CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                wf.writeframes(data)

                self.float_array = byte_to_float(data).tolist()

            stream.close()
            p.terminate()

    def read_from_audio(self):
        CHUNK = 1024
        with wave.open("test.wav", "rb") as wf:
            # Instantiate PyAudio and initialize PortAudio system resources (1)
            p = pyaudio.PyAudio()
            # Open stream (2)
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            # Play samples from the wave file (3)
            all_data = bytearray()
            start = time.time()
            while len(data := wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
                stream.write(data)
                all_data += data
                if time.time() - start > 0.5:
                    self.queue.put(all_data)
                    all_data = bytearray()
                    start = time.time()
                    # print("inside", self.queue.qsize())

            self.queue.put(all_data)
            # print("inside", self.queue.qsize())

            # Close stream (4)
            stream.close()
            # Release PortAudio system resources (5)
            p.terminate()
            self.finished = True
            # import pdb

            # pdb.set_trace()

    def remote_eval(self):
        # Initialization
        self.system_reset()
        recording = threading.Thread(target=self.read_from_audio)
        recording.start()

        while not self.finished or not self.queue.empty():
            # print(self.queue.qsize())
            data = byte_to_float(self.queue.get()).tolist()
            # print(self.queue.qsize())
            segment = SpeechSegment(
                index=self.source_segment_size,
                content=data,
                sample_rate=self.sample_rate,
                finished=False,
            )
            self.send_source(segment)
            output_segment = self.receive_prediction()
            # import pdb

            # pdb.set_trace()
            prediction_list = str(output_segment.content.replace(" ", ""))
            print(prediction_list, end=" ")
            sys.stdout.flush()
            # time.sleep(1)

        # print("Recording...")
        # counter = 0
        # while True:
        #     self.record_audio(counter)
        #     counter += 1
        #     segment = SpeechSegment(
        #         index=self.source_segment_size,
        #         content=self.float_array,
        #         sample_rate=self.sample_rate,
        #         finished=False,
        #     )
        #     self.send_source(segment)
        #     output_segment = self.receive_prediction()
        #     prediction_list = str(output_segment.content.replace(" ", ""))
        #     print(prediction_list, end=" ")
        #     sys.stdout.flush()


def pcm2float(sig, dtype="float32"):
    sig = np.asarray(sig)
    if sig.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    # pcm (16 bit) min = -32768, max = 32767, map it to -1 to 1 by dividing by max (32767)
    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte, dtype=np.int16), dtype="float32")
