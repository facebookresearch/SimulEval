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
import numpy as np

try:
    import wave
    import pyaudio
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
except:
    wave, pyaudio, load_silero_vad, read_audio, get_speech_timestamps = [
        None for _ in range(5)
    ]

from simuleval.data.segments import (
    Segment,
    segment_from_json_string,
    SpeechSegment,
    EmptySegment,
)
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
        if None in [wave, pyaudio, load_silero_vad, read_audio, get_speech_timestamps]:
            raise Exception(
                "Please install wave, pyaudio, and silero_vad to run the demo"
            )
        super().__init__(evaluator)
        self.float_array = np.asarray([])
        self.sample_rate = 16000
        self.finished = False
        self.queue = Queue(maxsize=0)
        self.VADmodel = load_silero_vad()
        self.silence_count = 0

    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1 if sys.platform == "darwin" else 2
        RATE = self.sample_rate
        RECORD_SECONDS = 10000  # Indefinite time

        with wave.open(f"output.wav", "wb") as wf:
            p = pyaudio.PyAudio()
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

            all_data = bytearray()
            start = time.time()
            for _ in range(0, round(RATE // CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                wf.writeframes(data)
                all_data += data
                if time.time() - start > (self.source_segment_size / 1000.0):
                    self.queue.put(all_data)
                    all_data = bytearray()
                    start = time.time()

            self.queue.put(all_data)
            stream.close()
            p.terminate()
            self.finished = True

    def remote_eval(self):
        # Initialization
        self.system_reset()
        recording = threading.Thread(target=self.record_audio)
        recording.start()

        # Start recording
        print("Recording...")
        while not self.finished or not self.queue.empty():
            data = byte_to_float(self.queue.get()).tolist()
            # VAD
            speech_timestamps = get_speech_timestamps(
                audio=data, model=self.VADmodel, sampling_rate=self.sample_rate
            )

            if len(speech_timestamps) != 0:  # has audio
                self.silence_count = 0
            else:
                self.silence_count += 1

            if self.silence_count <= 4:
                segment = SpeechSegment(
                    index=self.source_segment_size,
                    content=data,
                    sample_rate=self.sample_rate,
                    finished=False,
                )
                self.send_source(segment)
                output_segment = self.receive_prediction()
                if len(output_segment.content) == 0:
                    continue
                prediction_list = str(output_segment.content.replace(" ", ""))
                print(prediction_list, end=" ")
                sys.stdout.flush()

            else:
                segment = SpeechSegment(
                    index=self.source_segment_size,
                    content=[0.0 for _ in range(8192)],
                    sample_rate=self.sample_rate,
                    finished=True,
                )
                self.send_source(segment)
                output_segment = self.receive_prediction()
                self.silence_count = 0
                self.system_reset()


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
