"""PyAudio Example: Record a few seconds of audio and save to a wave file."""

import wave
import sys

import pyaudio


class DemoRemote:
    def __init__(self, record_seconds):
        self.record_seconds = record_seconds

    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1 if sys.platform == "darwin" else 2
        RATE = 22050
        RECORD_SECONDS = self.record_seconds

        with wave.open("output.wav", "wb") as wf:
            p = pyaudio.PyAudio()
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

            print("Recording...")
            for _ in range(0, round(RATE // CHUNK * RECORD_SECONDS)):
                wf.writeframes(stream.read(CHUNK))
            print("Done")

            stream.close()
            p.terminate()
