import math
import soundfile
from argparse import Namespace, ArgumentParser
from fairseq.models.streaming.agents import TestTimeWaitKS2T
from simuleval.data.segments import SpeechSegment, EmptySegment
from simuleval.utils import build_system_from_dir
from pathlib import Path

class AudioFrontEnd:
    def __init__(self, wav_file, segment_size) -> None:
        self.samples, self.sample_rate = soundfile.read(wav_file)
        # print(len(self.samples), self.samples[:100])
        self.samples = self.samples.tolist()
        self.segment_size = segment_size
        self.step = 0
    def send_segment(self):
        """
        This is the front-end logic in simuleval instance.py
        """

        num_samples = math.ceil(self.segment_size / 1000 * self.sample_rate)
        # print("self.segment_size", self.segment_size)
        # print('num_samples is', num_samples)
        # print('self.sample_rate is', self.sample_rate)
        if self.step < len(self.samples):
            if self.step + num_samples >= len(self.samples):
                samples = self.samples[self.step :]
                is_finished = True
            else:
                samples = self.samples[self.step : self.step + num_samples]
                is_finished = False
            self.step = min(self.step + num_samples, len(self.samples))
            # print("len(samples) is", len(samples))
            # import pdb 
            # pdb.set_trace()
            segment = SpeechSegment(
                index=self.step / self.sample_rate * 1000,
                content=samples,
                sample_rate=self.sample_rate,
                finished=is_finished,
            )
        else:
            # Finish reading this audio
            segment = EmptySegment(
                index=self.step / self.sample_rate * 1000,
                finished=True,
            )
        return segment
parser = ArgumentParser()
source_segment_size = 320  # milliseconds
audio_frontend = AudioFrontEnd(
    wav_file=Path(__file__).resolve().parent / "debug/test_no_silence.wav",
    segment_size=source_segment_size,
)
print("building system from dir")
system = build_system_from_dir(
    Path(__file__).resolve().parent / "models/s2t_es-en_tt-waitk_multidomain"
)
print("finished building system from dir")
system_states = system.build_states()
while True:
    speech_segment = audio_frontend.send_segment()
    # Translation happens here
    output_segment = system.pushpop(speech_segment, system_states)
    print(output_segment)
    if output_segment.finished:
        break