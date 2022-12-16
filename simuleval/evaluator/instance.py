# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
import math
from typing import Dict, List, Optional
import sacrebleu
from pathlib import Path

from simuleval.data.segments import TextSegment, SpeechSegment, EmptySegment

from simuleval.data.dataloader import GenericDataloader
from argparse import Namespace

try:
    import soundfile

    IS_IMPORT_SOUNDFILE = True
except:
    IS_IMPORT_SOUNDFILE = False


class Instance(object):
    def __init__(self, index, dataloader: GenericDataloader, args: Namespace):
        self.index = index
        self.finish_prediction = False
        self.dataloader = dataloader
        self.reset()
        self.source = self.dataloader[self.index]["source"]
        self.reference = self.dataloader[self.index]["target"]
        if args is not None:
            self.args = args
            self.latency_unit = getattr(args, "latency_unit", "word")

    def reset(self):
        self.step = 0
        self.elapsed = []
        self.prediction_list = []
        self.delays = []
        self.start_time = None
        self.metrics = {}

    @property
    def finish(self):
        return self.finish_prediction

    @finish.setter
    def finish(self, status: bool):
        self.finish_prediction = status

    def preprocess_target(self, target: str) -> str:
        """
        Preprocess the target, for example tokenization.
        """
        return target

    def preprocess_source(self, source: str):
        """
        Preprocess the source, for example tokenization.
        """
        raise NotImplementedError

    def receive_prediction(self, prediction: str):
        raise NotImplementedError

    def send_source(self, *args):
        raise NotImplementedError

    @property
    def source_length(self):
        raise NotImplementedError

    @property
    def prediction_length(self):
        return len(self.prediction_list)

    @property
    def target_length_latency(self):
        raise NotImplementedError

    @property
    def prediction(self):
        raise NotImplementedError

    @property
    def source_info(self):
        return self.source

    def summarize(self):
        return {
            "index": self.index,
            "prediction": self.prediction,
            "delays": self.delays,
            "elapsed": self.elapsed,
            "prediction_length": self.prediction_length,
            "reference": self.reference,
            "source": self.source_info,
            "source_length": self.source_length,
            "metric": self.metrics,
        }

    @classmethod
    def from_json(cls, json_string):
        info = json.loads(json_string)
        instance = cls(info["index"], None, None)
        instance.prediction_list = info["prediction"].split()
        instance.delays = info["delays"]
        instance.elapsed = info["elapsed"]
        instance.reference = info["reference"]
        instance.metrics = info["metric"]
        instance.finish_prediction = True
        return instance


class TextInputInstance(Instance):
    @property
    def source_length(self):
        return len(self.source)

    @property
    def source_info(self):
        return " ".join(self.source)

    def step_to_elapsed(self, *args):
        return 0

    def step_to_delay(self, step):
        return step

    def send_source(self, config_dict: Optional[Dict]):
        if self.step >= self.source_length:
            segment = EmptySegment(finished=True)
        else:
            segment = TextSegment(
                index=self.step,
                content=self.source[self.step],
                finished=(self.step == self.source_length - 1),
            )
            self.step += 1

        return segment


class TextOutputInstance(Instance):
    def receive_prediction(self, prediction: TextSegment):
        """
        Handler for receiving new predictions
        """

        if self.finish_prediction or prediction.is_empty:
            return

        if self.start_time is None:
            self.start_time = time.time()

        self.finish_prediction = prediction.finished

        if len(prediction.content) == 0:
            return

        current_time = time.time()

        if self.latency_unit == "word":
            prediction_list = [prediction.content]
        elif self.latency_unit == "char":
            prediction_list = prediction.content.split("")
        else:
            raise NotImplementedError

        self.prediction_list += prediction_list

        self.elapsed += [self.step_to_elapsed(self.step, current_time)] * len(
            prediction_list
        )
        self.delays += [self.step_to_delay(self.step)] * len(prediction_list)

    @property
    def target_length_latency(self):
        if self.latency_unit == "word":
            return len(self.reference.split(" "))
        elif self.latency_unit == "char":
            return len(self.reference)
        else:
            raise NotImplementedError

    @property
    def prediction(self):
        # TODO: Make configurable
        return " ".join([x for x in self.prediction_list])


class SpeechInputInstance(Instance):
    def __init__(self, index, dataloader, args):
        super().__init__(index, dataloader, args)
        self.sample_rate_value = None
        self.sample_list = None

    @property
    def sample_rate(self):
        if self.sample_rate_value is None:
            self.audio_info = self.dataloader.get_source_audio_info(self.index)
            self.sample_rate_value = self.audio_info.samplerate
        return self.sample_rate_value

    @property
    def samples(self) -> List[float]:
        if self.sample_list is None:
            self.sample_list = self.source
        return self.sample_list

    @property
    def is_finish_source(self):
        return self.step == len(self.samples)

    def send_source(self, segment_size=10):

        if self.step == 0:
            self.start_time = time.time()
        assert segment_size >= 1, "instance size has to larger than 1 ms"

        num_samples = math.ceil(segment_size / 1000 * self.sample_rate)

        if self.step < len(self.samples):
            if self.step + num_samples >= len(self.samples):
                # Pad zeros if the requested number of samples
                # are more than available samples.
                samples = self.samples[self.step :]
                is_finished = True
            else:
                samples = self.samples[self.step : self.step + num_samples]
                is_finished = False

            self.step = min(self.step + num_samples, len(self.samples))

            segment = SpeechSegment(
                index=self.len_sample_to_ms(self.step),
                content=samples,
                sample_rate=self.audio_info.samplerate,
                finished=is_finished,
            )

        else:
            # Finish reading this audio
            segment = EmptySegment(
                index=self.len_sample_to_ms(self.step),
                finished=True,
            )

        return segment

    @property
    def source_length(self):
        # In milliseconds
        return self.len_sample_to_ms(len(self.samples))

    @property
    def source_info(self):
        return str(self.audio_info).split("\n")

    def len_sample_to_ms(self, length):
        assert getattr(self, "sample_rate", None), "Read a audio file first"
        return length * 1000 / self.sample_rate

    def len_ms_to_samples(self, length):
        assert getattr(self, "sample_rate", None), "Read a audio file first"
        return math.ceil(length / 1000 * self.sample_rate)

    def step_to_delay(self, step):
        return self.len_sample_to_ms(self.step)

    def step_to_elapsed(self, step, current_time):
        return self.len_sample_to_ms(step) + (current_time - self.start_time) * 1000


class SpeechOutputInstance(Instance):
    def __init__(self, index, dataloader, args):
        super().__init__(index, dataloader, args)
        self.prediction_time = 0
        self.durations = []
        self.target_sample_rate = None
        assert IS_IMPORT_SOUNDFILE, "Please make sure soundfile is properly installed."

    @property
    def wav_path(self):
        wav_dir_path = Path(self.args.output) / "wavs"
        wav_dir_path.mkdir(exist_ok=True)
        wav_path = wav_dir_path / f"{self.index}_pred.wav"
        return wav_path.absolute()

    @property
    def prediction(self):
        return self.wav_path

    def summarize(self):
        samples = []
        # start from the first segment offset
        start = prev_end = self.delays[0]
        intervals = []

        for i, delay in enumerate(self.delays):

            start = max(prev_end, delay)

            if start > prev_end:
                # Wait source speech, add discontinuity with silence
                samples += [0.0] * int(
                    self.target_sample_rate * (start - prev_end) / 1000
                )
            samples += self.prediction_list[i]
            duration = self.durations[i]
            prev_end = start + duration
            intervals.append([start, duration])

        soundfile.write(self.wav_path, samples, self.target_sample_rate)

        return {
            "index": self.index,
            "prediction": self.wav_path.as_posix(),
            "delays": self.delays,
            "durations": self.durations,
            "prediction_offset": self.delays[0],
            "elapsed": [],
            "intervals": intervals,
            "prediction_length": len(samples) / self.target_sample_rate,
            "source_length": self.source_length,
            "reference": self.reference,
            "source": self.dataloader.get_source_audio_path(self.index),
        }

    def receive_prediction(self, segment: SpeechSegment):
        """
        Handler for receiving new predictions
        """
        if self.start_time is None:
            self.start_time = time.time()

        if self.finish_prediction:
            return

        self.finish_prediction = segment.finished

        if segment.is_empty:
            return

        if len(segment.content) == 0:
            return

        pred_duration = 1000 * len(segment.content) / segment.sample_rate

        if self.target_sample_rate is None:
            self.target_sample_rate = segment.sample_rate

        self.durations.append(pred_duration)
        self.prediction_list.append(segment.content)
        self.delays.append(self.step_to_delay(self.step))


class SpeechToTextInstance(SpeechInputInstance, TextOutputInstance):
    pass


class TextToTextInstance(TextInputInstance, TextOutputInstance):
    pass


class SpeechToSpeechInstance(SpeechInputInstance, SpeechOutputInstance):
    pass


INSTANCE_TYPE_DICT = {
    "speech-text": SpeechToTextInstance,
    "text-text": TextToTextInstance,
    "speech-speech": SpeechToSpeechInstance,
}


class LogInstance:
    def __init__(self, info: str) -> None:
        self.info = json.loads(info.strip())
        for key, value in self.info.items():
            setattr(self, key, value)

        self.reference = self.info.get("reference", "")
        self.source_length = self.info.get("source_length")  # just for testing!
        self.finish_prediction = True
