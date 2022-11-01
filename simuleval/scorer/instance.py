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

from simuleval import DEFAULT_EOS
from simuleval.metrics.latency import (
    AverageLagging,
    AverageProportion,
    DifferentiableAverageLagging,
)
import soundfile


def eval_all_latency(delays, src_len, ref_len=None):
    if ref_len is None:
        ref_len = len(delays)
    results = {}
    for name, func in {
        "AL": AverageLagging,
        "AP": AverageProportion,
        "DAL": DifferentiableAverageLagging,
    }.items():
        results[name] = func(delays, src_len, ref_len).item()

    return results


class Instance(object):
    def __init__(self, index, dataloader, args):
        self.index = index
        self.finish_prediction = False
        self.dataloader = dataloader
        self.reset()
        self.source = None
        self.reference = None
        if args is not None:
            self.args = args
            self.latency_unit = getattr(args, "latency_unit", "word")

    def get_reference(self):
        if self.reference is None:
            self.reference = self.preprocess_target(
                self.dataloader[self.index]["target"]
            )
        return self.reference

    def get_source(self):
        if self.source is None:
            self.source = self.preprocess_target(self.dataloader[self.index]["source"])
        return self.source

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
        if status:
            self.sentence_level_eval()
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

    def sentence_level_eval(self):
        self.metrics["sentence_bleu"] = sacrebleu.sentence_bleu(
            self.prediction, [self.get_reference()]
        ).score
        self.metrics["latency"] = eval_all_latency(
            self.delays,
            self.source_length,
            self.target_length_latency,
        )

    def summarize(self):
        return {
            "index": self.index,
            "prediction": self.prediction,
            "delays": self.delays,
            "elapsed": self.elapsed,
            "prediction_length": self.prediction_length,
            "reference": self.get_reference(),
            # "source": self.source_info(),
            # "source_length": self.source_length(),
            # "reference_length": self.reference_length(),
            "metric": self.metrics,
        }

    @classmethod
    def from_json(cls, json_string):
        info = json.loads(json_string)
        instance = cls(info["index"], None, None)
        instance.prediction_list = info["prediction"].split() + [DEFAULT_EOS]
        instance.delays = info["delays"]
        instance.elapsed = info["elapsed"]
        instance.reference = info["reference"]
        instance.metrics = info["metric"]
        instance.finish_prediction = True
        return instance


class TextInputInstance(Instance):
    def preprocess_source(self, source: str) -> List[str]:
        return source.strip().split()  # TODO: add configurable tokenizer

    def source_length(self):
        return len(self.get_source())

    def source_info(self):
        return " ".join(self.get_source())

    def send_source(self, config_dict: Optional[Dict]):
        if self.step == 0:
            self.start_time = time.time()

        if self.step >= self.source_length():
            dict_to_return = {"segment_id": self.step, "segment": DEFAULT_EOS}
            self.step = self.source_length() + 1  # Consider EOS
        else:
            dict_to_return = {
                "segment_id": self.step,
                "segment": self.source[self.step],
            }
            self.step += 1

        return dict_to_return


class TextOutputInstance(Instance):
    def receive_prediction(self, prediction: str):
        """
        Handler for receiving new predictions
        """
        if self.finish_prediction:
            return

        self.finish_prediction = DEFAULT_EOS == prediction

        if self.finish_prediction:
            self.sentence_level_eval()
            return

        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time()

        if self.latency_unit == "word":
            prediction_list = [prediction]
        elif self.latency_unit == "char":
            prediction_list = prediction.split("")
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
            return len(self.get_reference().split(" "))
        elif self.latency_unit == "char":
            return len(self.get_reference())
        else:
            raise NotImplementedError

    @property
    def prediction(self):
        # TODO: Make configurable
        return " ".join([x for x in self.prediction_list if x != DEFAULT_EOS])


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
            self.sample_list = self.get_source()
        return self.sample_list

    def send_source(self, segment_size=10):

        if self.step == 0:
            self.start_time = time.time()
        assert segment_size >= 1, "instance size has to larger than 1 ms"

        num_samples = math.ceil(segment_size / 1000 * self.sample_rate)

        if self.step < len(self.samples):
            if self.step + num_samples >= len(self.samples):
                # Pad zeros if the requested number of samples
                # are more than available samples.
                instance = self.samples[self.step :]
                is_finished = True
            else:
                instance = self.samples[self.step : self.step + num_samples]
                is_finished = False

            self.step = min(self.step + num_samples, len(self.samples))

            dict_to_return = {
                "segment_id": self.len_sample_to_ms(self.step),
                "segment": instance,
                "sample_rate": self.audio_info.samplerate,
                "dtype": "int16",
                "finished": is_finished,
            }

        else:
            # Finish reading this audio
            dict_to_return = {
                "segment_id": self.source_length,
                "segment": DEFAULT_EOS,
                "sample_rate": self.audio_info.samplerate,
                "dtype": "int16",
                "finished": True,
            }

        return dict_to_return

    @property
    def source_length(self):
        # In milliseconds
        return self.len_sample_to_ms(len(self.samples))

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

    def sentence_level_eval(self):
        super().sentence_level_eval()
        self.metrics["latency_ca"] = eval_all_latency(
            self.elapsed, self.source_length, self.target_length_latency
        )


class SpeechOutputInstance(Instance):
    def __init__(self, index, dataloader, args):
        super().__init__(index, dataloader, args)
        self.prediction_time = 0
        self.time_alignment = []
        self.target_sample_rate = None

    def summarize(self):
        target_duration = self.time_alignment[0][1]
        samples = []
        # from fairseq import pdb;pdb.set_trace()
        for i in range(len(self.prediction_list)):
            source_duration = self.time_alignment[i][1]
            # print(target_duration, source_duration)

            if target_duration >= source_duration:
                # No need to wait source speech
                samples += self.prediction_list[i]["samples"]
            else:
                # Wait source speech, add discontinuity
                offset = source_duration - target_duration  # in ms
                samples += [0.0] * int(
                    self.target_sample_rate * offset / 1000
                ) + self.prediction_list[i]["samples"]

            target_duration = len(samples) / self.target_sample_rate * 1000

        wav_path = Path(self.args.output) / "wavs" / f"{self.index}_pred.wav"
        soundfile.write(wav_path, samples, self.target_sample_rate)
        # with open(Path(self.args.output) / "wavs" / f"{self.index}_pred.txt", "w") as f:
        #    f.write(self.get_reference() + "\n")

        return {
            "index": self.index,
            "prediction": wav_path.absolute().as_posix(),
            "delays": self.time_alignment,
            "prediction_offset": self.time_alignment[0][1],
            "elapsed": [],
            "prediction_length": target_duration / 1000,
            "source_length": self.source_length,
            "reference": self.get_reference(),
        }

    def receive_prediction(self, prediction: str):
        """
        Handler for receiving new predictions
        """
        # from fairseq import pdb;pdb.set_trace()
        if self.finish_prediction:
            return

        info_dict = json.loads(prediction)

        pred_samples = info_dict["samples"]
        pred_duration = 1000 * len(info_dict["samples"]) / info_dict["sample_rate"]

        if self.target_sample_rate is None:
            self.target_sample_rate = info_dict["sample_rate"]

        self.finish_prediction = DEFAULT_EOS == pred_samples

        if self.finish_prediction:
            self.sentence_level_eval()
            return

        if self.start_time is None:
            self.start_time = time.time()

        # current_time = time.time()
        self.prediction_time += pred_duration

        self.prediction_list.append(info_dict)

        self.time_alignment.append(
            [self.prediction_time, self.step_to_delay(self.step)]
        )


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
    def __init__(self, info: Dict) -> None:
        self.info = info

    def summarize(self):
        return self.info

    @property
    def reference(self):
        return self.info.get("reference", "")

    @property
    def source_length(self):
        return self.info.get("source_length", 0)