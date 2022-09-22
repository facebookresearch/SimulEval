# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from doctest import Example
from optparse import Option
import time
import os
import math
from typing import Dict, List, Optional
import soundfile
import sacrebleu
import re
import os
import soundfile as sf

from simuleval import DEFAULT_EOS
from simuleval.metrics.latency import (
    AverageLagging,
    AverageProportion,
    DifferentiableAverageLagging,
)


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
        self.latency_unit = args.latency_unit
        self.reset()
        self.load_data()

    def load_data(self):
        self.target = self.preprocess_target(self.dataloader[self.index]["target"])
        self.source = self.preprocess_source(self.dataloader[self.index]["source"])

    def reset(self):
        self.step = 0
        self.elapsed = []
        self.hypos = []
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

    def preprocess_target(self, target: str):
        """
        Preprocess the target, for example tokenization.
        """
        return target.strip().split()

    def preprocess_source(self, source: str):
        """
        Preprocess the source, for example tokenization.
        """
        raise NotImplementedError

    def receive_prediction(self, prediction: str):
        raise NotImplementedError

    def send_source(self, *args):
        raise NotImplementedError

    def summarize(self):
        return {
            "index": self.index,
            "prediction": self.prediction(),
            "delays": self.delays,
            "elapsed": self.elapsed,
            "prediction_length": self.target_length(),
            "reference": self.reference_info(),
            "source": self.source_info(),
            "source_length": self.source_length(),
            "reference_length": self.reference_length(),
            "metric": self.metrics,
        }

    def prediction(self, eos=True, no_space=False):
        join_char = "" if no_space else " "
        if eos:
            return join_char.join(self.hypos)
        else:
            return join_char.join(x for x in self.hypos if x != DEFAULT_EOS)


class TextInputInstance(Instance):
    def preprocess_source(self, source: str) -> List[str]:
        return source.strip().split()  # TODO: add configurable tokenizer

    def source_length(self):
        return len(self.source)

    def source_info(self):
        return " ".join(self.source)

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
        if self.finish:
            return

        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time()

        if self.latency_unit == "word":
            prediction_list = prediction.split(" ")
        elif self.latency_unit == "char":
            prediction_list = prediction.split("")
        else:
            raise NotImplementedError

        self.elapsed += [self.step_to_elapsed(self.step, current_time)] * len(
            prediction_list
        )
        self.delays += [self.step_to_delay(self.step)] * len(prediction_list)

        self.finish = DEFAULT_EOS in prediction

    def prediction_length(self):
        return len(self.prediction)

    def sentence_level_eval(self):
        self.metrics["sentence_bleu"] = sacrebleu.sentence_bleu(
            self.prediction, [self.reference]
        ).score
        self.metrics["latency"] = eval_all_latency(
            self.delays,
            self.source_length(),
            self.latency_reference_length() + 1,
        )

    def latency_reference_length(self):
        if self.latency_unit == "word":
            return len(self.reference.split(" "))
        elif self.latency_unit == "char":
            return len(self.reference)
        else:
            raise NotImplementedError

    def sentence_level_eval(self):
        self.metrics["sentence_bleu"] = sacrebleu.sentence_bleu(
            self.prediction, self.reference()
        ).score
        self.metrics["latency"] = eval_all_latency(
            self.delays,
            self.source_length() + 1,
            len(self.delays),
        )


class SpeechInputInstance(Instance):
    def preprocess_source(self, source):
        self.audio_info = self.dataloader.get_source_audio_info(self.index)
        self.sample_rate = self.audio_info.samplerate
        self.samples = source
        return source

    def send_src(self, segment_size=10):

        if self.step == 0:
            self.start_time = time.time()
        assert segment_size >= 1, "instance size has to larger than 1 ms"

        num_samples = math.ceil(segment_size / 1000 * self.sample_rate)

        if self.step < len(self.samples):
            if self.step + num_samples > len(self.samples):
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
                "segment_id": self.source_length(),
                "segment": DEFAULT_EOS,
                "sample_rate": self.audio_info.samplerate,
                "dtype": "int16",
                "finished": True,
            }

        return dict_to_return

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


class SpeechToTextInstance(SpeechInputInstance, TextOutputInstance):
    def sentence_level_eval(self):
        self.metrics["latency_ca"] = eval_all_latency(
            self.elapsed, self.source_length(), self.reference_length() + 1
        )


class TextToTextInstance(TextInputInstance, TextOutputInstance):
    pass
