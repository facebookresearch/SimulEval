# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
import math
import soundfile
import sacrebleu

from simuleval import DEFAULT_EOS
from simuleval.metrics.latency import (
    AverageLagging,
    AverageProportion,
    DifferentiableAverageLagging
)


def eval_all_latency(delays, src_len):
    results = {}
    for name, func in {
        "AL": AverageLagging,
        "AP": AverageProportion,
        "DAL": DifferentiableAverageLagging
    }.items():
        results[name] = func(delays, src_len).item()

    return results


class Instance(object):
    def __init__(self, instance_id, source, target):
        self.finish_read = False
        self.finish_hypo = False
        self.target = self.preprocess_target(target)
        self.source = self.preprocess_source(source)
        self.step = 0
        self.elapsed = []
        self.hypos = []
        self.delays = []
        self.start_time = None
        self.metrics = {}
        self.instance_id = instance_id

    @property
    def finish(self):
        return self.finish_hypo

    @finish.setter
    def finish(self, status: bool):
        if status:
            self.sentence_level_eval()
        self.finish_hypo = status

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

    def recv_hypo(self, list_hypo: str):
        """
        Handler for receiving new predictions
        """
        if self.finish:
            return

        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time()

        for hypo in list_hypo:
            self.hypos.append(hypo)
            self.elapsed.append(self.step_to_elapsed(self.step, current_time))
            self.delays.append(self.step_to_delay(self.step))
            if hypo in [DEFAULT_EOS]:
                self.finish = True
                return

    def send_src(self, **kwargs):
        raise NotImplementedError

    def summarize(self):
        return {
            "index": self.instance_id,
            "prediction": self.prediction(),
            "delays": self.delays,
            "elapsed": self.elapsed,
            "prediction_length": self.target_length(),
            "reference": self.reference(),
            "source": self.source_info(),
            "source_length": self.source_length(),
            "reference_length": len(self.target),
            "metric": self.metrics,
        }

    def prediction(self, eos=True):
        if eos:
            return " ".join(self.hypos)
        else:
            return " ".join(x for x in self.hypos if x != DEFAULT_EOS)

    def source_length(self):
        raise NotImplementedError

    def target_length(self):
        return len(self.hypos)

    def reference(self):
        return " ".join(self.target)

    def source_info(self):
        raise NotImplementedError

    def sentence_level_eval(self):
        self.metrics["sentence_bleu"] = sacrebleu.sentence_bleu(
            self.prediction(), self.reference()
        ).score
        self.metrics["latency"] = eval_all_latency(
            self.delays, self.source_length() + 1)

    def step_to_delay(self, step):
        return step

    def step_to_elapsed(self, step, current_time):
        return (current_time - self.start_time) * 1000


class TextInstance(Instance):
    def preprocess_source(self, source):
        return source.strip().split()

    def source_length(self):
        return len(self.source)

    def source_info(self):
        return " ".join(self.source)

    def send_src(self, **kwargs):
        if self.step == 0:
            self.start_time = time.time()

        if self.step >= self.source_length():
            dict_to_return = {"segment_id": self.step, "segment": DEFAULT_EOS}
            # Consider EOS
            self.step = self.source_length() + 1
        else:
            dict_to_return = {"segment_id": self.step,
                              "segment": self.source[self.step]}
            self.step += 1

        return dict_to_return


class AudioInstance(Instance):
    def preprocess_source(self, source):
        # Only get info (sample rate), read audio file when first read request
        # happens
        self.audio_info = soundfile.info(source)
        self.sample_rate = self.audio_info.samplerate
        self.samples = []
        return source

    def send_src(self, segment_size=10):

        if self.step == 0:
            self.start_time = time.time()
            self.load_audio_from_path(self.source)

        assert segment_size >= 1, "instance size has to larger than 1 ms"

        num_samples = math.ceil(segment_size / 1000 * self.sample_rate)

        if self.step < len(self.samples):

            instance = self.samples[self.step: self.step + num_samples]

            self.step = min(self.step + num_samples, len(self.samples))

            dict_to_return = {
                "segment_id": self.len_sample_to_ms(self.step),
                "segment": instance,
                "sample_rate": self.audio_info.samplerate,
                "dtype": "int16",
            }

        else:
            # Finish reading this audio
            dict_to_return = {
                "segment_id": self.source_length(),
                "segment": DEFAULT_EOS,
                "sample_rate": self.audio_info.samplerate,
                "dtype": "int16",
            }

        return dict_to_return

    def load_audio_from_path(self, wav_path):
        assert os.path.isfile(wav_path) and wav_path.endswith('.wav')
        samples, _ = soundfile.read(wav_path, dtype="int16")
        self.samples = samples.tolist()

    def source_length(self):
        # In milliseconds
        return self.len_sample_to_ms(len(self.samples))

    def source_info(self):
        return str(self.audio_info).split("\n")

    def len_sample_to_ms(self, length):
        assert getattr(self, "sample_rate", None), "Read a audio file first"
        return length / self.sample_rate * 1000

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
            self.elapsed, self.source_length() + 1)
