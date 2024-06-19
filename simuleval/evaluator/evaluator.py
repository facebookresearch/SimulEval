# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# test
import contextlib
import json
import logging
import numbers
import os, time
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Dict, Generator, Optional
import editdistance
from pydub import AudioSegment

import pandas
import yaml
from simuleval.data.dataloader import GenericDataloader
from simuleval.data.dataloader import build_dataloader
from simuleval.data.dataloader.dataloader import IterableDataloader
from tqdm import tqdm

from .instance import INSTANCE_TYPE_DICT, LogInstance
from .scorers import get_scorer_class
from .scorers.latency_scorer import LatencyScorer
from .scorers.quality_scorer import QualityScorer

try:
    import sentencepiece

    IS_IMPORT_SPM = True
except Exception:
    IS_IMPORT_SPM = False


logger = logging.getLogger("simuleval.sentence_level_evaluator")


def get_audio_duration(wav_path):
    audio = AudioSegment.from_wav(wav_path)
    return len(audio) / 1000.0  # pydub provides length in milliseconds


def get_RTF(eval_took_s, audio_file_list, start_index, end_index):
    with open(audio_file_list, "r") as f:
        fnames = [i[:-1] for i in f.readlines()[start_index:end_index]]
    dur = list(map(get_audio_duration, fnames))
    return eval_took_s / sum(dur)


def get_real_wer(output_path, source_file, start_index, end_index):
    """Calculate the WER between the ASR output and the source text."""
    total_distance = 0
    total_ref_words = 0
    with open(f"{output_path}/asr.log", "r") as f:
        hyp_setences = [i.replace("\n", "") for i in f.readlines()]
    with open(f"{source_file}.txt", "r") as f:
        ref_sentences = [i.replace("\n", "") for i in f.readlines()][start_index:end_index]

    for hyp_sentence, ref_sentence in zip(hyp_setences, ref_sentences):
        hyp_words = hyp_sentence.split()
        ref_words = ref_sentence.split()

        total_distance += editdistance.eval(ref_words, hyp_words)
        total_ref_words += len(ref_words)

    return round(100.0 * total_distance / total_ref_words, 2)


class SentenceLevelEvaluator(object):
    """
    Sentence Level evaluator. It iterates over sentence pairs and run evaluation.


    .. code-block:: python

        for instance in self.maybe_tqdm(self.instances.values()):
            agent.reset()
            while not instance.finish_prediction:
                input_segment = instance.send_source(self.source_segment_size)
                output_segment = agent.pushpop(input_segment)
                instance.receive_prediction(output_segment)


    Attributes:
        instances: collections of sentence pairs. Instances also keep track of delays.
        latency_scorers (List[~simuleval.scorers.latency_scorer.LatencyScorer]): Scorers for latency evaluation.
        quality_scorers (List[~simuleval.scorers.latency_scorer.QualityScorer]): Scorers for quality evaluation.
        output: output directory

    Evaluator related command line arguments:

    .. argparse::
        :ref: simuleval.options.add_evaluator_args
        :passparser:
        :prog:
    """

    def __init__(
        self,
        dataloader: Optional[GenericDataloader],
        quality_scorers: Dict[str, QualityScorer],
        latency_scorers: Dict[str, LatencyScorer],
        args: Namespace,
    ) -> None:
        self.dataloader = dataloader
        self.quality_scorers = quality_scorers
        self.latency_scorers = latency_scorers
        self.instances = {}

        self.args = args
        self.output = Path(args.output) if args.output else None
        self.score_only = args.score_only
        self.no_scoring = args.no_scoring
        self.source_segment_size = getattr(args, "source_segment_size", 1)
        self.source_type = getattr(args, "source_type", None)
        self.target_type = getattr(args, "target_type", None)

        self.target_spm_model = None
        if args.eval_latency_unit == "spm":
            assert args.eval_latency_spm_model
            assert IS_IMPORT_SPM
            self.target_spm_model = sentencepiece.SentencePieceProcessor(model_file=args.eval_latency_spm_model)

        if self.source_type is None and self.target_type is None and self.output is not None:
            with open(self.output / "config.yaml") as f:
                configs = yaml.safe_load(f)
                self.source_type = configs["source_type"]
                self.target_type = configs["target_type"]

        assert self.source_type
        assert self.target_type

        if self.output is not None:
            os.makedirs(self.output, exist_ok=True)
            with open(self.output / "config.yaml", "w") as f:
                yaml.dump(
                    {"source_type": self.source_type, "target_type": self.source_type},
                    f,
                    default_flow_style=False,
                )

        self.instance_class = INSTANCE_TYPE_DICT[f"{self.source_type}-{self.target_type}"]
        self.start_index = getattr(args, "start_index", 0)
        self.end_index = getattr(args, "end_index", -1)

        if not self.score_only:
            if self.output:
                if self.args.continue_unfinished and (self.output / "instances.log").exists():
                    with open(self.output / "instances.log", "r") as f:
                        line = None
                        for line in f:  # noqa
                            pass
                        if line is not None:
                            last_info = json.loads(line.strip())
                            self.start_index = last_info["index"] + 1
                else:
                    self.output.mkdir(exist_ok=True, parents=True)
                    open(self.output / "instances.log", "w").close()
            if self.end_index < 0:
                assert self.dataloader is not None
                self.end_index = len(self.dataloader)

        self.build_instances()

        iterable = self.instances.values()
        if isinstance(self.dataloader, IterableDataloader):
            iterable = self.dataloader

        if not self.args.no_progress_bar and not self.score_only:
            self.iterator = tqdm(
                iterable,
                initial=self.start_index,
            )
        else:
            self.iterator = iterable
        self.start_t = time.time()

    def write_log(self, instance):
        if self.output is not None:
            with open(self.output / "instances.log", "a") as f:
                f.write(json.dumps(instance.summarize()) + "\n")

    def build_instances(self):
        if self.score_only:
            self.build_instances_from_log()
        else:
            self.build_instances_from_dataloader()

    def build_instances_from_log(self):
        self.instances = {}
        if self.output is not None:
            with open(self.output / "instances.log", "r") as f:
                for line in f:
                    instance = LogInstance(line.strip(), self.args.eval_latency_unit)
                    index = instance.index - self.start_index
                    self.instances[index] = instance
                    self.instances[index].set_target_spm_model(self.target_spm_model)

    def build_instances_from_dataloader(self):
        if isinstance(self.dataloader, IterableDataloader):
            return

        for i in self.get_indices():
            self.instances[i] = self.instance_class(i, self.dataloader, self.args)
            self.instances[i].set_target_spm_model(self.target_spm_model)

    def __len__(self) -> int:
        return self.end_index - self.start_index

    def get_indices(self) -> Generator:
        if self.end_index < 0:
            self.end_index = max(self.instances.keys()) + 1

        if self.start_index > self.end_index:
            return []

        for index in range(self.start_index, self.end_index):
            yield index

    @property
    def quality(self) -> Dict[str, float]:
        return {name: scorer(self.instances) for name, scorer in self.quality_scorers.items()}

    @property
    def latency(self) -> Dict[str, float]:
        return {name: scorer(self.instances) for name, scorer in self.latency_scorers.items()}

    @property
    def results(self):
        scores = {**self.quality, **self.latency}
        new_scores = {}
        for name, value in scores.items():
            if isinstance(value, numbers.Number):
                value = round(value, 3)
            new_scores[name] = [value]

        df = pandas.DataFrame(new_scores)
        return df

    def dump_results(self) -> None:
        results = self.results
        finish_t = time.time()
        results["TIME"] = finish_t - self.start_t

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", type=str, default="")
        parser.add_argument("--target", type=str, default="")
        parser.add_argument("--background", type=str, default=None)
        parser.add_argument("--use_api", action="store_true")
        parser.add_argument("--k", type=int, default=4)
        parser.add_argument("--dir", type=str, default=None)
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--model_id", type=str, default=None)
        parser.add_argument("--min_read_time", type=str, default=None)
        parser.add_argument("--min_lag_words", type=int, default=None)
        parser.add_argument("--start-index", type=int, default=None)
        parser.add_argument("--end-index", type=int, default=None)
        parser.add_argument("--source-segment-size", type=int, default=None)
        parser.add_argument("--use_asr_api", action="store_true")
        parser.add_argument("--asr_model_size", type=str, default=None)
        parser.add_argument("--prompt_id", type=int, default=0)
        parser.add_argument("--func_wrds", type=str, default="[]")
        parser.add_argument("--priming", action="store_true")
        custom_args, _ = parser.parse_known_args()

        if custom_args.asr_model_size is not None:
            audio_file_list = custom_args.source.replace(".txt", "")
            results["RTF1"] = get_RTF(
                results["TIME"],
                audio_file_list,
                custom_args.start_index,
                custom_args.end_index,
            )
            results["WER"] = get_real_wer(
                custom_args.output,
                custom_args.source,
                custom_args.start_index,
                custom_args.end_index,
            )
            results["min_read_time"] = custom_args.min_read_time
            results["min_lag_words"] = custom_args.min_lag_words
            results["src_seg_sz"] = custom_args.source_segment_size
            results["use_asr_api"] = custom_args.use_asr_api
            results["asr_model_size"] = custom_args.asr_model_size
        else:
            results["WER"] = None
        results["k"] = custom_args.k
        results["dir"] = custom_args.dir
        results["output"] = custom_args.output
        results["use_api"] = custom_args.use_api
        results["model_id"] = custom_args.model_id
        results["end_index"] = custom_args.end_index
        results["prompt_id"] = custom_args.prompt_id
        results["background"] = custom_args.background
        results["func_wrds"] = custom_args.func_wrds
        results["priming"] = custom_args.priming

        if self.output:
            results.to_csv(self.output / "scores.tsv", sep="\t", index=False)
            results.to_json(self.output / "scores.json", index=False, orient="records")

        logger.info("Results:")
        print(results.to_string(index=False))

    def dump_metrics(self) -> None:
        metrics = pandas.DataFrame([ins.metrics for ins in self.instances.values()])
        metrics = metrics.round(3)
        if self.output:
            metrics.to_csv(self.output / "metrics.tsv", sep="\t", index=False)

    def is_finished(self, instance) -> bool:
        if hasattr(instance, "source_finished_reading"):
            return instance.source_finished_reading
        return instance.finish_prediction

    def __call__(self, system):
        with open(self.output / "instances.log", "a") if self.output else contextlib.nullcontext() as file:
            system.reset()
            for sample in self.iterator:  # "sample" is an input-output-(background) pair(triplet)
                instance = (
                    self.instance_class(self.dataloader.cur_index, self.dataloader, self.args)
                    if isinstance(self.dataloader, IterableDataloader)
                    else sample
                )
                # update background info for the sentence
                if self.args.background is not None:
                    system._set_background(sample.background)
                while not self.is_finished(instance):
                    input_segment = instance.send_source(self.source_segment_size)
                    output_segment = system.pushpop(input_segment)
                    instance.receive_prediction(output_segment)
                    if instance.finish_prediction:
                        # if instance.finish_prediction where set by the reader,
                        # source_finished_reading will be set as well. If it is
                        # set by any of the intermediate components, then we didn't
                        # end yet. We are going to clear the state and continue
                        # processing the rest of the input.
                        system.reset()

                if not self.score_only and self.output:
                    file.write(json.dumps(instance.summarize()) + "\n")

        if self.output:
            self.build_instances_from_log()
        if not self.no_scoring:
            self.dump_results()
            self.dump_metrics()

    @classmethod
    def from_args(cls, args):
        if not args.score_only:
            dataloader = build_dataloader(args)
        else:
            dataloader = None

        latency_scorers = {}
        use_ref_len = not args.no_use_ref_len
        for name in args.latency_metrics:
            latency_scorers[name] = get_scorer_class("latency", name).from_args(args)
            if args.computation_aware:
                latency_scorers[name + "_CA"] = get_scorer_class("latency", name)(
                    computation_aware=True, use_ref_len=use_ref_len
                )

        quality_scorers = {}
        for name in args.quality_metrics:
            quality_scorers[name] = get_scorer_class("quality", name).from_args(args)

        return cls(dataloader, quality_scorers, latency_scorers, args)
