# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas
import numbers
from argparse import Namespace
from typing import Dict, Generator, List
from .scorers import get_scorer_class
from .scorers.latency_scorer import LatencyScorer
from .scorers.quality_scorer import QualityScorer

from .instance import INSTANCE_TYPE_DICT, LogInstance
import yaml
import logging
import json
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from statistics import mean
from pathlib import Path
from simuleval.data.dataloader import GenericDataloader, build_dataloader


logger = logging.getLogger("simuleval.sentence_level_evaluator")


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
        dataloader: GenericDataloader,
        quality_scorers: List[QualityScorer],
        latency_scorers: List[LatencyScorer],
        args: Namespace,
    ) -> None:
        self.dataloader = dataloader
        self.quality_scorers = quality_scorers
        self.latency_scorers = latency_scorers
        self.instances = {}

        self.args = args
        self.output = Path(args.output) if args.output else None
        self.score_only = args.score_only
        self.source_segment_size = getattr(args, "source_segment_size", 1)
        self.source_type = getattr(args, "source_type", None)
        self.target_type = getattr(args, "target_type", None)

        if self.source_type is None and self.target_type is None:
            with open(self.output / "config.yaml") as f:
                configs = yaml.safe_load(f)
                self.source_type = configs["source_type"]
                self.target_type = configs["target_type"]

        self.instance_class = INSTANCE_TYPE_DICT[
            f"{self.source_type}-{self.target_type}"
        ]
        self.start_index = args.start_index
        self.end_index = args.end_index

        if not self.score_only:
            if self.output:
                self.output.mkdir(exist_ok=True, parents=True)
                open(self.output / "instances.log", "w").close()
            if self.end_index < 0:
                self.end_index = len(self.dataloader)

        if self.args.no_progress_bar:
            self.maybe_tqdm = tqdm
        else:
            self.maybe_tqdm = lambda x: x

        self.build_instances()

    def write_log(self, instance):
        with open(self.output / "instances.log", "a") as f:
            f.write(json.dumps(instance.summarize()) + "\n")

    def build_instances(self):
        if self.score_only:
            self.build_instances_from_log()
        else:
            self.build_instances_from_dataloader()

    def build_instances_from_log(self):
        self.instances = {}
        with open(self.output / "instances.log", "r") as f:
            for line in f:
                instance = LogInstance(line.strip())
                self.instances[instance.index] = instance

    def build_instances_from_dataloader(self):
        for i in self.get_indices():
            self.instances[i] = self.instance_class(i, self.dataloader, self.args)

    def __len__(self) -> int:
        return self.end_index - self.start_index

    def get_indices(self) -> Generator:
        if self.end_index < 0:
            self.end_index = max(self.instances.keys()) + 1
        for index in range(self.start_index, self.end_index):
            yield index

    def reset(self) -> None:
        if len(self.instances) > 0:
            logger.warning("Resetting scorer")
        self.build_instances()

    @property
    def quality(self) -> Dict[str, float]:
        return {
            name: scorer(self.instances)
            for name, scorer in self.quality_scorers.items()
        }

    @property
    def latency(self) -> Dict[str, Dict[str, float]]:
        return {
            name: scorer(self.instances)
            for name, scorer in self.latency_scorers.items()
        }

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

    def __call__(self, system):
        with logging_redirect_tqdm(loggers=[logger]):
            for instance in self.maybe_tqdm(self.instances.values()):
                system.reset()
                while not instance.finish_prediction:
                    input_segment = instance.send_source(self.source_segment_size)
                    output_segment = system.pushpop(input_segment)
                    instance.receive_prediction(output_segment)
                if self.output:
                    self.write_log(instance)

        results = self.results
        if self.output:
            with open(self.output / "scores", "w") as f:
                f.write(results)

        logger.info(f"Results:")
        print(results.to_string(index=False))

    @classmethod
    def from_args(cls, args):
        if not args.score_only:
            dataloader = build_dataloader(args)
        else:
            dataloader = None

        latency_scorers = {}
        for name in args.latency_metrics:
            latency_scorers[name] = get_scorer_class("latency", name)()
            if args.computation_aware:
                latency_scorers[name + "_CA"] = get_scorer_class("latency", name)(
                    computation_aware=True
                )

        quality_scorers = {}
        for name in args.quality_metrics:
            quality_scorers[name] = get_scorer_class("quality", name)()

        return cls(dataloader, quality_scorers, latency_scorers, args)
