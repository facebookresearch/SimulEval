# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Dict, Generator, List
import sacrebleu
from .instance import SpeechToTextInstance, TextToTextInstance
import os
import sys
import logging
from statistics import mean
from pathlib import Path
from simuleval.utils.common import load_fairseq_manifest, get_fairseq_manifest_path
from simuleval.data.dataloader import GenericDataloader

logger = logging.getLogger("simuleval.sentence_level_scorer")


class SentenceLevelScorer(object):
    def __init__(self, dataloader: GenericDataloader, args: Namespace) -> None:
        self.args = args
        self.eval_latency_unit = args.eval_latency_unit
        self.sacrebleu_tokenizer = args.sacrebleu_tokenizer
        self.no_space = args.no_space
        self.dataloader = dataloader
        self.instances = {}
        self.start_index = args.start_index
        self.end_index = args.end_index
        if self.end_index < 0:
            self.end_index = len(self.dataloader)

        if args.source_type == "speech":
            self.instance_class = SpeechToTextInstance
        else:
            self.instance_class = TextToTextInstance

        self.reset()

    def __len__(self) -> int:
        return self.end_index - self.start_index

    def get_indices(self) -> Generator:
        for index in range(self.start_index, self.end_index):
            yield index

    def get_info(self) -> Dict[str, int]:
        return {"num_sentences": len(self)}

    def send_source(self, instance_id: int, segment_size: int) -> Dict:
        dict_to_return = self.instances[instance_id].send_source(
            segment_size=segment_size
        )
        dict_to_return["instance_id"] = instance_id
        return dict_to_return

    def reset(self) -> None:
        if len(self.instances) > 0:
            logger.warning("Resetting scorer")

        for i in self.get_indices():
            self.instances[i] = self.instance_class(i, self.dataloader, self.args)

    def get_translation_list(self) -> List[str]:
        not_finish_write_id = [
            i for i in self.get_indices() if not self.instances[i].finish_prediction
        ]
        empty_hypo_id = [
            str(i) for i in self.get_indices() if len(self.instances[i].prediction) == 0
        ]

        if len(not_finish_write_id) > 0:
            logger.warn(
                "Warning: these hypothesis don't have EOS in predictions",
                file=sys.stderr,
            )
            logger.warn(", ".join((str(x) for x in not_finish_write_id)), file=sys.stderr)
            for idx in not_finish_write_id:
                self.instances[idx].sentence_level_eval()

        if len(empty_hypo_id) > 0:
            logger.warn("Warning: these hypothesis are empty", file=sys.stderr)
            logger.warn(", ".join(empty_hypo_id), file=sys.stderr)

        translations = [self.instances[i].prediction for i in self.get_indices()]

        return translations

    def get_reference_list(self) -> List[str]:
        return [self.dataloader.target_list[i] for i in self.get_indices()]

    def get_quality_score(self) -> Dict[str, float]:

        try:
            bleu_score = sacrebleu.corpus_bleu(
                self.get_translation_list(),
                [self.get_reference_list()],
                tokenize=self.sacrebleu_tokenizer,
            ).score
        except Exception as e:
            logger.error(e, file=sys.stderr)
            bleu_score = 0

        return {"BLEU": bleu_score}

    def get_latency_score(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for metric in ["AL", "AP", "DAL"]:
            results[metric] = mean(
                [seg.metrics["latency"][metric] for seg in self.instances.values()]
            )
            if "latency_ca" in self.instances[0].metrics:
                results[metric + "_CA"] = mean(
                    [
                        seg.metrics["latency_ca"][metric]
                        for seg in self.instances.values()
                    ]
                )

            if "latency_text_w_time" in self.instances[0].metrics:
                results[metric + " (Time in ms)"] = mean(
                    [
                        seg.metrics["latency_text_w_time"][metric]
                        for seg in self.instances.values()
                    ]
                )

        return results

    def score(self):
        return {
            "Quality": self.get_quality_score(),
            "Latency": self.get_latency_score(),
        }
