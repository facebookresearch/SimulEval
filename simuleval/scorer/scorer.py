# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sacrebleu
from .instance import SpeechToTextInstance, TextToTextInstance
import os
import sys
import logging
from statistics import mean
from pathlib import Path
from simuleval.utils.common import load_fairseq_manifest, get_fairseq_manifest_path

logger = logging.getLogger("simuleval.sentence_level_scorer")


class SentenceLevelScorer(object):
    def __init__(self, dataloader, args):
        self.args = args
        self.eval_latency_unit = args.eval_latency_unit
        self.sacrebleu_tokenizer = args.sacrebleu_tokenizer
        self.no_space = args.no_space
        self.dataloader = dataloader
        self.instances = {}
        if args.source_type == "speech":
            self.instance_class = SpeechToTextInstance
        else:
            self.instance_class = TextToTextInstance

        self.reset()

    def __len__(self):
        return len(self.dataloader)

    def get_info(self):
        return {"num_sentences": len(self)}

    def send_src(self, instance_id, segment_size):
        dict_to_return = self.instances[instance_id].send_src(segment_size=segment_size)
        dict_to_return["instance_id"] = instance_id
        return dict_to_return

    def recv_hyp(self, instance_id, list_of_tokens):
        self.instances[instance_id].recv_hypo(list_of_tokens, self.eval_latency_unit)

    def reset(self):
        if len(self.instances) > 0:
            logger.warning("Resetting scorer")

        option_dict = {"eval_latency_unit": self.eval_latency_unit}

        for i in range(len(self)):
            self.instances[i] = self.instance_class(i, self.dataloader, self.args)

    def gather_translation(self):
        not_finish_write_id = [
            i for i in range(len(self)) if not self.instances[i].finish_hypo
        ]
        empty_hypo_id = [
            str(i)
            for i in range(len(self))
            if len(self.instances[i].prediction(no_space=self.no_space)) == 0
        ]

        if len(not_finish_write_id) > 0:
            print(
                "Warning: these hypothesis don't have EOS in predictions",
                file=sys.stderr,
            )
            print(", ".join((str(x) for x in not_finish_write_id)), file=sys.stderr)
            for idx in not_finish_write_id:
                self.instances[idx].sentence_level_eval()

        if len(empty_hypo_id) > 0:
            print("Warning: these hypothesis are empty", file=sys.stderr)
            print(", ".join(empty_hypo_id), file=sys.stderr)

        translations = [
            self.instances[i].prediction(eos=False, no_space=self.no_space)
            for i in range(len(self))
        ]

        return translations

    def get_quality_score(self):

        translations = self.gather_translation()

        try:
            bleu_score = sacrebleu.corpus_bleu(
                translations, [self.data["tgt"]], tokenize=self.sacrebleu_tokenizer
            ).score
        except Exception as e:
            print(e, file=sys.stderr)
            bleu_score = 0

        return {"BLEU": bleu_score}

    def get_latency_score(self):
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
