# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sacrebleu
from . instance import TextInstance, AudioInstance
import os
import sys
import logging
from statistics import mean

logger = logging.getLogger('simuleval.scorer')


class Scorer(object):
    def __init__(self, args):
        self.data = {
            "src": self.load_text_file(args.source),
            "tgt": self.load_text_file(args.target)
        }
        if args.source_timestamps is not None:
            self.data["src_timestamps"] = self.load_timestamp_file(
                args.source_timestamps
            )

        self.data_type = args.data_type
        self.eval_latency_unit = args.eval_latency_unit
        self.sacrebleu_tokenizer = args.sacrebleu_tokenizer
        self.no_space = args.no_space

        logger.info(f"Evaluating on {self.data_type}")
        logger.info(f"Source: {os.path.abspath(args.source)}")
        logger.info(f"Target: {os.path.abspath(args.target)}")
        logger.info(f"Number of sentences: {len(self)}")

        self.instances = {}
        if self.data_type == "text":
            self.instance_class = TextInstance
        elif self.data_type == "speech":
            self.instance_class = AudioInstance
        else:
            if self.data_type is None:
                logger.error(
                    "Please specify the data type (text or speech).\n"
                )
            else:
                logger.error(
                    f"{self.data_type} is not supported, "
                    "please choose from text or speech.\n"
                )
            sys.exit(1)

        self.reset()

    def get_info(self):
        return {
            "num_sentences": len(self),
            "data_type": self.data_type
        }

    def send_src(self, instance_id, segment_size):
        dict_to_return = self.instances[instance_id].send_src(
            segment_size=segment_size)
        dict_to_return["instance_id"] = instance_id
        return dict_to_return

    def recv_hyp(self, instance_id, list_of_tokens):
        self.instances[instance_id].recv_hypo(
            list_of_tokens,
            self.eval_latency_unit
        )

    def reset(self):
        if len(self.instances) > 0:
            logger.warning("Resetting scorer")

        option_dict = {
            "eval_latency_unit": self.eval_latency_unit
        }

        assert all(
            len(x) == len(self.data["src"]) for x in self.data.values()
        )

        total_length = len(self.data["src"])

        for i in range(total_length):
            data_dict = {
                key: self.data[key][i] for key in self.data.keys()
            }

            self.instances[i] = self.instance_class(
                i, data_dict, option_dict
            )

    def gather_translation(self):
        not_finish_write_id = [i for i in range(
            len(self)) if not self.instances[i].finish_hypo]
        empty_hypo_id = [str(i) for i in range(len(self)) if len(
            self.instances[i].prediction(no_space=self.no_space)) == 0]

        if len(not_finish_write_id) > 0:
            print(
                "Warning: these hypothesis don't have EOS in predictions",
                file=sys.stderr)
            print(
                ", ".join((str(x) for x in not_finish_write_id)),
                file=sys.stderr
            )
            for idx in not_finish_write_id:
                self.instances[idx].sentence_level_eval()

        if len(empty_hypo_id) > 0:
            print("Warning: these hypothesis are empty", file=sys.stderr)
            print(", ".join(empty_hypo_id), file=sys.stderr)

        translations = [self.instances[i].prediction(
            eos=False, no_space=self.no_space) for i in range(len(self))]

        return translations

    def get_quality_score(self):

        translations = self.gather_translation()

        try:
            bleu_score = sacrebleu.corpus_bleu(
                translations,
                [self.data["tgt"]],
                tokenize=self.sacrebleu_tokenizer
            ).score
        except Exception as e:
            print(e, file=sys.stderr)
            bleu_score = 0

        return {"BLEU": bleu_score}

    def get_latency_score(self):
        results = {}
        for metric in ["AL", "AP", "DAL"]:
            results[metric] = mean(
                [seg.metrics["latency"][metric]
                    for seg in self.instances.values()]
            )
            if "latency_ca" in self.instances[0].metrics:
                results[metric + "_CA"] = mean(
                    [seg.metrics["latency_ca"][metric]
                        for seg in self.instances.values()]
                )

            if "latency_text_w_time" in self.instances[0].metrics:
                results[metric + " (Time in ms)"] = mean(
                    [seg.metrics["latency_text_w_time"][metric]
                        for seg in self.instances.values()]
                )

        return results

    def score(self):
        return {
            'Quality': self.get_quality_score(),
            'Latency': self.get_latency_score(),
        }

    @staticmethod
    def load_text_file(file, split=False):
        with open(file) as f:
            if split:
                return [r.strip().split() for r in f]
            else:
                return [r.strip() for r in f]

    def __len__(self):
        return len(self.data["tgt"])

    @staticmethod
    def load_timestamp_file(timestamp_file, splitter=None):
        with open(timestamp_file) as f:
            return [
                [float(x) for x in line.strip().split(splitter)]
                for line in f
            ]


