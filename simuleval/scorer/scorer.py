# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from distutils import log
import imp
from typing import Dict, Generator, List, Optional, Union
import sacrebleu
from .instance import INSTANCE_TYPE_DICT
import os
import sys
import logging
import json
from statistics import mean
from pathlib import Path
from simuleval.utils.common import load_fairseq_manifest, get_fairseq_manifest_path
from simuleval.data.dataloader import GenericDataloader

logger = logging.getLogger("simuleval.sentence_level_scorer")


class SentenceLevelScorer(object):
    def __init__(
        self,
        dataloader: Optional[GenericDataloader],
        args: Namespace,
        reset: bool = True,
    ) -> None:
        self.dataloader = dataloader
        self.instances = {}
        self.sacrebleu_tokenizer = "13a"

        if args is not None:
            self.start_index = args.start_index
            self.end_index = args.end_index
            if self.end_index < 0:
                self.end_index = len(self.dataloader)
            self.args = args
            # self.eval_latency_unit = args.eval_latency_unit
            # self.sacrebleu_tokenizer = args.sacrebleu_tokenizer
            # self.no_space = args.no_space
            self.output = Path(args.output)

            self.instance_class = INSTANCE_TYPE_DICT[
                f"{args.source_type}-{args.target_type}"
            ]

        if reset:
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
        raise NotImplementedError

    def get_reference_list(self) -> List[str]:
        return [self.instances[i].reference for i in self.get_indices()]

    def get_quality_score(self) -> Dict[str, float]:
        from fairseq import pdb

        pdb.set_trace()
        bleu_score = sacrebleu.corpus_bleu(
            self.get_translation_list(),
            [self.get_reference_list()],
            tokenize=self.sacrebleu_tokenizer,
        ).score

        from fairseq import pdb

        pdb.set_trace()

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


class SentenceLevelTextScorer(SentenceLevelScorer):
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
            )
            logger.warn(", ".join((str(x) for x in not_finish_write_id)))
            for idx in not_finish_write_id:
                self.instances[idx].sentence_level_eval()

        if len(empty_hypo_id) > 0:
            logger.warn("Warning: these hypothesis are empty")
            logger.warn(", ".join(empty_hypo_id))

        translations = [self.instances[i].prediction for i in self.get_indices()]

        return translations

    @classmethod
    def from_logdir(cls, logdir: Union[Path, str]):
        logdir = Path(logdir)
        instances = {}

        instance_class = INSTANCE_TYPE_DICT["text", "text"]

        with open(logdir / "instances.log", "r") as f:
            for line in f:
                instance = instance_class.from_json(line.strip())
                instances[instance.index] = instance
        scorer = cls(None, None, False)
        scorer.start_index = 0
        scorer.end_index = len(instances.keys())
        scorer.instances = instances
        return scorer


class SentenceLevelSpeechScorer(SentenceLevelScorer):
    def __init__(
        self,
        dataloader: Optional[GenericDataloader],
        args: Namespace,
        reset: bool = True,
    ) -> None:
        super().__init__(dataloader, args, reset)
        self.pre_wavs_dir = self.output / "wavs"
        self.pre_wavs_dir.mkdir(exist_ok=True)

    def get_translation_list(self) -> List[str]:
        logger.warn("Beta feature: Evaluating speech output")
        try:
            from ust_common.evaluation import prepare_w2v_audio_finetuning_data
            from ust_common.evaluation import fairseq_w2v_ctc_infer
        except:
            logger.warn("Please install ust_common.")
            return ["" for _ in range(len(self))]

        prepare_w2v_audio_finetuning_data(
            self.pre_wavs_dir, self.output / "asr_prep_data", output_subset_name="eval"
        )
        fairseq_w2v_ctc_infer(
            self.output / "asr_prep_data",
            "/checkpoint/annl/s2st/eval/asr/model/wav2vec2/wav2vec_vox_960h_pl.pt",
            "eval",
            self.output / "asr_out",
        )

        translations_w_id = load_fairseq_manifest(
            self.output / "asr_out" / "eval_asr_predictions.tsv"
        )
        translations_w_id = sorted(
            translations_w_id, key=lambda x: int(x["id"].split("_")[-1])
        )
        return [x["transcription"].lower() for x in translations_w_id]

    def get_reference_list(self) -> List[str]:
        if len(self.instances.keys()) > 0:
            return super().get_reference_list()
        else:
            refer_list = []
            with open(self.output / "instances.log", "r") as f:
                for line in f:
                    refer_list.append(json.loads(line.strip())["reference"])
            return refer_list

    @classmethod
    def from_logdir(cls, logdir: Union[Path, str], target_type: str = "text"):
        logdir = Path(logdir)
        instances = {}
        args = Namespace(
            output=logdir,
            start_index=0,
            end_index=sum(1 for line in open(logdir / "instances.log")),
            source_type="speech",
            target_type="speech",
        )
        scorer = cls(None, args, False)
        scorer.instances = instances
        return scorer


def compute_score(logdir: Union[Path, str]):
    logdir = Path(logdir)
    if (logdir / "wavs").exists():
        scorer = SentenceLevelSpeechScorer.from_logdir(logdir)
    else:
        scorer = SentenceLevelTextScorer.from_logdir(logdir)

    print(json.dumps(scorer.score(), indent=4))
