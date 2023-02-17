# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sacrebleu
from pathlib import Path
from typing import Dict
from sacrebleu.metrics.bleu import BLEU
import subprocess

QUALITY_SCORERS_DICT = {}


def register_quality_scorer(name):
    def register(cls):
        QUALITY_SCORERS_DICT[name] = cls
        return cls

    return register


class QualityScorer:
    def __init__(self) -> None:
        pass

    def __call__(self, instances: Dict) -> float:
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        pass


def add_sacrebleu_args(parser):
    parser.add_argument(
        "--sacrebleu-tokenizer",
        type=str,
        default=sacrebleu.metrics.METRICS["BLEU"].TOKENIZER_DEFAULT,
        choices=sacrebleu.metrics.METRICS["BLEU"].TOKENIZERS,
        help="Tokenizer in sacrebleu",
    )


@register_quality_scorer("BLEU")
class SacreBLEUScorer(QualityScorer):
    """
    SacreBLEU Scorer

    Usage:
        :code:`--quality-metrics BLEU`

    Additional command line arguments:

    .. argparse::
        :ref: simuleval.evaluator.scorers.quality_scorer.add_sacrebleu_args
        :passparser:
        :prog:
    """

    def __init__(self, tokenizer: str = "13a") -> None:
        super().__init__()
        self.logger = logging.getLogger("simuleval.scorer.bleu")
        self.tokenizer = tokenizer

    def __call__(self, instances: Dict) -> float:
        try:
            return (
                BLEU(tokenize=self.tokenizer)
                .corpus_score(
                    [ins.prediction for ins in instances.values()],
                    [[ins.reference for ins in instances.values()]],
                )
                .score
            )
        except Exception as e:
            self.logger.error(str(e))
            return 0

    @staticmethod
    def add_args(parser):
        add_sacrebleu_args(parser)

    @classmethod
    def from_args(cls, args):
        return cls(args.sacrebleu_tokenizer)


@register_quality_scorer("ASR_BLEU")
class ASRSacreBLEUScorer(QualityScorer):
    """
    ASR + SacreBLEU Scorer (BETA version)

    Usage:
        :code:`--quality-metrics ASR_BLEU`

    Additional command line arguments:

    .. argparse::
        :ref: simuleval.evaluator.scorers.quality_scorer.add_sacrebleu_args
        :passparser:
        :prog:
    """

    def __init__(self, tokenizer: str = "13a", target_lang: str = "en") -> None:
        super().__init__()
        self.logger = logging.getLogger("simuleval.scorer.asr_bleu")
        self.tokenizer = tokenizer
        self.target_lang = target_lang

    def __call__(self, instances: Dict) -> float:
        transcripts = self.asr_transcribe(instances)
        score = (
            BLEU(tokenize=self.tokenizer)
            .corpus_score(
                transcripts,
                [[ins.reference for ins in instances.values()]],
            )
            .score
        )
        return score

    def asr_transcribe(self, instances):
        self.logger.warn("Beta feature: Evaluating speech output. Faieseq is required.")
        try:
            import fairseq

            fairseq_path = Path(fairseq.__path__[0]).parent  # type: ignore
        except Exception:
            self.logger.warn("Please install fairseq.")
            return ["" for _ in instances.keys()]

        wav_dir = Path(instances[0].prediction).absolute().parent
        root_dir = wav_dir.parent
        transcripts_path = root_dir / "asr_transcripts.txt"
        asr_cmd_bash_path = root_dir / "asr_cmd.bash"

        # This is a dummy reference. The bleu score will be compute separately.
        reference_path = root_dir / "instances.log"

        fairseq_asr_bleu_cmd = "\n".join(
            [
                f"cd {fairseq_path.as_posix()}/examples/speech_to_speech/asr_bleu/",
                " ".join(
                    [
                        "python compute_asr_bleu.py",
                        f"--reference_path {reference_path.as_posix()}",
                        f"--lang {self.target_lang}",
                        f"--audio_dirpath {wav_dir.as_posix()}",
                        "--reference_format txt",
                        f"--transcripts_path {(root_dir / 'asr_transcripts.txt').as_posix()}",
                    ]
                ),
            ]
        )
        with open(asr_cmd_bash_path, "w") as f:
            f.write(fairseq_asr_bleu_cmd + "\n")

        process = subprocess.Popen(["bash", asr_cmd_bash_path], stdout=subprocess.PIPE)
        _, stderr = process.communicate()

        if process.returncode != 0:
            self.logger.error("ASR on target speech failed:")
            self.logger.error(str(stderr) + "\n")
            return ["" for _ in instances.keys()]

        with open(transcripts_path, "r") as f:
            transcripts = [line.strip() for line in f]

        for idx, item in enumerate(transcripts):
            with open(wav_dir / f"{idx}_pred.txt", "w") as f:
                f.write(item.lower() + "\n")

        return transcripts

    @staticmethod
    def add_args(parser):
        add_sacrebleu_args(parser)
        parser.add_argument(
            "--target-speech-lang",
            type=str,
            default="en",
            help="The language of target speech",
        )

    @classmethod
    def from_args(cls, args):
        return cls(args.sacrebleu_tokenizer, args.target_speech_lang)
