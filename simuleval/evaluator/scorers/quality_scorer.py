import logging
import sacrebleu
from pathlib import Path
from typing import List, Dict

QUALITY_SCORERS_DICT = {}


def register_quality_scorer(name):
    def register(cls):
        QUALITY_SCORERS_DICT[name] = cls
        return cls

    return register


class QualityScorer:
    def __init__(self) -> None:
        pass

    def __call__(self, references, translations) -> float:
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        pass


def add_sacrebleu_args(parser):
    parser.add_argument("--sacrebleu-tokenizer", type=str, default="13a", help="Tokenizer in sacrebleu")


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
            return sacrebleu.corpus_bleu(
                [ins.prediction for ins in instances.values()],
                [[ins.reference for ins in instances.values()]],
                tokenize=self.tokenizer,
            ).score
        except:
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

    def __init__(self, tokenizer: str = "13a") -> None:
        super().__init__()
        self.logger = logging.getLogger("simuleval.scorer.asr_bleu")
        self.tokenizer = "13a"  # todo make it configurable

    def __call__(self, instances) -> float:
        return sacrebleu.corpus_bleu(
            self.asr_transcribe(instances),
            [[ins.reference for ins in instances.values()]],
            tokenize=self.tokenizer,
        ).score

    def asr_transcribe(self, instances):
        self.logger.warn("Beta feature: Evaluating speech output")
        try:
            from ust_common.evaluation import prepare_w2v_audio_finetuning_data
            from ust_common.evaluation import fairseq_w2v_ctc_infer
        except:
            self.logger.warn("Please install ust_common.")
            return ["" for _ in range(len(self))]

        from simuleval.utils.fairseq import load_fairseq_manifest

        wav_dir = Path(instances[0].prediction).absolute().parent
        root_dir = wav_dir.parent
        # TODO make it configurable
        if not (root_dir / "asr_prep_data").exists():
            prepare_w2v_audio_finetuning_data(
                wav_dir,
                root_dir / "asr_prep_data",
                output_subset_name="eval",
                waveform_filename_pattern="*",
            )
        if not (root_dir / "asr_out").exists():
            fairseq_w2v_ctc_infer(
                root_dir / "asr_prep_data",
                "/checkpoint/annl/s2st/eval/asr/model/wav2vec2/wav2vec_vox_960h_pl.pt",
                "eval",
                root_dir / "asr_out",
            )

        translations_w_id = load_fairseq_manifest(
            root_dir / "asr_out" / "eval_asr_predictions.tsv"
        )
        translations_w_id = sorted(
            translations_w_id, key=lambda x: int(x["id"].split("_")[-1])
        )

        translation_list = []
        for idx, item in enumerate(translations_w_id):
            with open(wav_dir / f"{idx}_pred.txt", "w") as f:
                f.write(item["transcription"].lower() + "\n")
            translation_list.append(item["transcription"].lower())

        return translation_list

    @staticmethod
    def add_args(parser):
        add_sacrebleu_args(parser)

    @classmethod
    def from_args(cls, args):
        return cls(args.sacrebleu_tokenizer)
