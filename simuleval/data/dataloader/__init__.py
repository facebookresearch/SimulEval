import logging
from typing import Any
from simuleval import SUPPORTED_SOURCE_MEDIUM, SUPPORTED_TARGET_MEDIUM
from simuleval.utils.fairseq import use_fairseq
from .t2t_dataloader import TextToTextDataloader
from .s2t_dataloader import SpeechToTextDataloader
from .fairseq_s2t_dataloader import (
    FairseqSpeechToTextDataloader,
    FairseqSpeechToSpeechDataloader,
)

DATALOADER_DICT = {
    "text-text": TextToTextDataloader,
    "speech-text": SpeechToTextDataloader,
    "fairseq_speech-text": FairseqSpeechToTextDataloader,
    "fairseq_speech-speech": FairseqSpeechToSpeechDataloader,
}

logger = logging.getLogger("simuleval.dataloader")


def build_dataloader(args) -> Any:
    assert args.source_type in SUPPORTED_SOURCE_MEDIUM
    assert args.target_type in SUPPORTED_TARGET_MEDIUM
    logger.info(f"Evaluating from {args.source_type} to {args.target_type}.")
    if not use_fairseq(args):
        assert args.source
        return DATALOADER_DICT[f"{args.source_type}-{args.target_type}"].from_files(
            args.source, args.target
        )
    else:
        logger.info(f"Using Fairseq S2T manifest.")
        assert args.source_type == "speech"
        return DATALOADER_DICT[
            f"fairseq_{args.source_type}-{args.target_type}"
        ].from_args(args)


from .dataloader import GenericDataloader
from .s2t_dataloader import SpeechToTextDataloader
from .t2t_dataloader import TextToTextDataloader
from .fairseq_s2t_dataloader import FairseqSpeechToTextDataloader
