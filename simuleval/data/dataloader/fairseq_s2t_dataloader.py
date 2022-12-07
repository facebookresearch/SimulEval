from __future__ import annotations
import io
import sys
import logging
from argparse import Namespace
from .s2t_dataloader import SpeechToTextDataloader
from pathlib import Path
from typing import Callable, List, Union, Tuple
import soundfile

from simuleval.utils.fairseq import get_audio_file_path, get_fairseq_manifest_path

logger = logging.getLogger("simuleval.fairseq_s2t_dataloader")

try:
    from fairseq.data.audio.speech_to_text_dataset import (
        SpeechToTextDataset,
    )
    from fairseq.tasks.speech_to_text import SpeechToTextTask
except:
    pass


class FairseqSpeechToTextDataloader(SpeechToTextDataloader):
    """
    Load speech-to-text data in fairseq-s2t format.

    .. argparse::
        :ref: simuleval.options.add_fairseq_data_args
        :passparser:
        :prog:

    .. note::
        fairseq has to be installed to use this feature.

    """

    def __init__(self, fairseq_s2t_dataset: SpeechToTextDataset) -> None:
        self.fairseq_s2t_dataset = fairseq_s2t_dataset

    def __len__(self):
        return len(self.fairseq_s2t_dataset)

    def get_source(self, index: int) -> List:
        return self.fairseq_s2t_dataset[index].source.tolist()

    def get_target(self, index: int) -> str:
        return self.fairseq_s2t_dataset.txt_compressor.decompress(
            self.fairseq_s2t_dataset.tgt_texts[index]
        )

    def get_source_audio_info(self, index: int) -> float:
        return soundfile.info(get_audio_file_path(self.get_source_audio_path(index)))

    def get_source_audio_path(self, index: int) -> float:
        return self.fairseq_s2t_dataset.audio_paths[index]

    @classmethod
    def from_args(cls, args: Namespace) -> FairseqSpeechToTextDataloader:
        # check_fairseq_args(args)
        if args.fairseq_manifest:
            manifest_path = Path(args.fairseq_manifest)
            args.fairseq_data = manifest_path.parent.as_posix()
            args.fairseq_gen_subset = manifest_path.name.replace(".tsv", "")

        else:
            manifest_path = get_fairseq_manifest_path(
                args.fairseq_data, args.fairseq_gen_subset
            )

        logger.info(f"Manifest: {manifest_path.as_posix()}")

        logger.info(f"Config: {args.fairseq_config}")

        task_args = Namespace(
            data=args.fairseq_data, config_yaml=args.fairseq_config, seed=1
        )
        task = SpeechToTextTask.setup_task(task_args)
        task.load_dataset(args.fairseq_gen_subset)
        dataset = task.datasets[args.fairseq_gen_subset]
        return cls(dataset)


class FairseqSpeechToSpeechDataloader(FairseqSpeechToTextDataloader):
    # For now we still use S2T dataset for evaluation
    pass
