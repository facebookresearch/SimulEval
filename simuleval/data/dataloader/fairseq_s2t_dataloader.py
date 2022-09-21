from __future__ import annotations
import sys
import logging
from argparse import Namespace
from .s2t_dataloader import SpeechToTextDataloader
from pathlib import Path
from typing import Callable, List, Union, Tuple
import soundfile

from simuleval.utils.common import load_fairseq_manifest, get_fairseq_manifest_path
from simuleval.utils.fairseq import check_fairseq_args
from fairseq.data.audio.data_cfg import get_config_from_yaml

logger = logging.getLogger("simuleval.fairseq_s2t_dataloader")


class FairseqSpeechToTextDataloader(SpeechToTextDataloader):
    def preprocess_source(self, source: Union[Path, str]) -> Tuple[List, int]:
        samples, sample_rate = soundfile.read(source, dtype="float32")
        samples = samples.tolist()
        return source, sample_rate

    @classmethod
    def from_args(cls, args: Namespace) -> FairseqSpeechToTextDataloader:
        check_fairseq_args(args)
        if args.fairseq_manifest:
            manifest_path = args.fairseq_manifest
        else:
            manifest_path = get_fairseq_manifest_path(
                args.fairseq_data, args.fairseq_gen_subset
            )

        logger.info(f"Manifest: {manifest_path.as_posix()}")
        datalist = load_fairseq_manifest(manifest_path)

        logger.info(f"Config: {args.fairseq_config}")
        config = get_config_from_yaml(Path(args.fairseq_data) / args.fairseq_config)

        fairseq_audio_root = Path(config["audio_root"])

        return cls(
            [fairseq_audio_root / x["audio"] for x in datalist],
            [x["tgt_text"] for x in datalist],
        )
