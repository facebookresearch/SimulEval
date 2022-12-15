import logging
from typing import Any
from argparse import ArgumentParser
from .dataloader import (
    GenericDataloader,
    register_dataloader,
    register_dataloader_class,
    SUPPORTED_MEDIUM,
    SUPPORTED_SOURCE_MEDIUM,
    SUPPORTED_TARGET_MEDIUM,
    DATALOADER_DICT,
)
from .t2t_dataloader import TextToTextDataloader
from .s2t_dataloader import SpeechToTextDataloader


logger = logging.getLogger("simuleval.dataloader")


def build_dataloader(args) -> Any:
    dataloader_key = getattr(args, "dataloader", None)
    if dataloader_key is not None:
        assert dataloader_key in DATALOADER_DICT, f"{dataloader_key} is not defined"
        logger.info(f"Evaluating from dataloader {dataloader_key}.")
        return DATALOADER_DICT[dataloader_key].from_args(args)

    assert args.source_type in SUPPORTED_SOURCE_MEDIUM
    assert args.target_type in SUPPORTED_TARGET_MEDIUM

    logger.info(f"Evaluating from {args.source_type} to {args.target_type}.")
    return DATALOADER_DICT[f"{args.source_type}-to-{args.target_type}"].from_args(args)
