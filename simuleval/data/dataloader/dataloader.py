# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Union

SUPPORTED_MEDIUM = ["text", "speech"]
SUPPORTED_SOURCE_MEDIUM = ["youtube", "text", "speech"]
SUPPORTED_TARGET_MEDIUM = ["text", "speech"]
DATALOADER_DICT = {}


def register_dataloader(name):
    def register(cls):
        DATALOADER_DICT[name] = cls
        return cls

    return register


def register_dataloader_class(name, cls):
    DATALOADER_DICT[name] = cls


class GenericDataloader:
    """
    Load source and target data

    .. argparse::
        :ref: simuleval.options.add_data_args
        :passparser:
        :prog:

    """

    def __init__(
        self,
        source_list: List[str],
        target_list: Union[List[str], List[None]],
        tgt_lang_list: Optional[List[str]] = None,
    ) -> None:
        self.source_list = source_list
        self.target_list = target_list
        self.tgt_lang_list = tgt_lang_list
        assert len(self.source_list) == len(self.target_list)

    def __len__(self):
        return len(self.source_list)

    def get_source(self, index: int) -> Any:
        return self.preprocess_source(self.source_list[index])

    def get_target(self, index: int) -> Any:
        return self.preprocess_target(self.target_list[index])

    def get_tgt_lang(self, index: int) -> Optional[str]:
        if getattr(self, "tgt_lang_list", None) is None or index >= len(
            self.tgt_lang_list
        ):
            return None
        else:
            return self.tgt_lang_list[index]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "source": self.get_source(index),
            "target": self.get_target(index),
            "tgt_lang": self.get_tgt_lang(index),
        }

    def preprocess_source(self, source: Any) -> Any:
        raise NotImplementedError

    def preprocess_target(self, target: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(args.source, args.target)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--source",
            type=str,
            help="Source file.",
        )
        parser.add_argument(
            "--target",
            type=str,
            help="Target file.",
        )
        parser.add_argument(
            "--source-type",
            type=str,
            choices=SUPPORTED_SOURCE_MEDIUM,
            help="Source Data type to evaluate.",
        )
        parser.add_argument(
            "--target-type",
            type=str,
            choices=SUPPORTED_TARGET_MEDIUM,
            help="Data type to evaluate.",
        )
        parser.add_argument(
            "--source-segment-size",
            type=int,
            default=1,
            help="Source segment size, For text the unit is # token, for speech is ms",
        )
        parser.add_argument(
            "--tgt-lang",
            type=str,
            default=None,
            help="Target language",
        )


class IterableDataloader:
    cur_index: int

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __next__(self):
        ...
