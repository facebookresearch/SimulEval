from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Union
from .dataloader import GenericDataloader


class TextToTextDataloader(GenericDataloader):
    def __init__(self) -> None:
        super().__init__()
        self.source_splitter = lambda x: x.split()
        self.target_splitter = lambda x: x

    def set_source_splitter(self, function: Callable) -> None:
        # TODO, make is configurable
        self.splitter = function

    def preprocess_source(self, source: str) -> List:
        return self.source_splitter(source)

    def preprocess_target(self, source: str) -> List:
        return self.target_splitter(source)

    @classmethod
    def from_files(
        cls, source: Union[Path, str], target: Union[Path, str]
    ) -> TextToTextDataloader:
        with open(source) as f:
            source_list = f.readlines()
        with open(target) as f:
            target_list = f.readlines()
        dataloader = cls(source_list, target_list)
        return dataloader
