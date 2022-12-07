from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Union, Optional
from .dataloader import GenericDataloader


class TextToTextDataloader(GenericDataloader):
    def __init__(
        self, source_list: List[str], target_list: List[Optional[str]]
    ) -> None:
        super().__init__(source_list, target_list)
        self.source_splitter = lambda x: x.split()
        self.target_splitter = lambda x: x

    def set_source_splitter(self, function: Callable) -> None:
        # TODO, make is configurable
        self.splitter = function

    def preprocess_source(self, source: str) -> List:
        return self.source_splitter(source)

    def preprocess_target(self, target: str) -> List:
        return self.target_splitter(target)

    @classmethod
    def from_files(
        cls, source: Union[Path, str], target: Optional[Union[Path, str]]
    ) -> TextToTextDataloader:
        assert source
        with open(source) as f:
            source_list = f.readlines()
        if target:
            with open(target) as f:
                target_list = f.readlines()
        else:
            target_list = [None for _ in source_list]
        dataloader = cls(source_list, target_list)
        return dataloader
