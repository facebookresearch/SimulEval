from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Union, Tuple
from .dataloader import GenericDataloader
import soundfile


class SpeechToTextDataloader(GenericDataloader):
    def preprocess_source(self, source: Union[Path, str]) -> Tuple[List, int]:
        samples, sample_rate = soundfile.read(source, dtype="float32")
        samples = samples.tolist()
        return source, sample_rate

    def preprocess_target(self, target: str) -> List:
        return target

    @classmethod
    def from_files(
        cls, source: Union[Path, str], target: Union[Path, str]
    ) -> SpeechToTextDataloader:
        with open(source) as f:
            source_list = f.readlines()
        with open(target) as f:
            target_list = f.readlines()
        dataloader = cls(source_list, target_list)
        return dataloader
