from __future__ import annotations
from pathlib import Path
from typing import List, Union
from .dataloader import GenericDataloader
from simuleval.data.dataloader import register_dataloader
from argparse import Namespace

try:
    import soundfile

    IS_IMPORT_SOUNDFILE = True
except Exception:
    IS_IMPORT_SOUNDFILE = False


@register_dataloader("speech-to-text")
class SpeechToTextDataloader(GenericDataloader):
    def preprocess_source(self, source: Union[Path, str]) -> List[float]:
        assert IS_IMPORT_SOUNDFILE, "Please make sure soundfile is properly installed."
        samples, _ = soundfile.read(source, dtype="float32")
        samples = samples.tolist()
        return samples

    def preprocess_target(self, target: str) -> List:
        return target

    def get_source_audio_info(self, index: int) -> float:
        return soundfile.info(self.get_source_audio_path(index))

    def get_source_audio_path(self, index: int):
        return self.source_list[index]

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

    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "text"
        args.target_type = "text"
        return cls.from_files(args.source, args.target)
