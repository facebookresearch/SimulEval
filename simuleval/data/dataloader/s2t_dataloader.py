# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from pathlib import Path
from typing import List, Union, Optional
from .dataloader import GenericDataloader
from simuleval.data.dataloader import register_dataloader
from argparse import Namespace
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp as youtube_dl
    from pydub import AudioSegment
except ImportError:
    yt_dlp = AudioSegment = None

try:
    import soundfile

    IS_IMPORT_SOUNDFILE = True
except Exception:
    IS_IMPORT_SOUNDFILE = False


def download_youtube_video(url):
    def get_video_id(url):
        url_data = urlparse(url)
        query = parse_qs(url_data.query)
        video = query.get("v", [])
        if video:
            return video[0]
        else:
            raise Exception("unrecoginzed url format.")

    id = get_video_id(url)
    name = f"{id}.wav"

    if not Path(name).exists():
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": id,  # name the file "downloaded_video" with original extension
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    sound = AudioSegment.from_wav(name)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export(name, format="wav")
    return name


def load_list_from_file(file_path: Union[Path, str]) -> List[str]:
    with open(file_path) as f:
        return [line.strip() for line in f]


@register_dataloader("speech-to-text")
class SpeechToTextDataloader(GenericDataloader):
    def __init__(
        self,
        source_list: List[str],
        target_list: List[str],
        tgt_lang_list: Optional[List[str]] = None,
    ) -> None:
        super().__init__(source_list, target_list, tgt_lang_list)

    def preprocess_source(self, source: Union[Path, str]) -> List[float]:
        assert IS_IMPORT_SOUNDFILE, "Please make sure soundfile is properly installed."
        samples, _ = soundfile.read(source, dtype="float32")
        samples = samples.tolist()
        return samples

    def preprocess_target(self, target: str) -> str:
        return target

    def get_source_audio_info(self, index: int) -> soundfile._SoundFileInfo:
        return soundfile.info(self.get_source_audio_path(index))

    def get_source_audio_path(self, index: int):
        return self.source_list[index]

    @classmethod
    def from_files(
        cls,
        source: Union[Path, str],
        target: Union[Path, str],
        tgt_lang: Union[Path, str],
    ) -> SpeechToTextDataloader:
        source_list = load_list_from_file(source)
        target_list = load_list_from_file(target)
        tgt_lang_list = []
        if tgt_lang is not None:
            tgt_lang_list = load_list_from_file(tgt_lang)
        dataloader = cls(source_list, target_list, tgt_lang_list)
        return dataloader

    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "speech"
        args.target_type = "text"
        return cls.from_files(args.source, args.target, args.tgt_lang)


@register_dataloader("speech-to-speech")
class SpeechToSpeechDataloader(SpeechToTextDataloader):
    @classmethod
    def from_files(
        cls,
        source: Union[Path, str],
        target: Union[Path, str],
        tgt_lang: Union[Path, str, None] = None,
    ) -> SpeechToSpeechDataloader:
        source_list = load_list_from_file(source)
        target_list = load_list_from_file(target)
        tgt_lang_list = []
        if tgt_lang is not None:
            tgt_lang_list = load_list_from_file(tgt_lang)
        dataloader = cls(source_list, target_list, tgt_lang_list)
        return dataloader

    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "speech"
        args.target_type = "speech"
        return cls.from_files(args.source, args.target, args.tgt_lang)


@register_dataloader("youtube-to-text")
class YoutubeToTextDataloader(SpeechToTextDataloader):
    @classmethod
    def from_youtube(
        cls, source: Union[Path, str], target: Union[Path, str]
    ) -> YoutubeToTextDataloader:
        assert AudioSegment is not None
        assert youtube_dl is not None
        source_list = [download_youtube_video(source)]
        target_list = [target]
        dataloader = cls(source_list, target_list)
        return dataloader

    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "youtube"
        args.target_type = "text"
        return cls.from_youtube(args.source, args.target)


@register_dataloader("youtube-to-speech")
class YoutubeToSpeechDataloader(YoutubeToTextDataloader):
    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "youtube"
        args.target_type = "speech"
        return cls.from_youtube(args.source, args.target)
