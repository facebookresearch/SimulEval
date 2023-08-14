from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction

import whisper
import numpy


@entrypoint
class WaitkWhisper(SpeechToTextAgent):
    """
    The agent generate the number of seconds from an input audio.
    """

    def __init__(self, args):
        super().__init__(args)
        self.waitk_lagging = args.waitk_lagging
        self.source_segment_size = args.source_segment_size
        self.source_language = args.source_language
        self.continuous_write = args.continuous_write
        self.model_size = args.model_size
        self.model = whisper.load_model(self.model_size)
        self.task = args.task
        if self.task == "translate":
            assert (
                self.source_language != "en"
            ), "source language must be different from en for translation task"

    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk-lagging", default=1, type=int)
        parser.add_argument("--source-language", default="en", type=str)
        parser.add_argument(
            "--continuous-write",
            default=1,
            type=int,
            help="Max number of words to write at each step",
        )
        parser.add_argument("--model-size", default="tiny", type=str)
        parser.add_argument(
            "--task",
            default="transcribe",
            type=str,
            choices=["transcribe", "translate"],
        )

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if not states.source_finished:
            if (
                length_in_seconds * 1000 / self.source_segment_size
            ) < self.waitk_lagging:
                return ReadAction()

        previous_translation = " ".join(states.target)
        # We use the previous translation as a prefix.
        options = whisper.DecodingOptions(
            prefix=previous_translation,
            language=self.source_language,
            without_timestamps=True,
            fp16=False,
        )

        # We encode the whole audio to get the full transcription each time a new audio chunk is received.
        audio = whisper.pad_or_trim(numpy.array(states.source).astype("float32"))
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        output = self.model.decode(mel, options)
        prediction = output.text.split()

        if not states.source_finished and self.continuous_write > 0:
            prediction = prediction[: self.continuous_write]

        return WriteAction(
            content=" ".join(prediction),
            finished=states.source_finished,
        )
