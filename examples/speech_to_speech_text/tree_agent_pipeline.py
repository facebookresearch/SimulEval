from typing import Optional
from simuleval.agents import TreeAgentPipeline
from examples.speech_to_speech.english_counter_agent import (
    EnglishSpeechCounter as EnglishSpeechToSpeech,
)
from examples.speech_to_speech.english_counter_agent import TTSModel
from examples.speech_to_text.english_counter_agent import (
    EnglishSpeechCounter as EnglishSpeechToText,
)
from simuleval.agents import TextToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents import AgentPipeline
from simuleval.agents.states import AgentStates
from simuleval.data.segments import SpeechSegment


class EnglishWait2SpeechToText(EnglishSpeechToText):
    def __init__(self, args):
        super().__init__(args)
        args.wait_seconds = 2

    @staticmethod
    def add_args(parser):
        pass


class EnglishWait2SpeechToSpeech(EnglishSpeechToSpeech):
    def __init__(self, args):
        super().__init__(args)
        args.wait_seconds = 2

    @staticmethod
    def add_args(parser):
        pass


class DummyTreePipeline(TreeAgentPipeline):
    pipeline = {
        EnglishSpeechToSpeech: [EnglishWait2SpeechToText, EnglishWait2SpeechToSpeech],
        EnglishWait2SpeechToSpeech: [],
        EnglishWait2SpeechToText: [],
    }
