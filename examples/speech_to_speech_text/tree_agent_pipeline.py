from typing import Optional
from simuleval.agents import TreeAgentPipeline
from examples.speech_to_speech.english_counter_agent import (
    EnglishSpeechCounter as EnglishSpeechToSpeech,
)
from examples.speech_to_text.english_counter_agent import (
    EnglishSpeechCounter as EnglishSpeechToText,
)
from simuleval.agents.actions import WriteAction
from simuleval.agents.agent import SpeechToTextAgent
from simuleval.agents.states import AgentStates


class EnglishWait2SpeechToText(EnglishSpeechToText):
    def __init__(self, args):
        super().__init__(args)
        args.wait_seconds = 2

    @staticmethod
    def add_args(parser):
        pass


class DotSpeechToText(SpeechToTextAgent):
    def policy(self, states: Optional[AgentStates] = None):
        return WriteAction(
            content=".",
            finished=states.source_finished,
        )


class EnglishWait2SpeechToSpeech(EnglishSpeechToSpeech):
    def __init__(self, args):
        super().__init__(args)
        args.wait_seconds = 2

    @staticmethod
    def add_args(parser):
        pass


class DummyTreePipeline(TreeAgentPipeline):
    # pipeline is a dict, used to instantiate agents in from_args
    pipeline = {
        EnglishSpeechToSpeech: [EnglishWait2SpeechToText, EnglishWait2SpeechToSpeech],
        EnglishWait2SpeechToSpeech: [],
        EnglishWait2SpeechToText: [],
    }


class TemplateTreeAgentPipeline(TreeAgentPipeline):
    def __init__(self, args) -> None:
        speech_speech = self.pipeline[0](args)
        speech_speech_wait2 = self.pipeline[1](args)
        speech_text = self.pipeline[2](args)

        module_dict = {
            speech_speech: [speech_text, speech_speech_wait2],
            speech_speech_wait2: [],
            speech_text: [],
        }

        super().__init__(module_dict, args)

    @classmethod
    def from_args(cls, args):
        return cls(args)


class InstantiatedTreeAgentPipeline(TemplateTreeAgentPipeline):
    pipeline = [
        EnglishSpeechToSpeech,
        EnglishWait2SpeechToSpeech,
        EnglishWait2SpeechToText,
    ]


class AnotherInstantiatedTreeAgentPipeline(TemplateTreeAgentPipeline):
    pipeline = [
        EnglishSpeechToSpeech,
        EnglishWait2SpeechToSpeech,
        DotSpeechToText,  # swap the speech_text module in the pipeline
    ]
