from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction


@entrypoint
class EnglishSpeechCounter(SpeechToTextAgent):
    """
    The agent generate the number of seconds from an input audio.
    """

    def __init__(self, args):
        super().__init__(args)
        self.wait_seconds = args.wait_seconds

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-seconds", default=1, type=int)

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states
        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = round(
                len(states.source) / states.source_sample_rate
            )
        if not states.source_finished and length_in_seconds < self.wait_seconds:
            return ReadAction()

        prediction = f"{length_in_seconds} second"

        return WriteAction(
            content=prediction,
            finished=states.source_finished,
        )
