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

    def policy(self):
        length_in_seconds = round(
            len(self.states.source) / self.states.source_sample_rate
        )
        if not self.states.source_finished and length_in_seconds < self.wait_seconds:
            return ReadAction()

        prediction = f"{length_in_seconds} second"

        return WriteAction(
            content=prediction,
            finished=self.states.source_finished,
        )
