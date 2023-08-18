from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction


@entrypoint
class CounterInTargetLanguage(SpeechToTextAgent):
    """
    The agent generate the number of seconds from an input audio and output it in the target language text
    """

    def __init__(self, args):
        super().__init__(args)
        self.wait_seconds = args.wait_seconds
        self.tgt_lang = args.tgt_lang

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-seconds", default=1, type=int)
        parser.add_argument(
            "--tgt-lang", default="en", type=str, # choices=["en", "es", "de"]
        )

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states
        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = round(len(states.source) / states.source_sample_rate)
        if not states.source_finished and length_in_seconds < self.wait_seconds:
            return ReadAction()

        prediction = f"{length_in_seconds} "
        if self.tgt_lang == "en":
            prediction += "seconds"
        elif self.tgt_lang == "es":
            prediction += "segundos"
        elif self.tgt_lang == "de":
            prediction += "sekunden"
        else:
            prediction += "<unknown>"

        return WriteAction(
            content=prediction,
            finished=states.source_finished,
        )
