from simuleval.agents import AgentPipeline
from .silero_vad import SileroVADAgent
from ..speech_to_text.english_counter_agent import EnglishSpeechCounter


class EnglishCounterAgentPipeline(AgentPipeline):
    pipeline = [
        SileroVADAgent,
        EnglishSpeechCounter,
    ]

    # @classmethod
    # def add_args(cls, parser: ArgumentParser):
    #     super().add_args(parser)
    #     parser.add_argument(
    #         "--checkpoint",
    #         type=str,
    #         required=True,
    #         help="Path to the model checkpoint.",
    #     )
    #     parser.add_argument(
    #         "--config-yaml", type=str, default=None, help="Path to config yaml"
    #     )

    # @classmethod
    # def from_args(cls, args):
    #     return cls(args)