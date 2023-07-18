from simuleval.agents import AgentPipeline
from .silero_vad import SileroVADAgent
from ..speech_to_text.english_counter_agent import EnglishSpeechCounter


class EnglishCounterAgentPipeline(AgentPipeline):
    pipeline = [
        SileroVADAgent,
        EnglishSpeechCounter,
    ]