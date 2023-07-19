from simuleval.agents import AgentPipeline
from examples.demo.silero_vad import SileroVADAgent
from examples.speech_to_text.english_counter_agent import EnglishSpeechCounter


class EnglishCounterAgentPipeline(AgentPipeline):
    pipeline = [
        SileroVADAgent,
        EnglishSpeechCounter,
    ]