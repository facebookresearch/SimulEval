from simuleval.agents import AgentPipeline
from examples.demo.silero_vad import SileroVADAgent
from examples.speech_to_text.counter_in_tgt_lang import CounterInTargetLanguage


class CounterInTargetLanguageAgentPipeline(AgentPipeline):
    pipeline = [
        SileroVADAgent,
        CounterInTargetLanguage,
    ]
