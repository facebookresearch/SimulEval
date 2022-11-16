import os
import sys
import simuleval

sys.path.append(os.path.join(simuleval.__path__[0], "..", "examples"))

from simuleval.agents import SpeechToTextAgent
from fairseq_speech.generic_agent import FairseqSimulSpeechInputAgent


class FairseqSimulS2TAgent(FairseqSimulSpeechInputAgent, SpeechToTextAgent):
    pass
