import os
import sys
import torch
import simuleval
from typing import Dict

sys.path.append(os.path.join(simuleval.__path__[0], "..", "examples"))

from fairseq_speech.generic_agent import FairseqSimulS2TAgent
from fairseq_speech.utils import test_time_waitk_agent


@test_time_waitk_agent
class FairseqTestWaitKS2TAgent(FairseqSimulS2TAgent):
    """
    Test-time Wait-K agent for speech-to-text translation.
    This agent load an offline model and run Wait-K policy
    """
    pass
