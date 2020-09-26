# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . agent import Agent
from . text_agent import TextAgent
from . speech_agent import SpeechAgent

BUILDIN_AGENTS = [
    Agent,
    TextAgent,
    SpeechAgent
]
