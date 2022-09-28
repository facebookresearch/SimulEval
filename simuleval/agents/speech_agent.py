# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .agent import Agent


class SpeechToTextAgent(Agent):
    source_type = "speech"
    target_type = "text"

    def __init__(self, args) -> None:
        super().__init__(args)
        self.source_segment_size = 10  # in ms


class SpeechToSpeechAgent(Agent):
    source_type = "speech"
    target_type = "speech"
