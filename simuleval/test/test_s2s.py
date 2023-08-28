# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from pathlib import Path
from typing import Optional
from simuleval.agents.states import AgentStates

import simuleval.cli as cli
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment

ROOT_PATH = Path(__file__).parents[2]


def test_s2s(root_path=ROOT_PATH):
    args_path = Path.joinpath(root_path, "examples", "speech_to_speech")
    os.chdir(args_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = [
            "--agent",
            os.path.join(
                root_path, "examples", "speech_to_speech", "english_alternate_agent.py"
            ),
            "--user-dir",
            os.path.join(root_path, "examples"),
            "--agent-class",
            "agents.EnglishAlternateAgent",
            "--source-segment-size",
            "1000",
            "--source",
            os.path.join(root_path, "examples", "speech_to_speech", "source.txt"),
            "--target",
            os.path.join(root_path, "examples", "speech_to_speech", "reference/en.txt"),
            "--output",
            tmpdirname,
            "--tgt-lang",
            os.path.join(
                root_path, "examples", "speech_to_speech", "reference/tgt_lang.txt"
            ),
        ]
        cli.main()


def test_stateless_agent(root_path=ROOT_PATH):
    class EnglishAlternateAgent(SpeechToSpeechAgent):
        waitk = 0
        wait_seconds = 3
        vocab = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

        def policy(self, states: Optional[AgentStates] = None):
            if states is None:
                states = states

            length_in_seconds = round(len(states.source) / states.source_sample_rate)
            if (
                not self.states.source_finished
                and length_in_seconds < self.wait_seconds
            ):
                return ReadAction()

            if length_in_seconds % 2 == 0:
                samples, fs = self.tts_model.synthesize(
                    f"{8 - length_in_seconds} even even"
                )
            else:
                samples, fs = self.tts_model.synthesize(
                    f"{8 - length_in_seconds} odd odd"
                )

            prediction = f"{length_in_seconds} second"

            return WriteAction(
                SpeechSegment(
                    content=samples,
                    sample_rate=fs,
                    finished=self.states.source_finished,
                ),
                content=prediction,
                finished=self.states.source_finished,
            )

    args = None
    agent_stateless = EnglishAlternateAgent.from_args(args)
    agent_state = agent_stateless.build_states()
    agent_stateful = EnglishAlternateAgent.from_args(args)

    for _ in range(10):
        segment = SpeechSegment(0, "A")
        output_1 = agent_stateless.pushpop(segment, agent_state)
        output_2 = agent_stateful.pushpop(segment)
        assert output_1.content == output_2.content
