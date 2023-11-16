# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from pathlib import Path

import simuleval.cli as cli
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment
from simuleval.evaluator.instance import LogInstance

ROOT_PATH = Path(__file__).parents[2]


def test_s2t(root_path=ROOT_PATH):
    args_path = Path.joinpath(root_path, "examples", "speech_to_text")
    os.chdir(args_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = [
            "--agent",
            os.path.join(
                root_path, "examples", "speech_to_text", "counter_in_tgt_lang_agent.py"
            ),
            "--user-dir",
            os.path.join(root_path, "examples"),
            "--agent-class",
            "agents.EnglishSpeechCounter",
            "--source-segment-size",
            "1000",
            "--source",
            os.path.join(root_path, "examples", "speech_to_text", "source.txt"),
            "--target",
            os.path.join(root_path, "examples", "speech_to_text", "reference/en.txt"),
            "--output",
            tmpdirname,
            "--tgt-lang",
            os.path.join(
                root_path, "examples", "speech_to_text", "reference/tgt_lang.txt"
            ),
        ]
        cli.main()

        with open(os.path.join(tmpdirname, "instances.log"), "r") as f:
            for line in f:
                instance = LogInstance(line.strip())
                assert (
                    instance.prediction
                    == "1 segundos 2 segundos 3 segundos 4 segundos 5 segundos 6 segundos 7 segundos"
                )


def test_statelss_agent(root_path=ROOT_PATH):
    class EnglishSpeechCounter(SpeechToTextAgent):
        wait_seconds = 3

        def policy(self, states=None):
            if states is None:
                states = self.states

            length_in_seconds = round(len(states.source) / states.source_sample_rate)
            if not states.source_finished and length_in_seconds < self.wait_seconds:
                return ReadAction()

            prediction = f"{length_in_seconds} second"

            return WriteAction(
                content=prediction,
                finished=states.source_finished,
            )

    args = None
    agent_stateless = EnglishSpeechCounter.from_args(args)
    agent_state = agent_stateless.build_states()
    agent_stateful = EnglishSpeechCounter.from_args(args)

    for _ in range(10):
        segment = SpeechSegment(0, "A")
        output_1 = agent_stateless.pushpop(segment, agent_state)
        output_2 = agent_stateful.pushpop(segment)
        assert output_1.content == output_2.content


def test_s2t_with_tgt_lang(root_path=ROOT_PATH):
    args_path = Path.joinpath(root_path, "examples", "speech_to_text")
    os.chdir(args_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = [
            "--agent",
            os.path.join(
                root_path, "examples", "speech_to_text", "counter_in_tgt_lang_agent.py"
            ),
            "--user-dir",
            os.path.join(root_path, "examples"),
            "--agent-class",
            "agents.CounterInTargetLanguage",
            "--source-segment-size",
            "1000",
            "--source",
            os.path.join(root_path, "examples", "speech_to_text", "source.txt"),
            "--target",
            os.path.join(root_path, "examples", "speech_to_text", "reference/en.txt"),
            "--output",
            tmpdirname,
            "--tgt-lang",
            os.path.join(
                root_path, "examples", "speech_to_text", "reference/tgt_lang.txt"
            ),
        ]
        cli.main()

        with open(os.path.join(tmpdirname, "instances.log"), "r") as f:
            for line in f:
                instance = LogInstance(line.strip())
                assert (
                    instance.prediction
                    == "1 segundos 2 segundos 3 segundos 4 segundos 5 segundos 6 segundos 7 segundos"
                )
