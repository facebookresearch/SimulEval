# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
from pathlib import Path
import urllib.request
from argparse import Namespace

import pytest

import simuleval.cli as cli
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import TextSegment
import logging

logger = logging.getLogger()


ROOT_PATH = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT_PATH))  # may be needed for import from `examples`

from examples.quick_start.spm_detokenizer_agent import (
    SentencePieceModelDetokenizerAgent,
)


def test_agent(root_path=ROOT_PATH):
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = [
            "--user-dir",
            os.path.join(root_path, "examples"),
            "--agent-class",
            "examples.quick_start.first_agent.DummyWaitkTextAgent",
            "--source",
            os.path.join(root_path, "examples", "quick_start", "source.txt"),
            "--target",
            os.path.join(root_path, "examples", "quick_start", "target.txt"),
            "--output",
            tmpdirname,
        ]
        cli.main()


def test_statelss_agent(root_path=ROOT_PATH):
    class DummyWaitkTextAgent(TextToTextAgent):
        waitk = 0
        vocab = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

        def policy(self, states=None):
            if states is None:
                states = self.states

            lagging = len(states.source) - len(states.target)

            if lagging >= self.waitk or states.source_finished:
                prediction = self.vocab[len(states.source)]

                return WriteAction(prediction, finished=(lagging <= 1))
            else:
                return ReadAction()

    args = None
    agent_stateless = DummyWaitkTextAgent.from_args(args)
    agent_state = agent_stateless.build_states()
    agent_stateful = DummyWaitkTextAgent.from_args(args)

    for _ in range(10):
        segment = TextSegment(0, "A")
        output_1 = agent_stateless.pushpop(segment, agent_state)
        output_2 = agent_stateful.pushpop(segment)
        assert output_1.content == output_2.content


@pytest.mark.parametrize("detokenize_only", [True, False])
def test_spm_detokenizer_agent(detokenize_only):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tokenizer_file = f"{tmpdirname}/tokenizer.model"
        tokenizer_url = "https://huggingface.co/facebook/seamless-m4t-large/resolve/main/tokenizer.model"
        urllib.request.urlretrieve(tokenizer_url, tokenizer_file)

        args = Namespace()
        args.sentencepiece_model = tokenizer_file
        args.detokenize_only = detokenize_only

        output = []
        delays = []
        agent = SentencePieceModelDetokenizerAgent.from_args(args)
        agent_state = agent.build_states()
        segments = [
            TextSegment(0, "▁Let ' s"),
            TextSegment(1, "▁do ▁it ▁with"),
            TextSegment(2, "out ▁hesitation .", finished=True),
        ]
        for i, segment in enumerate(segments):
            output_segment = agent.pushpop(segment, agent_state)
            if not output_segment.is_empty:
                output.append(output_segment.content)
                delays += [i] * len(output_segment.content.split())
        if detokenize_only:
            assert output == ["Let's", "do it with", "out hesitation."]
            assert delays == [0, 1, 1, 1, 2, 2]
        else:
            assert output == ["Let's do it", "without hesitation."]
            assert delays == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("detokenize_only", [True, False])
def test_spm_detokenizer_agent_pipeline(detokenize_only, root_path=ROOT_PATH):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tokenizer_file = f"{tmpdirname}/tokenizer.model"
        tokenizer_url = "https://huggingface.co/facebook/seamless-m4t-large/resolve/main/tokenizer.model"
        urllib.request.urlretrieve(tokenizer_url, tokenizer_file)

        cli.sys.argv[1:] = [
            "--user-dir",
            os.path.join(root_path, "examples"),
            "--agent-class",
            "examples.quick_start.spm_detokenizer_agent.DummyPipeline",
            "--source",
            os.path.join(root_path, "examples", "quick_start", "spm_source.txt"),
            "--target",
            os.path.join(root_path, "examples", "quick_start", "spm_target.txt"),
            "--output",
            tmpdirname,
            "--segment-k",
            "3",
            "--sentencepiece-model",
            tokenizer_file,
        ]
        if detokenize_only:
            cli.sys.argv.append("--detokenize-only")
        cli.main()
