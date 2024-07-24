# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from pathlib import Path
import simuleval.cli as cli

ROOT_PATH = Path(__file__).parents[2]


def test_visualize(root_path=ROOT_PATH):
    args_path = Path.joinpath(root_path, "examples", "speech_to_text")
    os.chdir(args_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = [
            "--agent",
            os.path.join(root_path, "examples", "speech_to_text", "whisper_waitk.py"),
            "--source-segment-size",
            "500",
            "--waitk-lagging",
            "3",
            "--source",
            os.path.join(root_path, "examples", "speech_to_text", "source.txt"),
            "--target",
            os.path.join(
                root_path, "examples", "speech_to_text", "reference/transcript.txt"
            ),
            "--output",
            "output",
            "--quality-metrics",
            "WER",
            "--visualize",
        ]
        cli.main()

        visual_folder_path = os.path.join("output", "visual")
        source_path = os.path.join(
            root_path, "examples", "speech_to_text", "source.txt"
        )
        source_length = 0

        with open(source_path, "r") as f:
            source_length = len(f.readlines())
        images = list(Path(visual_folder_path).glob("*.png"))
        assert len(images) == source_length


def test_visualize_score_only(root_path=ROOT_PATH):
    args_path = Path.joinpath(root_path, "examples", "speech_to_text")
    os.chdir(args_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = ["--score-only", "--output", "output", "--visualize"]
        cli.main()

        visual_folder_path = os.path.join("output", "visual")
        source_path = os.path.join(
            root_path, "examples", "speech_to_text", "source.txt"
        )
        source_length = 0

        with open(source_path, "r") as f:
            source_length = len(f.readlines())
        images = list(Path(visual_folder_path).glob("*.png"))
        assert len(images) == source_length
