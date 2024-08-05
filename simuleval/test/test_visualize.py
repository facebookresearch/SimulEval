# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from pathlib import Path
import simuleval.cli as cli
import shutil
import json

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
        shutil.rmtree("output")


def test_visualize_score_only(root_path=ROOT_PATH):
    args_path = Path.joinpath(root_path, "examples", "speech_to_text")
    os.chdir(args_path)

    # Create sample instances.log and config.yaml in output directory
    output = Path("output")
    output.mkdir()
    os.chdir(output)
    with open("config.yaml", "w") as config:
        config.write("source_type: speech\n")
        config.write("target_type: speech")
    with open("instances.log", "w") as instances:
        json.dump(
            {
                "index": 0,
                "prediction": "This is a synthesized audio file to test your simultaneous speech, to speak to speech, to speak translation system.",
                "delays": [
                    1500.0,
                    2000.0,
                    2500.0,
                    3000.0,
                    3500.0,
                    4000.0,
                    4500.0,
                    5000.0,
                    5500.0,
                    6000.0,
                    6500.0,
                    6849.886621315192,
                    6849.886621315192,
                    6849.886621315192,
                    6849.886621315192,
                    6849.886621315192,
                    6849.886621315192,
                    6849.886621315192,
                    6849.886621315192,
                ],
                "elapsed": [
                    1947.3278522491455,
                    2592.338800430298,
                    3256.8109035491943,
                    3900.0539779663086,
                    4561.986684799194,
                    5216.205835342407,
                    5874.6888637542725,
                    6526.906728744507,
                    7193.655729293823,
                    7852.792739868164,
                    8539.628744125366,
                    9043.279374916267,
                    9043.279374916267,
                    9043.279374916267,
                    9043.279374916267,
                    9043.279374916267,
                    9043.279374916267,
                    9043.279374916267,
                    9043.279374916267,
                ],
                "prediction_length": 19,
                "reference": "This is a synthesized audio file to test your simultaneous speech to text and to speech to speach translation system.",
                "source": [
                    "test.wav",
                    "samplerate: 22050 Hz",
                    "channels: 1",
                    "duration: 6.850 s",
                    "format: WAV (Microsoft) [WAV]",
                    "subtype: Signed 16 bit PCM [PCM_16]",
                ],
                "source_length": 6849.886621315192,
            },
            instances,
        )

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
        shutil.rmtree("output")
