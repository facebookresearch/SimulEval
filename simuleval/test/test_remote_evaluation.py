# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import tempfile
import subprocess
from simuleval.utils.functional import find_free_port


def test_remote_eval(binary="simuleval", root_path=""):
    port = find_free_port()
    p_1 = subprocess.Popen(
        [
            binary,
            "--standalone",
            "--remote-port",
            str(port),
            "--agent",
            os.path.join(root_path, "examples/quick_start/first_agent.py"),
        ]
    )
    time.sleep(5)

    assert p_1.returncode != 1

    with tempfile.TemporaryDirectory() as tmpdirname:
        p_2 = subprocess.Popen(
            [
                binary,
                "--remote-eval",
                "--remote-port",
                str(port),
                "--source",
                os.path.join(root_path, "examples/quick_start/source.txt"),
                "--target",
                os.path.join(root_path, "examples/quick_start/target.txt"),
                "--dataloader",
                "text-to-text",
                "--output",
                tmpdirname,
            ]
        )
    _ = p_2.communicate()[0]

    assert p_2.returncode == 0
    p_1.kill()
    p_2.kill()
