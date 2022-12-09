# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import tempfile
import time
from simuleval.utils.functional import find_free_port


def test_remote_eval():
    port = find_free_port()
    p_1 = subprocess.Popen(
        [
            "simuleval",
                "--standalone",
                "--remote-port",
                str(port),
                "--agent",
                "examples/quick_start/first_agent.py",
            ]
    )
    time.sleep(5)

    assert p_1.returncode != 1

    with tempfile.TemporaryDirectory() as tmpdirname:
        p_2 = subprocess.Popen(
            [
                "simuleval",
                "--remote-eval",
                "--remote-port", str(port),
                "--source", "examples/quick_start/source.txt",
                "--target", "examples/quick_start/target.txt",
                "--dataloader", "text-to-text",
                "--output", tmpdirname
            ]
        )
    _ = p_2.communicate()[0]

    assert p_2.returncode == 0
    p_1.kill()
    p_2.kill()