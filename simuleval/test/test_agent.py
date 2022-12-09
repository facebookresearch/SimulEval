# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import tempfile


def test_cmd():
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = subprocess.Popen(
            [
                "simuleval",
                "--agent", "examples/quick_start/first_agent.py",
                "--source", "examples/quick_start/source.txt",
                "--target", "examples/quick_start/target.txt",
                "--output", tmpdirname
                ]
        )
    _ = result.communicate()[0]
    returncode = result.returncode
    assert returncode == 0
