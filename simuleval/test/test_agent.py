# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile


def test_agent(binary="simuleval", root_path=""):
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = subprocess.Popen(
            [
                binary,
                "--agent",
                os.path.join(root_path, "examples/quick_start/first_agent.py"),
                "--source",
                os.path.join(root_path, "examples/quick_start/source.txt"),
                "--target",
                os.path.join(root_path, "examples/quick_start/target.txt"),
                "--output",
                tmpdirname,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    _ = result.communicate()[0]
    returncode = result.returncode
    assert returncode == 0
