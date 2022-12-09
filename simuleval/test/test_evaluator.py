# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile


def test_score_only():
    with tempfile.TemporaryDirectory() as tmpdirname:
        p = subprocess.run(
            "simuleval"
            " --agent examples/quick_start/first_agent.py"
            " --source examples/quick_start/source.txt"
            " --target examples/quick_start/target.txt"
            f" --output {tmpdirname} && simuleval --score-only --output {tmpdirname}",
            shell=True,
        )
    returncode = p.returncode
    assert returncode == 0
