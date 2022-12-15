# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile


def test_score_only(binary="simuleval", root_path=""):
    with tempfile.TemporaryDirectory() as tmpdirname:
        p = subprocess.run(
            f"{binary}"
            f" --agent {os.path.join(root_path, 'examples/quick_start/first_agent.py')}"
            f" --source {os.path.join(root_path, 'examples/quick_start/source.txt')}"
            f" --target {os.path.join(root_path, 'examples/quick_start/target.txt')}"
            f" --output {tmpdirname} && simuleval --score-only --output {tmpdirname}",
            shell=True,
        )
    returncode = p.returncode
    assert returncode == 0
