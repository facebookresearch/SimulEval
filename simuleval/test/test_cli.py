# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import time


def test_simuleval():
    result = subprocess.Popen(
        [
            "simuleval",
            "--agent", "examples/dummy/dummy_waitk_text_agent.py",
            "--source", "examples/data/src.txt",
            "--target", "examples/data/tgt.txt",
            ]
    )
    _ = result.communicate()[0]
    returncode = result.returncode
    assert returncode == 0


def test_simuleval_server_client():
    server_proc = subprocess.Popen(
        [
            "simuleval",
            "--server-only",
            "--source", "examples/data/src.txt",
            "--target", "examples/data/tgt.txt",
            "--data-type", "text"
        ]
    )

    time.sleep(3)

    client_proc = subprocess.Popen(
        [
            "simuleval",
            "--clienonly",
            "--agent", "examples/dummy/dummy_waitk_text_agent.py",
            "--waitk", "1",
            "--num-process", "1",
            ]
    )
    _ = client_proc.communicate()[0]
    returncode = client_proc.returncode
    assert returncode == 0

    server_proc.terminate()
