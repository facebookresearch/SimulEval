# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import requests
from multiprocessing import Process
from simuleval.online import start_server
from simuleval.utils.functional import find_free_port
from types import SimpleNamespace

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def test_server():
    args = SimpleNamespace()
    args.hostname = "localhost"
    args.port = find_free_port()
    args.data_type = "text"
    args.source = os.path.join(CURRENT_PATH, "data", "text", "src.txt")
    args.target = os.path.join(CURRENT_PATH, "data", "text", "tgt.txt")
    args.output = None
    args.eval_latency_unit = "word"
    args.sacrebleu_tokenizer = "13a"
    args.no_space = False
    server_process = Process(target=start_server, args=(args, ))
    server_process.start()
    time.sleep(2)

    from requests.exceptions import ConnectionError
    success = 0
    try:
        request = requests.get(f'http://{args.hostname}:{args.port}')
    except ConnectionError:
        print('Web site does not exist')
    else:
        success = 1
    assert success == 1

    data = request.json()

    assert data["num_sentences"] == 10
    assert data["data_type"] == "text"

    server_process.kill()
