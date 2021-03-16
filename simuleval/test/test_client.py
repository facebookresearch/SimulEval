# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import tempfile
from multiprocessing import Process
from types import SimpleNamespace
import simuleval
from simuleval.cli import evaluate
from simuleval.online import start_server
from simuleval.utils.functional import find_free_port
from simuleval.utils.agent_finder import check_data_type, find_agent_cls

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def test_client():
    args = SimpleNamespace()
    args.hostname = "localhost"
    args.port = find_free_port()
    args.data_type = "text"
    args.source = os.path.join(CURRENT_PATH, "data", "text", "src.txt")
    args.target = os.path.join(CURRENT_PATH, "data", "text", "tgt.txt")
    args.agent = os.path.join(CURRENT_PATH, "..", "..",
                              "examples", "dummy", "dummy_waitk_text_agent.py")
    args.output = None
    args.num_processes = 1
    args.waitk = 1
    args.eval_latency_unit = "word"
    args.sacrebleu_tokenizer = "13a"
    args.no_space = False
    _, agent_cls = find_agent_cls(args)
    check_data_type(args, agent_cls)

    def eval(num_process):
        server_process = Process(target=start_server, args=(args, ))
        server_process.start()
        time.sleep(1)
        client = simuleval.online.start_client(args)
        with tempfile.TemporaryDirectory() as tmpdirname:
            args.output = tmpdirname
            args.num_processes = num_process
            evaluate(args, client, server_process)

    eval(1)
    eval(2)
