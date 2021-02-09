# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
import time
import logging
import json
from multiprocessing import Pool, Process, Manager
from functools import partial

import simuleval
from simuleval import options
from simuleval import READ_ACTION, WRITE_ACTION
from simuleval.online import start_client, start_server
from simuleval.utils.agent_finder import find_agent_cls
from simuleval.utils.functional import split_list_into_chunks


logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger('simuleval.cli')


class DataWriter(object):
    def __init__(self, args, q):
        self.queue = q
        self.output_dir = args.output
        if self.output_dir is None:
            logger.warning("No output directory")
            self.started = False
            self.proc = None
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except BaseException as be:
            logger.error(f'Failed to write results to {self.output_dir}.')
            logger.error(be)
            logger.error('Skip writing predictions')
            self.started = False
            return

        logger.info(f"Output dir: {self.output_dir}")
        path = os.path.join(self.output_dir, "instances.log")
        self.proc = Process(target=self.write_loop, args=(path, q))
        self.proc.start()
        self.started = True

    @staticmethod
    def write_loop(path, q):
        logger.info(f"Start data writer (process id {os.getpid()})")
        with open(path, 'w') as f:
            while True:
                try:
                    m = q.get()
                    f.write(json.dumps(m) + '\n')
                    f.flush()
                except EOFError:
                    break

    def write_scores(self, scores):
        if self.started:
            with open(os.path.join(self.output_dir, "scores"), 'w') as f:
                f.write(json.dumps(scores, indent=4))

    def kill(self):
        if self.proc is not None:
            self.proc.kill()
            logger.info("Close data writer")


def decode(args, client, result_queue, instance_ids):
    # Find agent and load related arguments
    agent_name, agent_cls = find_agent_cls(args)
    logger.info(
        f"Evaluating {agent_name} (process id {os.getpid()}) "
        f"on instances from {instance_ids[0]} to {instance_ids[-1]}"
    )

    parser = options.general_parser()
    options.add_agent_args(parser, agent_cls)
    args, _ = parser.parse_known_args()

    # Data type check
    info = client.corpus_info()
    data_type = info['data_type']
    if data_type != agent_cls.data_type:
        logger.error(
            f"Data type mismatch 'server.data_type {data_type}', '{args.agent_cls}.data_type: {args.agent_cls.data_type}'")
        sys.exit(1)

    # build agents
    agent = agent_cls(args)

    # Decode
    for instance_id in instance_ids:
        states = agent.build_states(args, client, instance_id)
        while not states.finish_hypo():
            action = agent.policy(states)
            if action == READ_ACTION:
                states.update_source()
            elif action == WRITE_ACTION:
                prediction = agent.predict(states)
                states.update_target(prediction)
            else:
                raise SystemExit(f"Undefined action name {action}")
        sent_info = client.get_scores(instance_id)
        result_queue.put(sent_info)
        logger.debug(f"Instance {instance_id} finished, results:\n{json.dumps(sent_info, indent=4)}")


def evaluate(args, client, server_process=None):
    info = client.corpus_info()
    num_sentences = info['num_sentences']
    indices = list(range(num_sentences))
    num_processes = args.num_processes
    manager = Manager()
    result_queue = manager.Queue()
    data_writer = DataWriter(args, result_queue)

    if num_processes > 1:
        if num_processes > num_sentences:
            logger.warn(
                f"Number of processes is larger than number sentences ({num_processes}, {num_sentences})."
                f"Will only use {num_sentences} processes"
            )
            num_processes = num_sentences

        # Multi process, split test set into num_processes pieces
        with Pool(args.num_processes) as p:
            p.map(
                partial(decode, args, client, result_queue),
                split_list_into_chunks(indices, num_processes),
            )
    else:
        decode(args, client, result_queue, indices)

    scores = client.get_scores()
    logger.info("Evaluation results:\n" + json.dumps(scores, indent=4))
    logger.info("Evaluation finished")

    data_writer.write_scores(scores)
    data_writer.kill()

    if server_process is not None:
        server_process.kill()
        logger.info("Shutdown server")


def main():
    parser = options.general_parser()
    options.add_server_args(parser)
    args, _ = parser.parse_known_args()

    if not args.server_only:
        _main(args.client_only)
    else:
        server()


def _main(client_only=False):
    parser = options.general_parser()
    options.add_server_args(parser)

    if not client_only:
        options.add_data_args(parser)

    args, _ = parser.parse_known_args()

    if not client_only:
        _, agent_cls = find_agent_cls(args)
        if args.data_type is None:
            args.data_type = agent_cls.data_type
        logging.getLogger("tornado.access").setLevel(logging.WARNING)
        server_process = Process(
            target=start_server, args=(args, ))
        server_process.start()
        time.sleep(3)
    else:
        server_process = None

    client = start_client(args)
    evaluate(args, client, server_process)


def server():
    parser = argparse.ArgumentParser()
    options.add_server_args(parser)
    options.add_data_args(parser)
    args = parser.parse_args()
    simuleval.online.start_server(args)
