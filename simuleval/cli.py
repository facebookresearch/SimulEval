# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
import argparse
import time
import logging
import subprocess
import json
from functools import partial
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager

import simuleval
from simuleval import options
from simuleval import READ_ACTION, WRITE_ACTION
from simuleval.online import start_client, start_server
from simuleval.scorer.scorer import SentenceLevelScorer
from simuleval.utils.agent import find_agent_cls, infer_data_types_from_agent
from simuleval.utils.functional import split_list_into_chunks
from simuleval.data.dataloader import build_dataloader


logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger("simuleval.cli")


def mkdir_output_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except BaseException as be:
        logger.error(f"Failed to write results to {path}.")
        logger.error(be)
        logger.error("Skip writing predictions.")
        return False


class DataWriter(object):
    def __init__(self, args, q):
        self.queue = q
        self.output_dir = args.output
        if self.output_dir is None:
            logger.warning("No output directory")
            self.started = False
            self.proc = None
            return

        self.started = mkdir_output_dir(self.output_dir)
        if not self.started:
            return

        logger.info(f"Output dir: {self.output_dir}")
        path = os.path.join(self.output_dir, "instances.log")
        self.proc = Process(target=self.write_loop, args=(path, q))
        self.proc.start()
        self.started = True

    @staticmethod
    def write_loop(path, q):
        logger.info(f"Start data writer (process id {os.getpid()})")
        with open(path, "w") as f:
            while True:
                try:
                    m = q.get()
                    f.write(json.dumps(m) + "\n")
                    f.flush()
                except EOFError:
                    break

    def write_scores(self, scores):
        if self.started:
            with open(os.path.join(self.output_dir, "scores"), "w") as f:
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
    # build agents
    agent = agent_cls(args)
    agent.set_client(client)

    # Decode
    index_generator = instance_ids if args.no_progress_bar else tqdm(instance_ids)
    for instance_id in index_generator:
        agent.reset()
        agent.eval(index=instance_id)
        sent_info = client.get_scores(instance_id)
        result_queue.put(sent_info)
        logger.debug(
            f"Instance {instance_id} finished, results:\n{json.dumps(sent_info, indent=4)}"
        )


def evaluate(args, client, server_process=None):
    info = client.corpus_info()
    num_sentences = info["num_sentences"]
    if args.end_index < 0:
        args.end_index = num_sentences
    indices = list(range(num_sentences))[args.start_index : args.end_index]
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
    logger.setLevel(args.log_level.upper())

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
        agent_name, agent_cls = find_agent_cls(args)
        logger.info(f"Evaluating on agent {agent_name}")
        infer_data_types_from_agent(args, agent_cls)
        dataloader = build_dataloader(args)
        scorer = SentenceLevelScorer(dataloader, args)
        logging.getLogger("tornado.access").setLevel(logging.WARNING)
        server_process = Process(target=start_server, args=(args, scorer), daemon=True)
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


def submit_slurm_job(args: argparse.Namespace) -> None:
    assert mkdir_output_dir(args.output)
    os.system(f"cp {args.agent} {args.output}/agent.py")
    command = " ".join(sys.argv)
    command = re.sub(r"(--slurm\S*(\s+[^-]\S+)*)", "", command).strip()
    command = re.sub(
        r"--agent\s+\S+", f"--agent {args.output}/agent.py", command
    ).strip()
    command = command.replace("--", "\\\n\t--")
    script = f"""#!/bin/bash
#SBATCH --time={args.slurm_time}
#SBATCH --partition={args.slurm_partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output="{args.output}/slurm-%j.log"
#SBATCH --job-name="{args.slurm_job_name}"

cd {os.path.abspath(os.getcwd())}

CUDA_VISIBLE_DEVICES=$SLURM_LOCALID {command}
    """
    script_file = os.path.join(args.output, "script.sh")
    with open(script_file, "w") as f:
        f.writelines(script)

    process = subprocess.Popen(
        ["sbatch", script_file],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    logger.info("Using slurm.")
    logger.info(f"sbatch stdout: {stdout.decode('utf-8').strip()}")
    logger.info(f"sbatch stderr: {stderr.decode('utf-8').strip()}")


def main():
    args = options.get_slurm_args()
    if args.slurm:
        submit_slurm_job(args)
        return

    parser = options.general_parser()
    args, _ = parser.parse_known_args()

    options.add_server_args(parser)
    args, _ = parser.parse_known_args()
    logger.setLevel(args.log_level.upper())

    if not args.server_only:
        _main(args.client_only)
    else:
        server()
