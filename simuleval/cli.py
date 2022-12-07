# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
import argparse
import logging
import subprocess
import json
import multiprocessing
from simuleval import options
from simuleval import options, EVALUATION_SYSTEM_LIST
from simuleval.utils.agent import import_file
from simuleval.evaluator import (
    build_evaluator,
    build_remote_evaluator,
    SentenceLevelEvaluator,
)
from simuleval.agents.service import start_agent_service


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


def submit_slurm_job() -> None:
    args = options.get_slurm_args()
    assert mkdir_output_dir(args.output)
    os.system(f"cp {args.agent} {args.output}/agent.py")
    _args = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if str(arg).isdigit() or str(arg).startswith("--"):
            _args.append(arg)
        else:
            _args.append(f'"{arg}"')
    command = " ".join(_args)
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
    stderr = stderr.decode("utf-8").strip()
    if len(stderr) > 0:
        logger.info(f"sbatch stderr: {stderr.decode('utf-8').strip()}")


def check_evaluation_system_list():
    if len(EVALUATION_SYSTEM_LIST) == 0:
        logger.error(
            "Please use @simuleval decorator to indicate the system you want to evaluate."
        )
    elif len(EVALUATION_SYSTEM_LIST) > 1:
        logger.error("More than on system is not supported right now.")
    else:
        logger.info(f"Evaluate system: {EVALUATION_SYSTEM_LIST[0].__name__}")


def check_argument(name):
    parser = options.general_parser()
    args, _ = parser.parse_known_args()
    return getattr(args, name)


def import_user_system():
    import_file(check_argument("agent"))


def main():
    if check_argument("remote_eval"):
        remote_evaluate()
        return

    if check_argument("score_only"):
        scoring()
        return

    if check_argument("slurm"):
        submit_slurm_job()
        return

    system = build_system()

    if check_argument("standalone"):
        start_agent_service(system)
    else:
        evaluate(system)


def build_system():

    import_user_system()

    check_evaluation_system_list()
    system_class = EVALUATION_SYSTEM_LIST[0]

    # General Options
    parser = options.general_parser()
    options.add_data_args(parser)
    options.add_evaluator_args(parser)

    # System Options
    system_class.add_args(parser)

    args = parser.parse_args()

    # build system
    system = system_class.from_args(args)
    return system


def evaluate(system):

    parser = options.general_parser()
    options.add_data_args(parser)
    options.add_evaluator_args(parser)
    system.add_args(parser)

    args = parser.parse_args()
    args.source_type = system.source_type
    args.target_type = system.target_type

    # build evaluator
    evaluator = build_evaluator(args)

    # evaluate system
    evaluator(system)


def scoring():
    args = options.get_evaluator_args()
    evaluator = SentenceLevelEvaluator.from_args(args)
    print(json.dumps(evaluator.results, indent=4))


def remote_evaluate():
    # build evaluator
    parser = options.general_parser()
    options.add_data_args(parser)
    options.add_evaluator_args(parser)
    args = parser.parse_args()
    evaluator = build_remote_evaluator(args)

    # evaluate system
    evaluator.remote_eval()


if __name__ == "__main__":
    main()
