# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from simuleval import options
from simuleval.utils.agent import import_file
from simuleval.utils.slurm import submit_slurm_job
from simuleval.evaluator import (
    build_evaluator,
    build_remote_evaluator,
    SentenceLevelEvaluator,
)
from simuleval.agents.service import start_agent_service

EVALUATION_SYSTEM_LIST = []

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)


logger = logging.getLogger("simuleval.cli")


def check_evaluation_system_list():
    if len(EVALUATION_SYSTEM_LIST) == 0:
        logger.error(
            "Please use @entrypoint decorator to indicate the system you want to evaluate."
        )
        sys.exit(1)
    if len(EVALUATION_SYSTEM_LIST) > 1:
        logger.error("More than one system is not supported right now.")
        sys.exit(1)


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
    options.add_dataloader_args(parser)
    options.add_evaluator_args(parser)

    # System Options
    system_class.add_args(parser)

    args = parser.parse_args()

    # build system
    system = system_class.from_args(args)
    logger.info(f"Evaluate system: {system}")
    return system


def evaluate(system):

    parser = options.general_parser()
    options.add_evaluator_args(parser)
    options.add_dataloader_args(parser)
    system.add_args(parser)

    args = parser.parse_args()
    args.source_type = system.source_type
    args.target_type = system.target_type

    # build evaluator
    evaluator = build_evaluator(args)

    # evaluate system
    evaluator(system)


def scoring():
    parser = options.general_parser()
    options.add_evaluator_args(parser)
    options.add_dataloader_args(parser)
    args = parser.parse_args()
    evaluator = SentenceLevelEvaluator.from_args(args)
    print(evaluator.results)


def remote_evaluate():
    # build evaluator
    parser = options.general_parser()
    options.add_evaluator_args(parser)
    args = parser.parse_args()
    evaluator = build_remote_evaluator(args)

    # evaluate system
    evaluator.remote_eval()


if __name__ == "__main__":
    main()
