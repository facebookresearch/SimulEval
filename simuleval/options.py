# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import logging
import os
import sys
from typing import List, Optional

from simuleval.data.dataloader import (
    DATALOADER_DICT,
    GenericDataloader,
    register_dataloader_class,
)
from simuleval.evaluator.scorers import get_scorer_class


def add_dataloader_args(
    parser: argparse.ArgumentParser, cli_argument_list: Optional[List[str]] = None
):
    if cli_argument_list is None:
        args, _ = parser.parse_known_args()
    else:
        args, _ = parser.parse_known_args(cli_argument_list)

    if args.dataloader_class:
        dataloader_module = importlib.import_module(
            ".".join(args.dataloader_class.split(".")[:-1])
        )
        dataloader_class = getattr(
            dataloader_module, args.dataloader_class.split(".")[-1]
        )
        register_dataloader_class(args.dataloader, dataloader_class)

    dataloader_class = DATALOADER_DICT.get(args.dataloader)
    if dataloader_class is None:
        dataloader_class = GenericDataloader
    dataloader_class.add_args(parser)


def add_evaluator_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--quality-metrics",
        nargs="+",
        default=["BLEU"],
        help="Quality metrics",
    )
    parser.add_argument(
        "--latency-metrics",
        nargs="+",
        default=["LAAL", "AL", "AP", "DAL", "ATD"],
        help="Latency metrics",
    )
    parser.add_argument(
        "--continue-unfinished",
        action="store_true",
        default=False,
        help="Continue the experiments in output dir.",
    )
    parser.add_argument(
        "--computation-aware",
        action="store_true",
        default=False,
        help="Include computational latency.",
    )
    parser.add_argument(
        "--no-use-ref-len",
        action="store_true",
        default=False,
        help="Include computational latency.",
    )
    parser.add_argument(
        "--eval-latency-unit",
        type=str,
        default="word",
        choices=["word", "char", "spm"],
        help="Basic unit used for latency calculation, choose from "
        "words (detokenized) and characters.",
    )
    parser.add_argument(
        "--eval-latency-spm-model",
        type=str,
        default=None,
        help="Pass the spm model path if the eval_latency_unit is spm.",
    )
    parser.add_argument(
        "--remote-address",
        default="localhost",
        help="Address to client backend",
    )
    parser.add_argument(
        "--remote-port",
        default=12321,
        help="Port to client backend",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        default=False,
        help="Do not use progress bar",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for evaluation.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="The last index for evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Required if using iterable dataloader.",
    )


def add_scorer_args(
    parser: argparse.ArgumentParser, cli_argument_list: Optional[List[str]] = None
):
    if cli_argument_list is None:
        args, _ = parser.parse_known_args()
    else:
        args, _ = parser.parse_known_args(cli_argument_list)

    for metric in args.latency_metrics:
        get_scorer_class("latency", metric).add_args(parser)

    for metric in args.quality_metrics:
        get_scorer_class("quality", metric).add_args(parser)


def import_user_module(module_path):
    module_path = os.path.abspath(module_path)
    module_parent, module_name = os.path.split(module_path)

    sys.path.insert(0, module_parent)
    importlib.import_module(module_name)
    sys.path.pop(0)


def general_parser(
    config_dict: Optional[dict] = None,
    parser: Optional[argparse.ArgumentParser] = None,
):
    if parser is None:
        parser = argparse.ArgumentParser(
            add_help=False,
            description="SimulEval - Simultaneous Evaluation CLI",
            conflict_handler="resolve",
        )

    parser.add_argument(
        "--user-dir",
        default=None,
        help="path to a python module containing custom agents",
    )
    args, _ = parser.parse_known_args()
    if args.user_dir is not None:
        import_user_module(args.user_dir)

    parser.add_argument(
        "--remote-eval",
        action="store_true",
        help="Evaluate a standalone agent",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--slurm", action="store_true", default=False, help="Use slurm."
    )
    parser.add_argument("--agent", default=None, help="Agent file")
    parser.add_argument(
        "--agent-class",
        default=None,
        help="The full string of class of the agent.",
    )
    parser.add_argument(
        "--system-dir",
        default=None,
        help="Directory that contains everything to start the simultaneous system.",
    )
    parser.add_argument(
        "--system-config",
        default="main.yaml",
        help="Name of the config yaml of the system configs.",
    )
    parser.add_argument("--dataloader", default=None, help="Dataloader to use")
    parser.add_argument(
        "--dataloader-class", default=None, help="Dataloader class to use"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=[x.lower() for x in logging._levelToName.values()],
        help="Log level.",
    )
    scoring_arg_group = parser.add_mutually_exclusive_group()
    scoring_arg_group.add_argument(
        "--score-only",
        action="store_true",
        default=False,
        help="Only score the inference file.",
    )
    scoring_arg_group.add_argument(
        "--no-scoring",
        action="store_true",
        help="No scoring after inference",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model."
    )
    dtype_arg_group = parser.add_mutually_exclusive_group()
    dtype_arg_group.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        type=str,
        help=(
            "Choose between half-precision (fp16) and single precision (fp32) floating point formats."
            + " Prefer this over the fp16 flag."
        ),
    )
    dtype_arg_group.add_argument(
        "--fp16", action="store_true", default=False, help="Use fp16."
    )

    return parser


def add_slurm_args(parser):
    parser.add_argument("--slurm-partition", default="", help="Slurm partition.")
    parser.add_argument("--slurm-job-name", default="simuleval", help="Slurm job name.")
    parser.add_argument("--slurm-time", default="2:00:00", help="Slurm partition.")
