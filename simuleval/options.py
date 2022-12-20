# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
from simuleval.data.dataloader import DATALOADER_DICT, GenericDataloader


def add_dataloader_args(parser: argparse.ArgumentParser):
    args, _ = parser.parse_known_args()
    dataloader_class = DATALOADER_DICT.get(args.dataloader)
    if dataloader_class is None:
        dataloader_class = GenericDataloader
    dataloader_class.add_args(parser)


def add_evaluator_args(parser):
    parser.add_argument(
        "--quality-metrics",
        nargs="+",
        default=["BLEU"],
        help="Quality metrics",
    )
    parser.add_argument(
        "--latency-metrics",
        nargs="+",
        default=["AL", "AP", "DAL", "ATD"],
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
        "--eval-latency-unit",
        type=str,
        default="word",
        choices=["word", "char"],
        help="Basic unit used for latency calculation, choose from "
        "words (detokenized) and characters.",
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
    parser.add_argument("--output", type=str, default=None, help="Output directory")


def general_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--agent", default=None, help="Agent type")
    parser.add_argument("--dataloader", default=None, help="Dataloader to use")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=[x.lower() for x in logging._levelToName.values()],
        help="Log level.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        default=False,
        help="Only score the inference file.",
    )
    return parser


def add_slurm_args(parser):
    parser.add_argument(
        "--slurm-partition", default="learnaccel,ust", help="Slurm partition."
    )
    parser.add_argument("--slurm-job-name", default="simuleval", help="Slurm job name.")
    parser.add_argument("--slurm-time", default="10:00:00", help="Slurm partition.")
    args, _ = parser.parse_known_args()
