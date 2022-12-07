# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
import os
from sacrebleu import TOKENIZERS


def add_fairseq_data_args(parser):
    """
    Fairseq related data options.
    Useful if you use fairseq based models.
    """
    parser.add_argument(
        "--fairseq-data", type=str, default=None, help="Set fairseq data root."
    )
    parser.add_argument(
        "--fairseq-config", type=str, default=None, help="Set fairseq data root."
    )
    parser.add_argument(
        "--fairseq-gen-subset",
        type=str,
        default=None,
        help="Subset to evaluate. Assume there is a gen_subset.tsv file in fairseq_root",
    )
    parser.add_argument(
        "--fairseq-manifest",
        type=str,
        default=None,
        help="Use fairseq manifest (tsv) format input",
    )


def add_data_args(parser):
    parser.add_argument(
        "--source",
        type=str,
        default=os.environ.get("SIMULEVAL_SOURCE", None),
        help="Source file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=os.environ.get("SIMULEVAL_SOURCE", None),
        help="Target file.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        choices=["text", "speech"],
        help="Source Data type to evaluate.",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        choices=["text", "speech"],
        help="Data type to evaluate.",
    )
    parser.add_argument(
        "--source-segment-size",
        type=int,
        default=1,
        help="Source segment size, For text the unit is # token, for speech is ms",
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
    add_fairseq_data_args(parser)
    return parser


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
        default=["AL", "AP", "DAL"],
        help="Latency metrics",
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


def get_slurm_args():
    parser = general_parser()
    parser.add_argument(
        "--slurm-partition", default="learnaccel,ust", help="Slurm partition."
    )
    parser.add_argument("--slurm-job-name", default="simuleval", help="Slurm job name.")
    parser.add_argument("--slurm-time", default="10:00:00", help="Slurm partition.")
    args, _ = parser.parse_known_args()
    return args
