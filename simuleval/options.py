# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
import os
from simuleval import DEFAULT_HOSTNAME, DEFAULT_PORT
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
        "--source-timestamps",
        type=str,
        default=None,
        help="Timesteps for generating the source words.",
    )
    parser.add_argument(
        "--eval-latency-unit",
        type=str,
        default="word",
        choices=["word", "char"],
        help="Basice unit used for latency caculation, choose from "
        "words (detokenized) and characters.",
    )
    parser.add_argument(
        "--sacrebleu-tokenizer",
        type=str,
        default="13a",
        choices=TOKENIZERS.keys(),
        help="Tokenizer for sacrebleu.",
    )
    parser.add_argument(
        "--no-space",
        action="store_true",
        help="No space is added between received words.",
    )
    parser.add_argument(
        "--latency-unit",
        default="word",
        help="No space is added between received words.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for evaluation",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="The last index for evaluation",
    )
    add_fairseq_data_args(parser)


def add_server_args(parser):
    parser.add_argument(
        "--hostname", type=str, default=DEFAULT_HOSTNAME, help="Server hostname"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Server port number"
    )
    parser.add_argument(
        "--server-only", action="store_true", help="Only start the server."
    )
    parser.add_argument(
        "--client-only", action="store_true", help="Only start the client"
    )


def general_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=None, help="Agent type")
    parser.add_argument(
        "--num-processes", type=int, default=1, help="Number of threads used by agent"
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Local evaluation"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        default=False,
        help="Do not use progress bar",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=[x.lower() for x in logging._levelToName.values()],
        help="Log level.",
    )
    return parser


def get_slurm_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--slurm", action="store_true", default=False, help="Use slurm."
    )
    parser.add_argument(
        "--slurm-partition", default="learnaccel,ust", help="Slurm partition."
    )
    parser.add_argument("--slurm-job-name", default="simuleval", help="Slurm job name.")
    parser.add_argument(
        "--slurm-time", default="3:00:00", help="Slurm partition."
    )
    args, _ = parser.parse_known_args()
    return args


def add_agent_args(parser, agent_cls):
    agent_cls.add_args(parser)
    parser.add_argument(
        "--max-gen-len", type=int, default=200, help="Max length of translation"
    )
