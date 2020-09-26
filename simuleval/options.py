# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from simuleval import DEFAULT_HOSTNAME, DEFAULT_PORT


def add_data_args(parser):
    parser.add_argument('--data-type', type=str, choices=["text", "speech"],
                        default=os.environ.get("SIMULEVAL_DATATYPE", None),
                        help='Data type to evaluate')
    parser.add_argument('--source', type=str, default=os.environ.get("SIMULEVAL_SOURCE", None),
                        help='Source file')
    parser.add_argument('--target', type=str, default=os.environ.get("SIMULEVAL_SOURCE", None),
                        help='Target file')


def add_server_args(parser):
    parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                        help='Server hostname')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='Server port number')


def general_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default=None,
                        help='Agent type')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='Number of threads used by agent')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Local evaluation')
    parser.add_argument('--slurm', action="store_true", default=False,
                        help='Local evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    return parser


def add_agent_args(parser, agent_cls):
    agent_cls.add_args(parser)
